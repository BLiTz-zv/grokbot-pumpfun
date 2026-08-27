"""Пульс рынка: то, что пайплайн видит своими глазами.

Агент-тайминг должен оценивать фон, но до этого модуля он получал на вход
внутренние счётчики пайплайна — сколько токенов в буфере да сколько
позиций открыто. По таким данным рыночный режим не определяется, и модель
не столько оценивала, сколько сочиняла.

Здесь собирается то, что известно достоверно, потому что мы это наблюдали
сами: сколько лончей в минуту идёт через сокет, какая доля доживает до
разбора, сколько SOL они успевают собрать, чем кончились наши последние
сделки. Ничего внешнего, никаких котировок BTC — только собственное окно
наблюдения, и агент прямо предупреждён, что видит именно его.

Все окна скользящие: у процесса, живущего сутками, статистика за всё
время бесполезна — она размазывает вчерашний штиль по сегодняшнему шторму.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

# Окно наблюдения за потоком лончей.
LAUNCH_WINDOW_SECONDS = 900.0

# Сколько последних закрытий учитывать в статистике исходов.
OUTCOME_MEMORY = 50


class MarketPulse:
    """Скользящая статистика потока лончей и собственных исходов."""

    def __init__(
        self,
        window_seconds: float = LAUNCH_WINDOW_SECONDS,
        outcome_memory: int = OUTCOME_MEMORY,
    ) -> None:
        self.window = window_seconds
        self.launches: deque[tuple[float, float]] = deque()   # (когда, SOL в кривой)
        self.passed: deque[float] = deque()                   # когда дошёл до разбора
        self.bought: deque[float] = deque()
        self.outcomes: deque[float] = deque(maxlen=outcome_memory)   # pnl_pct
        self.rugs: deque[float] = deque(maxlen=outcome_memory)       # 1.0 слив, 0.0 нет

    # -- вход --------------------------------------------------------------

    def record_launch(self, sol_in_curve: float = 0.0, now: float | None = None) -> None:
        self.launches.append((now or time.time(), max(0.0, sol_in_curve)))
        self._prune(now)

    def record_passed(self, now: float | None = None) -> None:
        """Лонч пережил фильтр монитора и пошёл в платный разбор."""
        self.passed.append(now or time.time())
        self._prune(now)

    def record_bought(self, now: float | None = None) -> None:
        self.bought.append(now or time.time())
        self._prune(now)

    def record_outcome(self, pnl_pct: float, rug_loss_pct: float = 60.0) -> None:
        self.outcomes.append(pnl_pct)
        self.rugs.append(1.0 if -pnl_pct >= rug_loss_pct else 0.0)

    def seed_from_log(self, records: Iterable[dict[str, Any]], rug_loss_pct: float = 60.0) -> int:
        """Поднять исходы из лога: после рестарта память не должна быть пустой."""
        closes = [r for r in records if r.get("type") == "close" and r.get("final", True)]
        memory = self.outcomes.maxlen or len(closes)
        for record in closes[-memory:]:
            self.record_outcome(float(record.get("pnl_pct") or 0.0), rug_loss_pct)
        return len(self.outcomes)

    def _prune(self, now: float | None = None) -> None:
        cutoff = (now or time.time()) - self.window
        while self.launches and self.launches[0][0] < cutoff:
            self.launches.popleft()
        while self.passed and self.passed[0] < cutoff:
            self.passed.popleft()
        while self.bought and self.bought[0] < cutoff:
            self.bought.popleft()

    # -- выход -------------------------------------------------------------

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        """Числа для агента. Все — из собственного окна наблюдения."""
        now = now or time.time()
        self._prune(now)
        minutes = self.window / 60.0

        sols = [sol for _, sol in self.launches if sol > 0]
        launches = len(self.launches)

        data: dict[str, Any] = {
            "окно_минут": round(minutes, 1),
            "час_utc": datetime.fromtimestamp(now, tz=UTC).hour,
            "лончей_в_минуту": round(launches / minutes, 2) if minutes else 0.0,
            "лончей_в_окне": launches,
            "доля_доживших_до_разбора": (
                round(len(self.passed) / launches, 4) if launches else 0.0
            ),
            "покупок_в_окне": len(self.bought),
            "медиана_sol_в_кривой": round(statistics.median(sols), 3) if sols else 0.0,
        }

        if self.outcomes:
            wins = [pct for pct in self.outcomes if pct > 0]
            data.update({
                "закрытых_сделок_в_памяти": len(self.outcomes),
                "доля_прибыльных": round(len(wins) / len(self.outcomes), 4),
                "медиана_результата_pct": round(statistics.median(self.outcomes), 2),
                "доля_сливов": round(sum(self.rugs) / len(self.rugs), 4),
            })
        else:
            data["закрытых_сделок_в_памяти"] = 0

        return data

    def is_thin(self) -> bool:
        """Поток почти пуст — оценивать по нему рынок нечестно."""
        return len(self.launches) < 5
