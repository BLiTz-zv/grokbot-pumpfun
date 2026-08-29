"""Состояние, которое обязано пережить рестарт.

Процесс, торгующий сутками, будет перезапущен: деплой, OOM, ребут хоста.
Если после рестарта он забудет открытые позиции и счётчики дня, то купит
то же самое второй раз и превысит дневной лимит убытка — оба лимита
считаются от нуля.

Файл пишется атомарно: сначала во временный, потом os.replace. Так на
диске никогда не лежит наполовину записанный JSON, даже если процесс
убили в момент сохранения.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import Position

log = logging.getLogger(__name__)

STATE_VERSION = 1


class PipelineState(BaseModel):
    """Снимок всего, что нельзя потерять."""

    version: int = STATE_VERSION
    day: str = ""                                   # сутки UTC, к которым относятся счётчики
    trades_today: int = 0
    realized_pnl_sol: float = 0.0
    grok_calls_today: int = 0
    losing_streak: int = 0
    cooldown_until: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    updated_at: float = 0.0

    @property
    def is_empty(self) -> bool:
        return not self.positions and not self.trades_today and not self.realized_pnl_sol


class StateStore:
    """Атомарное чтение и запись состояния в один JSON-файл."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> PipelineState | None:
        """Прочитать состояние. None, если файла нет или он испорчен."""
        if not self.path.exists():
            return None
        try:
            raw: Any = json.loads(self.path.read_text(encoding="utf-8"))
            state = PipelineState.model_validate(raw)
        except Exception as exc:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            log.error(
                "состояние %s не читается (%s) — отложено в %s, стартуем с чистого",
                self.path, exc, backup,
            )
            with contextlib.suppress(OSError):
                os.replace(self.path, backup)
            return None

        if state.version != STATE_VERSION:
            log.warning("состояние версии %d, ожидалась %d — счётчики дня сброшены",
                        state.version, STATE_VERSION)
        return state

    def save(self, state: PipelineState) -> None:
        """Записать состояние атомарно. Сбой записи не роняет торговлю."""
        state.updated_at = time.time()
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(state.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except OSError as exc:
            log.error("не удалось сохранить состояние в %s: %s", self.path, exc)
            tmp.unlink(missing_ok=True)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class InstanceLock:
    """Замок «один бот на одно состояние».

    Два процесса на одном файле состояния — это два бота на одном кошельке:
    они перезапишут позиции друг друга, дважды выберут дневной лимит и
    купят один и тот же токен. Замок стоит копейки, а спасает от сценария,
    который иначе обнаруживается по деньгам.

    Замок от мёртвого процесса перехватывается: падение не должно
    оставлять систему незапускаемой.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(str(path) + ".lock")
        self.acquired = False

    def _holder(self) -> int | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return int(data.get("pid") or 0) or None
        except (OSError, ValueError, TypeError):
            return None

    @staticmethod
    def _alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:      # процесс есть, но чужой
            return True
        return True

    def acquire(self) -> bool:
        """Занять замок. False, если его держит живой процесс.

        Пишем через O_EXCL: два процесса, одновременно увидевшие пустой
        файл, не должны оба решить, что замок их.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            pid = self._holder()
            if pid and pid != os.getpid() and self._alive(pid):
                log.error("состояние %s уже занято процессом %d — второй бот на том "
                          "же кошельке не запускается", self.path.stem, pid)
                return False
            if self.path.exists():
                if pid and not self._alive(pid):
                    log.warning("замок остался от мёртвого процесса %d, перехватываем", pid)
                elif pid == os.getpid():
                    self.acquired = True
                    return True
                with contextlib.suppress(OSError):
                    self.path.unlink()
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            except OSError as exc:
                log.error("не удалось занять замок %s: %s", self.path, exc)
                return False
            try:
                os.write(
                    fd,
                    json.dumps({"pid": os.getpid(), "started": time.time()}).encode(),
                )
            except OSError as exc:
                log.error("не удалось занять замок %s: %s", self.path, exc)
                with contextlib.suppress(OSError):
                    os.close(fd)
                    self.path.unlink()
                return False
            os.close(fd)
            self.acquired = True
            return True
        log.error("не удалось занять замок %s: гонка с другим процессом", self.path)
        return False

    def release(self) -> None:
        if not self.acquired:
            return
        if self._holder() == os.getpid():
            with contextlib.suppress(OSError):
                self.path.unlink()
        self.acquired = False

    def __enter__(self) -> InstanceLock:
        if not self.acquire():
            raise RuntimeError(f"состояние занято другим процессом: {self.path}")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


def describe(state: PipelineState) -> str:
    """Строка для стартового лога."""
    age = max(0.0, time.time() - state.updated_at) / 60 if state.updated_at else 0.0
    return (
        f"день {state.day or '?'}, сделок {state.trades_today}, "
        f"PnL {state.realized_pnl_sol:+.4f} SOL, "
        f"открытых позиций {len(state.positions)}, "
        f"вызовов Grok {state.grok_calls_today}, "
        f"записано {age:.0f} мин назад"
    )
