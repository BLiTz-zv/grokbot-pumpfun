"""Риск-менеджер: пять ограничителей и размер позиции.

Последний гейт перед исполнением, и единственный, который не спрашивает
Grok ни о чём. Все пороги — из конфига:

1. потолок SOL на сделку
2. дневной лимит убытка (достигнут — пайплайн стоит до следующего дня)
3. максимум сделок в день
4. максимум одновременно открытых позиций
5. стоп-лосс в процентах, мониторится фоновой задачей

Размер позиции пропорционален скорингу, но не выше потолка и не больше
30% остатка дневного лимита убытка.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from .models import Config, Position, RiskConfig, TradeDecision
from .state import PipelineState, StateStore, describe

log = logging.getLogger(__name__)

# Доля остатка дневного лимита, которой можно рискнуть в одной сделке.
MAX_SHARE_OF_REMAINING_BUDGET = 0.30

# Меньше этого объёма сделка не имеет смысла: съедят комиссии и чаевые.
MIN_TRADE_SOL = 0.01


class RiskManager:
    """Состояние дня, открытые позиции и решение о размере."""

    def __init__(
        self,
        config: Config,
        clock: Callable[[], float] = time.time,
        store: StateStore | None = None,
    ) -> None:
        self.config = config
        self.risk: RiskConfig = config.risk
        self.clock = clock
        self.store = store
        self.day = self._today()
        self.trades_today = 0
        self.realized_pnl_sol = 0.0          # отрицательное = убыток
        self.positions: dict[str, Position] = {}
        self.grok_calls_today = 0

    # -- состояние на диске ------------------------------------------------

    def restore(self) -> bool:
        """Поднять состояние с диска. True, если что-то восстановлено.

        Позиции восстанавливаются всегда: они реально открыты на цепочке,
        сколько бы времени ни прошло. Счётчики дня — только если файл от
        сегодняшних суток: чужой день своих лимитов нам не диктует.
        """
        if self.store is None:
            return False
        state = self.store.load()
        if state is None:
            return False

        self.positions = dict(state.positions)
        if state.day == self.day:
            self.trades_today = state.trades_today
            self.realized_pnl_sol = state.realized_pnl_sol
            self.grok_calls_today = state.grok_calls_today
        else:
            log.info("состояние от %s, сегодня %s — счётчики дня начинаем заново",
                     state.day or "?", self.day)
        log.info("состояние восстановлено: %s", describe(state))
        if self.halted:
            log.warning("после восстановления дневной лимит убытка уже выбран — "
                        "торговли сегодня не будет")
        return True

    def persist(self) -> None:
        """Сохранить состояние. Вызывается после каждого изменения денег."""
        if self.store is None:
            return
        self.store.save(
            PipelineState(
                day=self.day,
                trades_today=self.trades_today,
                realized_pnl_sol=self.realized_pnl_sol,
                grok_calls_today=self.grok_calls_today,
                positions=self.positions,
            )
        )

    # -- сутки -------------------------------------------------------------

    def _today(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def roll_day_if_needed(self) -> bool:
        """Новые сутки — обнулить счётчики. Открытые позиции не трогаем."""
        today = self._today()
        if today != self.day:
            log.info("новые сутки %s: счётчики сброшены (было сделок %d, PnL %.4f)",
                     today, self.trades_today, self.realized_pnl_sol)
            self.day = today
            self.trades_today = 0
            self.realized_pnl_sol = 0.0
            self.grok_calls_today = 0
            self.persist()
            return True
        return False

    # -- состояние ---------------------------------------------------------

    @property
    def daily_loss(self) -> float:
        """Убыток за сутки положительным числом. Прибыль -> 0."""
        return max(0.0, -self.realized_pnl_sol)

    @property
    def remaining_loss_budget(self) -> float:
        return max(0.0, self.risk.daily_loss_limit_sol - self.daily_loss)

    @property
    def halted(self) -> bool:
        """Дневной лимит убытка выбран — до конца суток не торгуем."""
        self.roll_day_if_needed()
        return self.daily_loss >= self.risk.daily_loss_limit_sol

    @property
    def open_count(self) -> int:
        return len(self.positions)

    # -- решение -----------------------------------------------------------

    def position_size(self, score: float) -> float:
        """Размер позиции: пропорционален скорингу, ограничен сверху дважды."""
        by_score = self.risk.max_sol_per_trade * max(0.0, min(1.0, score))
        by_budget = self.remaining_loss_budget * MAX_SHARE_OF_REMAINING_BUDGET
        return round(min(by_score, by_budget), 6)

    def evaluate(self, mint: str, score: float) -> TradeDecision:
        """Пропустить сделку или нет, и на какую сумму."""
        self.roll_day_if_needed()

        if self.halted:
            return TradeDecision(
                approved=False,
                reason=f"daily_loss_limit_hit ({self.daily_loss:.4f} SOL)",
            )
        if self.trades_today >= self.risk.max_trades_per_day:
            return TradeDecision(
                approved=False,
                reason=f"max_trades_per_day ({self.trades_today}/{self.risk.max_trades_per_day})",
            )
        if self.open_count >= self.risk.max_open_positions:
            return TradeDecision(
                approved=False,
                reason=f"max_open_positions ({self.open_count}/{self.risk.max_open_positions})",
            )
        if mint in self.positions:
            return TradeDecision(approved=False, reason="already_open")

        size = self.position_size(score)
        if size < MIN_TRADE_SOL:
            return TradeDecision(
                approved=False,
                reason=f"size_too_small ({size:.6f} SOL)",
                size_sol=size,
            )
        return TradeDecision(approved=True, size_sol=size, reason="ok")

    # -- учёт --------------------------------------------------------------

    def register_open(self, position: Position) -> None:
        self.roll_day_if_needed()
        self.positions[position.mint] = position
        self.trades_today += 1
        self.persist()

    def register_close(self, mint: str, pnl_sol: float) -> Position | None:
        position = self.positions.pop(mint, None)
        self.realized_pnl_sol += pnl_sol
        self.persist()
        if self.halted:
            log.warning("дневной лимит убытка выбран, торговля остановлена до %s",
                        "следующих суток UTC")
        return position

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "day": self.day,
            "trades_today": self.trades_today,
            "open_positions": self.open_count,
            "realized_pnl_sol": round(self.realized_pnl_sol, 6),
            "remaining_loss_budget": round(self.remaining_loss_budget, 6),
            "halted": self.halted,
            "grok_calls_today": self.grok_calls_today,
        }


# --------------------------------------------------------------------------
# Стоп-лосс
# --------------------------------------------------------------------------


def stop_loss_triggered(position: Position, price: float, stop_loss_pct: float) -> bool:
    if position.entry_price <= 0 or price <= 0:
        return False
    drawdown = (position.entry_price - price) / position.entry_price * 100.0
    return drawdown >= stop_loss_pct


class StopLossWatcher:
    """Фоновая задача: опрашивает цены открытых позиций и зовёт продажу.

    Цена и продажа приходят снаружи колбэками — сюда не тянется ни RPC, ни
    executor, поэтому это тестируется без сети.
    """

    def __init__(
        self,
        manager: RiskManager,
        price_fn: Callable[[str], Awaitable[float]],
        sell_fn: Callable[[Position, float], Awaitable[None]],
    ) -> None:
        self.manager = manager
        self.price_fn = price_fn
        self.sell_fn = sell_fn
        self._task: asyncio.Task | None = None

    async def check_once(self) -> list[str]:
        """Один проход по открытым позициям. Возвращает закрытые минты."""
        triggered: list[str] = []
        for position in list(self.manager.positions.values()):
            try:
                price = await self.price_fn(position.mint)
            except Exception as exc:
                log.warning("цена для %s недоступна: %s", position.mint, exc)
                continue
            if stop_loss_triggered(position, price, self.manager.risk.stop_loss_pct):
                log.info("стоп-лосс по %s: вход %.9f, сейчас %.9f",
                         position.mint, position.entry_price, price)
                await self.sell_fn(position, price)
                triggered.append(position.mint)
        return triggered

    async def run(self) -> None:
        interval = self.manager.risk.stop_loss_poll_seconds
        while True:
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("стоп-лосс упал на проходе: %s", exc)
            await asyncio.sleep(interval)

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self.run(), name="stop-loss-watcher")
        return self._task

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
