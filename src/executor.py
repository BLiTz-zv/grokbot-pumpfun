"""Исполнение сделок на Solana.

ЗАГЛУШКА ПО ЗАМЫСЛУ в части отправки транзакций: `LiveExecutor` поднимает
NotImplementedError, а рядом лежит список шагов, которые нужно дописать
руками. Код, подписывающий транзакции приватным ключом, здесь не
сгенерирован.

Всё остальное настоящее и, что важнее, честное: dry-run исполняется по
математике кривой из `curve.py` — с комиссией, с проскальзыванием и с
влиянием собственной заявки на цену. Раньше он покупал по цене котировки,
и dry-run показывал прибыль, которой в live не бывает.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, Field

from .curve import (
    TOTAL_SUPPLY,
    CurveState,
    buy_quote,
    price_from_reserves,
    sell_quote,
    state_from_any,
)
from .models import Config, Position, Token

log = logging.getLogger(__name__)

DRY_RUN_TX = "dry_run"

__all__ = [
    "TOTAL_SUPPLY",
    "BaseExecutor",
    "DryRunExecutor",
    "ExecutionResult",
    "LiveExecutor",
    "build_executor",
    "new_position",
    "price_from_reserves",
]


class ExecutionResult(BaseModel):
    """Итог попытки исполнения."""

    ok: bool
    tx_hash: str = ""
    price: float = 0.0           # средняя цена исполнения, а не котировка
    token_amount: float = 0.0
    sol_amount: float = 0.0
    fee_sol: float = 0.0
    impact_pct: float = 0.0
    error: str = ""
    state_after: CurveState | None = Field(default=None)


class BaseExecutor:
    """Общая часть: котировки, состояние кривой, расчёт исполнения."""

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self.market = config.market
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> BaseExecutor:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.data.rest_url,
                timeout=self.config.data.request_timeout,
            )
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _coin(self, mint: str) -> dict[str, Any]:
        if self._client is None:
            return {}
        try:
            resp = await self._client.get(f"/coins/{mint}")
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            log.warning("данные по %s недоступны: %s", mint, exc)
            return {}
        return data if isinstance(data, dict) else {}

    async def curve(self, mint: str, market_cap_sol: float = 0.0) -> CurveState | None:
        """Состояние кривой сейчас. None, если восстановить не из чего."""
        return state_from_any(await self._coin(mint), market_cap_sol)

    async def price(self, mint: str) -> float:
        """Спотовая цена. Ею меряются правила выхода — они про движение рынка,
        а не про исполнение конкретной заявки."""
        state = await self.curve(mint)
        return state.spot_price if state else 0.0

    async def buy(self, token: Token, size_sol: float) -> ExecutionResult:
        raise NotImplementedError

    async def sell(self, position: Position, fraction: float = 1.0) -> ExecutionResult:
        raise NotImplementedError

    # -- расчёт, общий для обоих режимов ----------------------------------

    def plan_buy(self, state: CurveState, size_sol: float) -> ExecutionResult:
        quote = buy_quote(state, size_sol, self.market.trade_fee_pct)
        if not quote.ok:
            return ExecutionResult(ok=False, error=quote.reason)
        if quote.impact_pct > self.market.max_price_impact_pct:
            return ExecutionResult(
                ok=False,
                error=(f"влияние на цену {quote.impact_pct:.2f}% выше "
                       f"потолка {self.market.max_price_impact_pct:.2f}%"),
                impact_pct=quote.impact_pct,
            )
        return ExecutionResult(
            ok=True,
            price=quote.avg_price,
            token_amount=quote.tokens,
            sol_amount=size_sol,
            fee_sol=quote.fee_sol,
            impact_pct=quote.impact_pct,
            state_after=quote.state_after,
        )

    def plan_sell(self, state: CurveState, tokens: float) -> ExecutionResult:
        quote = sell_quote(state, tokens, self.market.trade_fee_pct)
        if not quote.ok:
            return ExecutionResult(ok=False, error=quote.reason)
        return ExecutionResult(
            ok=True,
            price=quote.avg_price,
            token_amount=tokens,
            sol_amount=quote.sol_out,
            fee_sol=quote.fee_sol,
            impact_pct=quote.impact_pct,
            state_after=quote.state_after,
        )

    @staticmethod
    def _portion(position: Position, fraction: float) -> float:
        """Сколько токенов продаём. Хвост меньше процента добираем целиком:
        оставлять пыль в позиции незачем, она только мешает учёту."""
        fraction = max(0.0, min(1.0, fraction))
        tokens = position.token_amount * fraction
        if position.token_amount - tokens < position.token_amount * 0.01:
            tokens = position.token_amount
        return tokens


class DryRunExecutor(BaseExecutor):
    """Проходит весь путь, кроме отправки транзакции."""

    async def buy(self, token: Token, size_sol: float) -> ExecutionResult:
        state = await self.curve(token.mint, token.market_cap_sol)
        if state is None:
            # Позиция с неизвестной ценой входа неуправляема: ни одно
            # правило выхода на ней не срабатывает.
            log.warning("покупка %s отменена: состояние кривой неизвестно", token.mint[:8])
            return ExecutionResult(ok=False, error="состояние кривой неизвестно")

        result = self.plan_buy(state, size_sol)
        if not result.ok:
            log.warning("покупка %s отменена: %s", token.mint[:8], result.error)
            return result

        result.tx_hash = DRY_RUN_TX
        log.info("[dry-run] куплено %s: %.4f SOL -> %.0f токенов по %.12f "
                 "(комиссия %.4f SOL, влияние %.2f%%)",
                 token.mint[:8], size_sol, result.token_amount, result.price,
                 result.fee_sol, result.impact_pct)
        return result

    async def sell(self, position: Position, fraction: float = 1.0) -> ExecutionResult:
        state = await self.curve(position.mint)
        if state is None:
            return ExecutionResult(ok=False, error="состояние кривой неизвестно")

        tokens = self._portion(position, fraction)
        result = self.plan_sell(state, tokens)
        if not result.ok:
            return result

        result.tx_hash = DRY_RUN_TX
        log.info("[dry-run] продано %s: %.0f токенов -> %.4f SOL по %.12f "
                 "(комиссия %.4f SOL, влияние %.2f%%)",
                 position.mint[:8], tokens, result.sol_amount, result.price,
                 result.fee_sol, result.impact_pct)
        return result


class LiveExecutor(BaseExecutor):
    """Реальная отправка транзакций. Намеренно не реализована.

    Расчёт заявки при этом уже готов: `plan_buy` и `plan_sell` дают и
    ожидаемое количество токенов, и среднюю цену, и влияние на цену —
    из них берутся `max_sol_cost` и `min_sol_output` с нужным допуском.
    """

    def __init__(self, config: Config, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(config, client)
        self.rpc_url = config.solana.rpc_url
        self.jito = config.solana.jito

    async def buy(self, token: Token, size_sol: float) -> ExecutionResult:
        # TODO(live): покупка на бондинговой кривой pump.fun.
        #  1. Загрузить Keypair из config.solana.wallet_private_key (solders.keypair).
        #  2. Получить аккаунты кривой: bonding_curve, associated_bonding_curve,
        #     global, fee_recipient — и создать ATA покупателя, если её нет.
        #  3. Взять расчёт из self.plan_buy(state, size_sol): ожидаемые токены
        #     и средняя цена уже посчитаны с комиссией и проскальзыванием.
        #     max_sol_cost = size_sol * (1 + допуск), допуск порядка 1-2%.
        #  4. Собрать инструкцию `buy` программы
        #     6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P и ComputeBudget:
        #     цену за юнит и лимит.
        #  5. При config.solana.jito.enabled — добавить перевод чаевых
        #     (jito.tip_lamports) на tip-аккаунт и отправить бандл на
        #     jito.block_engine_url; иначе send_transaction через RPC.
        #  6. Дождаться подтверждения, вернуть ExecutionResult с реальными
        #     tx_hash, ценой исполнения и полученным количеством токенов.
        #     Чаевые Jito вычесть из sol_amount: это тоже стоимость сделки.
        raise NotImplementedError(
            "LiveExecutor.buy не реализован намеренно: допишите отправку "
            "транзакций сами, прежде чем включать mode: live"
        )

    async def sell(self, position: Position, fraction: float = 1.0) -> ExecutionResult:
        # TODO(live): продажа. Та же схема, что и buy, но инструкция `sell`,
        #  min_sol_output из self.plan_sell(state, tokens) с допуском вниз,
        #  и закрытие ATA после полного выхода (при частичном — не закрывать).
        raise NotImplementedError(
            "LiveExecutor.sell не реализован намеренно: допишите отправку "
            "транзакций сами, прежде чем включать mode: live"
        )


def build_executor(config: Config, client: httpx.AsyncClient | None = None) -> BaseExecutor:
    """Исполнитель по режиму из конфига."""
    if config.is_live:
        log.warning("режим live: используется LiveExecutor")
        return LiveExecutor(config, client)
    return DryRunExecutor(config, client)


def new_position(token: Token, result: ExecutionResult, score: float) -> Position:
    return Position(
        mint=token.mint,
        symbol=token.symbol,
        creator=token.creator,
        entry_price=result.price,
        peak_price=result.price,
        sol_spent=result.sol_amount,
        token_amount=result.token_amount,
        opened_at=time.time(),
        tx_hash=result.tx_hash,
        score=score,
    )
