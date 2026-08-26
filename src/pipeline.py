"""Оркестратор: связывает все ступени в один поток.

    монитор → анализатор → аудитор → нарратив → тайминг → скоринг →
    чекер → риск-гейт → исполнение

Каждая ступень либо пропускает токен дальше, либо пишет skip с причиной и
на этом заканчивает. Дорогие ступени стоят после дешёвых: до grok-4
доходит только то, что пережило фильтр кодом, метрики, трёх быстрых
агентов и скоринговый порог.

Запуск:
    python -m src.pipeline --config config.yaml
    python -m src.pipeline --config config.yaml --i-understand-the-risk   # для live
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import httpx

from .agents import AuditorAgent, CheckerAgent, NarrativeAgent, TimingAgent
from .analyzer import Analyzer, compute_metrics, enrich_token
from .executor import BaseExecutor, build_executor, new_position
from .log import TradeLog, setup_logging
from .models import Analysis, Config, Position, Token
from .monitor import LaunchMonitor
from .risk import RiskManager, StopLossWatcher
from .scoring import compute_scores, passes_threshold, weakest_component

log = logging.getLogger("pipeline")

# Сколько токенов разбираем одновременно. Больше — упрёмся в лимиты Grok.
MAX_CONCURRENT_TOKENS = 4


class Pipeline:
    """Держит агентов, состояние риска и лог; гоняет токены по ступеням."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.trade_log = TradeLog(config.logging.path, mode=config.mode)
        self.risk = RiskManager(config)

        self._grok_client = httpx.AsyncClient(timeout=config.grok.timeout_seconds)
        self.auditor = AuditorAgent(config, self._grok_client)
        self.narrative = NarrativeAgent(config, self._grok_client)
        self.timing = TimingAgent(config, self._grok_client)
        self.checker = CheckerAgent(config, self._grok_client)

        self.analyzer = Analyzer(config)
        self.executor: BaseExecutor = build_executor(config)
        self.monitor = LaunchMonitor(config, on_skip=self._log_monitor_skip)
        self.watcher = StopLossWatcher(self.risk, self._price, self._sell)

        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_TOKENS)
        self._tasks: set[asyncio.Task] = set()

    # -- жизненный цикл ----------------------------------------------------

    async def __aenter__(self) -> Pipeline:
        await self.analyzer.__aenter__()
        await self.executor.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.watcher.stop()
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.analyzer.__aexit__(*exc)
        await self.executor.__aexit__(*exc)
        await self._grok_client.aclose()

    async def run(self) -> None:
        """Главный цикл. Живёт, пока не отменят."""
        self.watcher.start()
        log.info("пайплайн запущен, режим %s, лог %s",
                 self.config.mode, self.config.logging.path)
        async for token in self.monitor.stream():
            if self.risk.halted:
                self.trade_log.skip(token, stage="risk", reason="daily_loss_limit_hit")
                continue
            task = asyncio.create_task(self._guarded(token), name=f"token-{token.mint[:8]}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _guarded(self, token: Token) -> None:
        async with self._semaphore:
            try:
                await self.process(token)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("токен %s уронил обработку: %s", token.mint, exc)
                self.trade_log.skip(token, stage="pipeline", reason="internal_error", detail=str(exc))

    # -- ступени -----------------------------------------------------------

    async def process(self, token: Token) -> Analysis | None:
        """Один токен от метрик до покупки. None, если отсеян."""
        log.info("разбираем %s (%s), покупателей %d",
                 token.symbol or "?", token.mint[:8], token.unique_buyers)

        # 2. Анализатор: сеть параллельно, метрики кодом.
        info, holders, trades = await self.analyzer.fetch(token.mint)
        enrich_token(token, info)
        metrics = compute_metrics(token, holders, trades)
        analysis = Analysis(token=token, metrics=metrics)

        ok, reason = self.analyzer.passes(metrics)
        if not ok:
            self.trade_log.skip(token, stage="analyzer", reason=reason,
                                detail=f"risk_score={metrics.risk_score}")
            return None

        # 3-5. Быстрые агенты параллельно. Тайминг обычно берётся из кэша.
        analysis.audit, analysis.narrative, analysis.timing = await asyncio.gather(
            self.auditor.run(token, trades, holders, metrics),
            self.narrative.run(token),
            self.timing.get(self._market_snapshot()),
        )

        # 6. Скоринг кодом.
        analysis.scores = compute_scores(analysis, self.config)
        ok, reason = passes_threshold(analysis.scores, self.config)
        if not ok:
            name, value = weakest_component(analysis.scores)
            self.trade_log.skip(token, stage="scoring", reason=reason,
                                detail=f"слабее всего {name}={value:.3f}",
                                scores=analysis.scores)
            return None

        # 7. Адверсариальный чекер на сильной модели.
        analysis.checker = await self.checker.run(analysis)
        if not analysis.checker.approve:
            self.trade_log.skip(token, stage="checker",
                                reason="checker_rejected",
                                detail=f"{analysis.checker.reason} "
                                       f"[{', '.join(analysis.checker.flags)}]",
                                scores=analysis.scores)
            return None

        # 8. Риск-гейт.
        decision = self.risk.evaluate(token.mint, analysis.scores.total)
        if not decision.approved:
            self.trade_log.skip(token, stage="risk", reason=decision.reason,
                                scores=analysis.scores)
            return None

        # 9. Исполнение.
        result = await self.executor.buy(token, decision.size_sol)
        if not result.ok:
            self.trade_log.skip(token, stage="executor", reason="execution_failed",
                                detail=result.error, scores=analysis.scores)
            return None

        position = new_position(token, result, analysis.scores.total)
        self.risk.register_open(position)
        self.trade_log.buy(analysis, size_sol=decision.size_sol,
                           entry_price=result.price, tx_hash=result.tx_hash)
        log.info("КУПЛЕНО %s на %.4f SOL, score %.3f, tx %s",
                 token.symbol or token.mint[:8], decision.size_sol,
                 analysis.scores.total, result.tx_hash)
        return analysis

    # -- вспомогательное ---------------------------------------------------

    def _market_snapshot(self) -> dict[str, Any]:
        """Что пайплайн знает о рынке сам — уходит тайминг-агенту как контекст."""
        return {
            "pending_launches": len(self.monitor.pending),
            "open_positions": self.risk.open_count,
            "trades_today": self.risk.trades_today,
            "realized_pnl_sol": round(self.risk.realized_pnl_sol, 4),
        }

    def _log_monitor_skip(self, token: Token, reason: str) -> None:
        self.trade_log.skip(token, stage="monitor", reason=reason)

    async def _price(self, mint: str) -> float:
        return await self.executor.price(mint)

    async def _sell(self, position: Position, price: float) -> None:
        """Продажа по стоп-лоссу: закрыть, посчитать PnL, записать в лог."""
        result = await self.executor.sell(position)
        proceeds = result.sol_amount if result.ok else position.token_amount * price
        pnl = proceeds - position.sol_spent
        self.risk.register_close(position.mint, pnl_sol=pnl)
        self.trade_log.close(position, exit_price=result.price or price,
                             pnl_sol=pnl, reason="stop_loss", tx_hash=result.tx_hash)
        log.info("ЗАКРЫТО %s по стоп-лоссу, PnL %+.4f SOL", position.mint[:8], pnl)


# --------------------------------------------------------------------------
# Точка входа
# --------------------------------------------------------------------------


LIVE_WARNING = """
================================================================
  РЕЖИМ LIVE

  Пайплайн будет отправлять РЕАЛЬНЫЕ транзакции реальным кошельком
  из config.yaml. Мемкоины на бондинговой кривой теряют стоимость
  полностью и обычно. Потолок на сделку {max_sol} SOL, дневной лимит
  убытка {daily} SOL — это ограничители, а не гарантия.

  Запуск в live требует флага --i-understand-the-risk.
================================================================
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="grokbot-pumpfun")
    parser.add_argument("--config", default="config.yaml", help="путь к конфигу")
    parser.add_argument("--i-understand-the-risk", action="store_true",
                        help="обязателен для запуска в режиме live")
    return parser.parse_args(argv)


def load_and_check(args: argparse.Namespace) -> Config:
    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"Конфига {path} нет. Скопируйте config.example.yaml в config.yaml.")
    config = Config.load(path)

    if config.is_live:
        print(LIVE_WARNING.format(
            max_sol=config.risk.max_sol_per_trade,
            daily=config.risk.daily_loss_limit_sol,
        ), file=sys.stderr)
        if not getattr(args, "i_understand_the_risk", False):
            raise SystemExit(
                "Отказ: mode: live без флага --i-understand-the-risk. "
                "Либо верните mode: dry-run, либо подтвердите флагом."
            )
    return config


async def amain(argv: list[str] | None = None) -> int:
    config = load_and_check(parse_args(argv))
    setup_logging(config)
    async with Pipeline(config) as pipeline:
        try:
            await pipeline.run()
        except asyncio.CancelledError:
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(amain(argv))
    except KeyboardInterrupt:
        print("\nостановлено", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
