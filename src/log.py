"""JSONL-логирование. Одна запись — одна строка.

Три типа записей:
  buy   — покупка, с полным контекстом решения (скоринг, оценки всех
          агентов, метрики, цена входа);
  skip  — токен не прошёл, с указанием ступени, причины и детали;
  close — закрытие позиции с PnL и временем удержания.

Лог — единственный источник правды о том, что пайплайн делал и почему.
Скоринг пишется разложенным по компонентам: без этого потом не понять,
какой именно агент тянул решения вниз.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .models import Analysis, Config, Position, Scores, Token

log = logging.getLogger(__name__)


def setup_logging(config: Config) -> None:
    """Человекочитаемый лог в stderr. JSONL пишется отдельно, в файл."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-16s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


class TradeLog:
    """Аппендер JSONL. Каждая запись флашится сразу — процесс может умереть."""

    def __init__(self, path: str | Path, mode: str = "dry-run") -> None:
        self.path = Path(path)
        self.mode = mode
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- запись ------------------------------------------------------------

    def _write(self, record: dict[str, Any]) -> dict[str, Any]:
        record.setdefault("ts", time.time())
        record.setdefault("mode", self.mode)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def buy(
        self,
        analysis: Analysis,
        *,
        size_sol: float,
        entry_price: float,
        tx_hash: str,
    ) -> dict[str, Any]:
        token = analysis.token
        return self._write(
            {
                "type": "buy",
                "mint": token.mint,
                "symbol": token.symbol,
                "name": token.name,
                "size_sol": round(size_sol, 6),
                "entry_price": entry_price,
                "tx_hash": tx_hash,
                "scores": analysis.scores.model_dump(),
                "audit": analysis.audit.model_dump() if analysis.audit else None,
                "narrative": analysis.narrative.model_dump() if analysis.narrative else None,
                "timing": analysis.timing.model_dump() if analysis.timing else None,
                "checker": analysis.checker.model_dump() if analysis.checker else None,
                "metrics": analysis.metrics.model_dump(),
                "token": {
                    "age_seconds": round(token.age_seconds),
                    "unique_buyers": token.unique_buyers,
                    "curve_progress": round(token.curve_progress, 4),
                    "market_cap_sol": round(token.market_cap_sol, 4),
                    "creator": token.creator,
                },
            }
        )

    def skip(
        self,
        token: Token,
        *,
        stage: str,
        reason: str,
        detail: str | None = None,
        scores: Scores | None = None,
    ) -> dict[str, Any]:
        return self._write(
            {
                "type": "skip",
                "mint": token.mint,
                "symbol": token.symbol,
                "stage": stage,
                "reason": reason,
                "detail": detail,
                "scores": scores.model_dump() if scores else None,
            }
        )

    def close(
        self,
        position: Position,
        *,
        exit_price: float,
        pnl_sol: float,
        reason: str,
        tx_hash: str = "",
    ) -> dict[str, Any]:
        held = max(0.0, time.time() - position.opened_at) if position.opened_at else 0.0
        pnl_pct = (
            (exit_price - position.entry_price) / position.entry_price * 100.0
            if position.entry_price
            else 0.0
        )
        return self._write(
            {
                "type": "close",
                "mint": position.mint,
                "symbol": position.symbol,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "pnl_sol": round(pnl_sol, 6),
                "pnl_pct": round(pnl_pct, 2),
                "hold_seconds": round(held, 1),
                "reason": reason,
                "tx_hash": tx_hash,
                "score": position.score,
            }
        )

    # -- чтение ------------------------------------------------------------

    def read(self) -> Iterator[dict[str, Any]]:
        yield from read_log(self.path)


def read_log(path: str | Path) -> Iterator[dict[str, Any]]:
    """Построчное чтение JSONL. Битые строки пропускаются с предупреждением."""
    file = Path(path)
    if not file.exists():
        return
    with file.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("%s:%d — строка не разобралась, пропущена", file, number)
                continue
            if isinstance(record, dict):
                yield record
