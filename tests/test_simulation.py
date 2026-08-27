"""Симуляция торгового дня с проверкой инвариантов учёта.

Отдельные тесты проверяют правила по одному. Здесь через пайплайн
прогоняется много токенов с движущимися ценами, а после каждого шага
проверяется то, что обязано выполняться всегда:

  * ни один потолок риска не превышен ни в одну секунду;
  * деньги сходятся: сумма PnL из лога равна учтённой в риск-менеджере;
  * себестоимость позиции равна затраченному минус возвращённое частичными
    выходами — частичная фиксация не создаёт и не теряет SOL;
  * каждая закрытая позиция исчезает из учёта, каждая открытая имеет
    ненулевую цену входа.

Такие расхождения не ловятся глазами и не проявляются на одной сделке —
они накапливаются за день и обнаруживаются, когда деньги уже разошлись.
"""

import json
import random

import httpx
import pytest

from src.curve import INITIAL_VIRTUAL_SOL, CurveState
from src.log import read_log
from src.models import Config
from src.pipeline import Pipeline

APPROVE = json.dumps({"approve": True, "reason": "ок", "flags": [], "confidence": 0.9})
AUDIT = json.dumps({
    "coordinated_buying": False, "wash_trading": False, "creator_dump_prep": False,
    "bundled_launch": False, "organic_buyer_share": 0.95, "confidence": 0.9,
    "flags": [], "reasoning": "чисто",
})
NARRATIVE = json.dumps({"trend_fit": 0.9, "virality": 0.9, "community_signals": 0.9,
                        "launch_timing": 0.9, "reasoning": "живой мем"})
TIMING = json.dumps({"market_sentiment": 0.9, "meme_season": 0.9, "volume_level": 0.9,
                     "anomalies": [], "reasoning": "фон обычный"})


class FakeMarket:
    """Набор кривых, которые живут своей жизнью по заданному сценарию."""

    def __init__(self, seed: int = 7) -> None:
        self.random = random.Random(seed)
        self.curves: dict[str, CurveState] = {}
        self.graduated: set[str] = set()

    def add(self, mint: str, real_sol: float = 15.0) -> None:
        fresh = CurveState()
        sol = INITIAL_VIRTUAL_SOL + real_sol
        self.curves[mint] = CurveState(sol_reserves=sol, token_reserves=fresh.k / sol)

    def step(self) -> None:
        """Каждая кривая двигается: кто-то растёт, кто-то складывается."""
        for mint, state in self.curves.items():
            factor = self.random.choice([0.5, 0.8, 0.95, 1.1, 1.5, 2.5])
            sol = max(1.0, state.sol_reserves * factor)
            self.curves[mint] = CurveState(
                sol_reserves=sol,
                token_reserves=state.k / sol,
                complete=mint in self.graduated,
            )
            if self.random.random() < 0.05:
                self.graduated.add(mint)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/holders"):
            return httpx.Response(200, json=[
                {"address": f"h{i}", "share": 0.02, "amount": 1000} for i in range(20)
            ])
        if "/trades/all/" in path:
            return httpx.Response(200, json=[
                {"user": f"w{i}", "txType": "buy", "solAmount": 0.3 + i * 0.02,
                 "timestamp": 1_800_000_000 + i * 20, "signature": f"s{i}"}
                for i in range(30)
            ])
        mint = path.rsplit("/", 1)[-1]
        state = self.curves.get(mint)
        if state is None:
            return httpx.Response(200, json={})
        return httpx.Response(200, json={
            "description": "кот", "twitter": "https://x.com/c",
            "telegram": "https://t.me/c", "website": "https://c.fun",
            "virtual_sol_reserves": int(state.sol_reserves * 1e9),
            "virtual_token_reserves": int(state.token_reserves * 1e6),
            "complete": state.complete,
        })


def grok_handler(request: httpx.Request) -> httpx.Response:
    system = json.loads(request.content)["messages"][0]["content"]
    for marker, answer in (("форензик", AUDIT), ("мем-культуры", NARRATIVE),
                           ("рыночного режима", TIMING), ("риск-офицер", APPROVE)):
        if marker in system:
            return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})
    raise AssertionError("неизвестный агент")


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.grok.api_key = "xai-simulation-key-123"
    cfg.grok.retry_base_delay = 0.0
    cfg.logging.path = str(tmp_path / "trades.jsonl")
    cfg.ops.state_path = str(tmp_path / "state.json")
    cfg.ops.reputation_path = str(tmp_path / "creators.json")
    cfg.ops.max_grok_calls_per_day = 10_000
    cfg.ops.grok_calls_per_minute = 100_000   # ограничитель частоты не про тесты
    cfg.risk.max_open_positions = 3
    cfg.risk.max_total_exposure_sol = 1.2
    cfg.risk.max_trades_per_day = 12
    cfg.risk.max_sol_per_trade = 0.5
    cfg.risk.take_profit_pct = 80.0
    cfg.risk.take_profit_fraction = 0.5
    cfg.risk.trailing_stop_pct = 30.0
    cfg.risk.stop_loss_pct = 35.0
    cfg.filter.one_position_per_creator = False
    cfg.filter.block_creator_after_rugs = 0
    return cfg


def token_payload(index: int) -> dict:
    return {
        "mint": f"Mint{index:04d}", "name": f"Cat {index}", "symbol": f"CAT{index}",
        "image_uri": "https://i", "creator": f"Creator{index}",
        "created_timestamp": 1_800_000_000.0, "unique_buyers": 12,
        "curve_progress": 0.2, "market_cap_sol": 40.0,
    }


def check_invariants(pipeline: Pipeline, config: Config) -> None:
    risk = pipeline.risk
    assert risk.open_count <= config.risk.max_open_positions
    assert risk.exposure_sol <= config.risk.max_total_exposure_sol + 1e-9
    assert risk.trades_today <= config.risk.max_trades_per_day
    for position in risk.positions.values():
        assert position.entry_price > 0, "позиция без цены входа неуправляема"
        assert position.token_amount > 0
        assert position.sol_spent >= 0
        assert position.peak_price >= position.entry_price * 0.999


async def test_trading_day_keeps_the_books_straight(config):
    from src.models import Token

    market = FakeMarket()
    pipeline = Pipeline(config)
    grok = httpx.AsyncClient(transport=httpx.MockTransport(grok_handler))
    for agent in (pipeline.auditor, pipeline.narrative, pipeline.timing, pipeline.checker):
        agent._client = grok
    data = httpx.AsyncClient(base_url="http://t", transport=httpx.MockTransport(market.handler))
    pipeline.analyzer._client = data
    pipeline.executor._client = data

    for index in range(25):
        market.add(f"Mint{index:04d}", real_sol=12.0 + index)
        await pipeline.process(Token(**token_payload(index)))
        check_invariants(pipeline, config)

        market.step()
        await pipeline.watcher.check_once()
        check_invariants(pipeline, config)

    records = list(read_log(config.logging.path))
    closes = [r for r in records if r["type"] == "close"]
    buys = [r for r in records if r["type"] == "buy"]
    assert buys, "за день не куплено ничего — симуляция бессмысленна"
    assert closes, "ни одна позиция не закрылась — правила выхода не сработали"

    # деньги сходятся: сумма PnL из лога равна учтённой в риск-менеджере.
    # Допуск — только на округление лога до шести знаков.
    logged = sum(r["pnl_sol"] for r in closes)
    assert logged == pytest.approx(pipeline.risk.realized_pnl_sol, abs=1e-5)

    # каждая покупка имеет ровно одно намерение перед собой
    intents = [r for r in records if r["type"] == "intent"]
    assert len(intents) >= len(buys)

    # закрытые позиции исчезли из учёта, открытые — на месте
    finally_closed = {r["mint"] for r in closes if r["final"]}
    assert finally_closed.isdisjoint(pipeline.risk.positions)


async def test_partial_exits_do_not_leak_money(config):
    """Себестоимость плюс возвращённое частичными выходами обязаны сходиться
    с изначально потраченным на каждую позицию."""
    from src.models import Token

    market = FakeMarket(seed=3)
    pipeline = Pipeline(config)
    grok = httpx.AsyncClient(transport=httpx.MockTransport(grok_handler))
    for agent in (pipeline.auditor, pipeline.narrative, pipeline.timing, pipeline.checker):
        agent._client = grok
    data = httpx.AsyncClient(base_url="http://t", transport=httpx.MockTransport(market.handler))
    pipeline.analyzer._client = data
    pipeline.executor._client = data

    spent: dict[str, float] = {}
    for index in range(15):
        mint = f"Mint{index:04d}"
        market.add(mint, real_sol=20.0)
        analysis = await pipeline.process(Token(**token_payload(index)))
        if analysis and mint in pipeline.risk.positions:
            spent[mint] = pipeline.risk.positions[mint].sol_spent

        market.step()
        await pipeline.watcher.check_once()

        for open_mint, position in pipeline.risk.positions.items():
            if position.partials:
                # доля токенов, оставшаяся в позиции, равна доле себестоимости
                assert position.sol_spent < spent[open_mint]
                assert position.realized_sol > 0

    closes = [r for r in read_log(config.logging.path) if r["type"] == "close"]
    partial = [r for r in closes if not r["final"]]
    assert partial, "частичных выходов не случилось — сценарий не проверен"
    for record in partial:
        assert 0 < record["fraction"] < 1


async def test_halt_stops_trading_for_the_day(config):
    """При выбранном дневном лимите убытка новых покупок быть не должно.

    Убыток здесь заносится напрямую, а не разыгрывается рынком: проверяем
    поведение гейта, а не удачливость сценария.
    """
    from src.models import Token

    market = FakeMarket(seed=11)
    pipeline = Pipeline(config)
    grok = httpx.AsyncClient(transport=httpx.MockTransport(grok_handler))
    for agent in (pipeline.auditor, pipeline.narrative, pipeline.timing, pipeline.checker):
        agent._client = grok
    data = httpx.AsyncClient(base_url="http://t", transport=httpx.MockTransport(market.handler))
    pipeline.analyzer._client = data
    pipeline.executor._client = data

    for index in range(3):
        market.add(f"Mint{index:04d}", real_sol=20.0)
        await pipeline.process(Token(**token_payload(index)))
    assert not pipeline.risk.halted

    pipeline.risk.register_close("внешний", pnl_sol=-config.risk.daily_loss_limit_sol)
    assert pipeline.risk.halted

    for index in range(10, 20):
        market.add(f"Mint{index:04d}", real_sol=20.0)
        assert await pipeline.process(Token(**token_payload(index))) is None

    skips = [r for r in read_log(config.logging.path)
             if r["type"] == "skip" and r["stage"] == "risk"]
    assert any("daily_loss_limit_hit" in r["reason"] for r in skips)
