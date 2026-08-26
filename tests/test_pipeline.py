"""Сквозной прогон пайплайна в dry-run на замоканном транспорте.

Проверяет проводку: что ступени идут в нужном порядке, что отказ на любой
из них пишется в лог с указанием ступени, и что в dry-run никакая
транзакция не отправляется.
"""

import json
import time

import httpx
import pytest

from src.log import read_log
from src.models import Config, Token
from src.pipeline import Pipeline, load_and_check, parse_args


def grok_handler(responses: dict[str, str]):
    """Отвечает разным JSON в зависимости от системного промпта агента."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system = body["messages"][0]["content"]
        for marker, content in responses.items():
            if marker in system:
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        raise AssertionError(f"неожиданный промпт: {system[:60]}")

    return handler


GOOD_AUDIT = json.dumps({
    "coordinated_buying": False, "wash_trading": False, "creator_dump_prep": False,
    "bundled_launch": False, "organic_buyer_share": 0.95, "confidence": 0.9,
    "flags": [], "reasoning": "чисто",
})
GOOD_NARRATIVE = json.dumps({
    "trend_fit": 0.9, "virality": 0.9, "community_signals": 0.9,
    "launch_timing": 0.9, "reasoning": "живой мем",
})
GOOD_TIMING = json.dumps({
    "market_sentiment": 0.9, "meme_season": 0.9, "volume_level": 0.9,
    "anomalies": [], "reasoning": "фон хороший",
})
APPROVE = json.dumps({"approve": True, "reason": "ок", "flags": [], "confidence": 0.9})
REJECT = json.dumps({"approve": False, "reason": "органика не бьётся",
                     "flags": ["contradiction"], "confidence": 0.9})


def data_handler(request: httpx.Request) -> httpx.Response:
    """Провайдер данных: холдеры, сделки, карточка токена."""
    path = request.url.path
    if path.endswith("/holders"):
        return httpx.Response(200, json=[
            {"address": f"h{i}", "share": 0.02, "amount": 1000} for i in range(20)
        ])
    if "/trades/all/" in path:
        base = time.time() - 600
        return httpx.Response(200, json=[
            {"user": f"w{i}", "txType": "buy", "solAmount": 0.3 + i * 0.02,
             "timestamp": base + i * 20, "signature": f"s{i}"}
            for i in range(30)
        ])
    return httpx.Response(200, json={
        "description": "милейший кот интернета",
        "twitter": "https://x.com/cat", "telegram": "https://t.me/cat",
        "website": "https://cat.fun",
        "virtual_sol_reserves": 30_000_000_000,
        "virtual_token_reserves": 900_000_000_000_000,
    })


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.mode = "dry-run"
    cfg.grok.api_key = "test"
    cfg.grok.retry_base_delay = 0.0
    cfg.logging.path = str(tmp_path / "trades.jsonl")
    cfg.filter.min_total_score = 0.65
    return cfg


def wire(pipeline: Pipeline, checker_answer: str) -> None:
    """Подменить весь сетевой транспорт на моки."""
    grok = httpx.AsyncClient(transport=httpx.MockTransport(grok_handler({
        "форензик": GOOD_AUDIT,
        "мем-культуры": GOOD_NARRATIVE,
        "рыночного режима": GOOD_TIMING,
        "риск-офицер": checker_answer,
    })))
    for agent in (pipeline.auditor, pipeline.narrative, pipeline.timing, pipeline.checker):
        agent._client = grok
    data = httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(data_handler))
    pipeline.analyzer._client = data
    pipeline.executor._client = data


def fresh_token() -> Token:
    return Token(
        mint="Mint1111", name="Cat", symbol="CAT", image_uri="https://i",
        creator="Creator1", created_timestamp=time.time() - 600,
        unique_buyers=12, curve_progress=0.2, market_cap_sol=30.0,
    )


async def test_dry_run_buys_and_logs_full_context(config):
    pipeline = Pipeline(config)
    wire(pipeline, APPROVE)
    analysis = await pipeline.process(fresh_token())

    assert analysis is not None
    assert analysis.checker.approve
    assert pipeline.risk.open_count == 1

    records = list(read_log(config.logging.path))
    buys = [r for r in records if r["type"] == "buy"]
    assert len(buys) == 1
    buy = buys[0]
    assert buy["tx_hash"] == "dry_run"          # ни одной реальной транзакции
    assert buy["mode"] == "dry-run"
    assert buy["scores"]["total"] >= config.filter.min_total_score
    assert buy["audit"]["organic_buyer_share"] == 0.95
    assert buy["narrative"] and buy["timing"] and buy["checker"]
    assert buy["metrics"]["trade_count"] == 30
    assert buy["entry_price"] > 0


async def test_checker_veto_stops_the_buy(config):
    pipeline = Pipeline(config)
    wire(pipeline, REJECT)
    assert await pipeline.process(fresh_token()) is None
    assert pipeline.risk.open_count == 0

    records = list(read_log(config.logging.path))
    assert [r["type"] for r in records] == ["skip"]
    assert records[0]["stage"] == "checker"
    assert "contradiction" in records[0]["detail"]


async def test_risk_gate_stops_the_buy(config):
    config.risk.max_open_positions = 0
    pipeline = Pipeline(config)
    wire(pipeline, APPROVE)
    assert await pipeline.process(fresh_token()) is None

    records = list(read_log(config.logging.path))
    assert records[-1]["stage"] == "risk"
    assert records[-1]["reason"].startswith("max_open_positions")


async def test_high_threshold_stops_before_checker(config):
    """Скоринговый порог экономит вызов сильной модели: чекер отвечать не должен."""
    config.filter.min_total_score = 0.99
    pipeline = Pipeline(config)
    wire(pipeline, APPROVE)
    pipeline.checker._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: (_ for _ in ()).throw(AssertionError("чекер вызван зря"))
        )
    )
    assert await pipeline.process(fresh_token()) is None
    records = list(read_log(config.logging.path))
    assert records[-1]["stage"] == "scoring"
    assert "слабее всего" in records[-1]["detail"]


async def test_stop_loss_closes_position_and_logs_pnl(config):
    pipeline = Pipeline(config)
    wire(pipeline, APPROVE)
    await pipeline.process(fresh_token())
    position = pipeline.risk.positions["Mint1111"]

    await pipeline._sell(position, price=position.entry_price * 0.5)

    assert pipeline.risk.open_count == 0
    closes = [r for r in read_log(config.logging.path) if r["type"] == "close"]
    assert len(closes) == 1
    assert closes[0]["reason"] == "stop_loss"
    assert closes[0]["tx_hash"] == "dry_run"


# --- защита режима live ---------------------------------------------------


def test_live_without_flag_refuses(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: live\n")
    with pytest.raises(SystemExit) as exc:
        load_and_check(parse_args(["--config", str(cfg)]))
    assert "--i-understand-the-risk" in str(exc.value)


def test_live_with_flag_allowed(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: live\n")
    config = load_and_check(parse_args(["--config", str(cfg), "--i-understand-the-risk"]))
    assert config.is_live


def test_missing_config_refuses(tmp_path):
    with pytest.raises(SystemExit):
        load_and_check(parse_args(["--config", str(tmp_path / "нет.yaml")]))


def test_dry_run_needs_no_flag(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("mode: dry-run\n")
    assert not load_and_check(parse_args(["--config", str(cfg)])).is_live
