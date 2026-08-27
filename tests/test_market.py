"""Пульс рынка: агент-тайминг должен получать измеренное, а не выдуманное."""

import pytest

from src.market import MarketPulse

NOW = 1_800_000_000.0


def pulse(window: float = 900.0) -> MarketPulse:
    return MarketPulse(window_seconds=window, outcome_memory=5)


# --- окно наблюдения ------------------------------------------------------


def test_empty_pulse_reports_zeros():
    snapshot = pulse().snapshot(now=NOW)
    assert snapshot["лончей_в_минуту"] == 0.0
    assert snapshot["доля_доживших_до_разбора"] == 0.0
    assert snapshot["закрытых_сделок_в_памяти"] == 0
    assert "доля_прибыльных" not in snapshot      # не выдумываем статистику из ничего


def test_launch_rate_is_per_minute():
    market = pulse(window=600.0)                  # окно 10 минут
    for index in range(30):
        market.record_launch(35.0, now=NOW - 500 + index)
    assert market.snapshot(now=NOW)["лончей_в_минуту"] == pytest.approx(3.0)


def test_old_launches_leave_the_window():
    market = pulse(window=600.0)
    market.record_launch(35.0, now=NOW - 5000)    # давно
    market.record_launch(35.0, now=NOW - 100)
    snapshot = market.snapshot(now=NOW)
    assert snapshot["лончей_в_окне"] == 1


def test_pass_share_counted():
    market = pulse()
    for index in range(20):
        market.record_launch(35.0, now=NOW - 100 + index)
    for index in range(4):
        market.record_passed(now=NOW - 50 + index)
    assert market.snapshot(now=NOW)["доля_доживших_до_разбора"] == pytest.approx(0.2)


def test_median_liquidity_reported():
    market = pulse()
    for sol in (31.0, 35.0, 40.0, 100.0):
        market.record_launch(sol, now=NOW - 10)
    assert market.snapshot(now=NOW)["медиана_sol_в_кривой"] == pytest.approx(37.5)


def test_zero_liquidity_launches_ignored_in_median():
    market = pulse()
    market.record_launch(0.0, now=NOW)
    market.record_launch(40.0, now=NOW)
    assert market.snapshot(now=NOW)["медиана_sol_в_кривой"] == pytest.approx(40.0)


def test_thin_stream_is_flagged():
    market = pulse()
    assert market.is_thin()
    for index in range(5):
        market.record_launch(35.0, now=NOW - index)
    assert not market.is_thin()


def test_hour_of_day_included():
    assert 0 <= pulse().snapshot(now=NOW)["час_utc"] <= 23


# --- исходы сделок --------------------------------------------------------


def test_outcomes_summarized():
    market = pulse()
    for pct in (150.0, -30.0, 80.0, -90.0):
        market.record_outcome(pct)
    snapshot = market.snapshot(now=NOW)
    assert snapshot["закрытых_сделок_в_памяти"] == 4
    assert snapshot["доля_прибыльных"] == pytest.approx(0.5)
    assert snapshot["медиана_результата_pct"] == pytest.approx(25.0)
    assert snapshot["доля_сливов"] == pytest.approx(0.25)      # только -90


def test_outcome_memory_is_bounded():
    market = pulse()                              # память на 5 исходов
    for pct in range(20):
        market.record_outcome(float(pct))
    assert market.snapshot(now=NOW)["закрытых_сделок_в_памяти"] == 5


def test_rug_threshold_respected():
    market = pulse()
    market.record_outcome(-55.0, rug_loss_pct=60.0)
    market.record_outcome(-65.0, rug_loss_pct=60.0)
    assert market.snapshot(now=NOW)["доля_сливов"] == pytest.approx(0.5)


# --- подъём из лога -------------------------------------------------------


def test_seed_from_log_restores_memory():
    market = pulse()
    records = [
        {"type": "buy", "mint": "A"},
        {"type": "close", "mint": "A", "pnl_pct": 120.0, "final": True},
        {"type": "close", "mint": "B", "pnl_pct": -80.0, "final": True},
        {"type": "skip", "mint": "C"},
    ]
    assert market.seed_from_log(records) == 2
    snapshot = market.snapshot(now=NOW)
    assert snapshot["доля_прибыльных"] == pytest.approx(0.5)
    assert snapshot["доля_сливов"] == pytest.approx(0.5)


def test_seed_ignores_partial_closes():
    """Частичная фиксация — не исход позиции: считать её отдельной сделкой
    значит удвоить статистику по каждому удачному входу."""
    market = pulse()
    records = [
        {"type": "close", "mint": "A", "pnl_pct": 120.0, "final": False},
        {"type": "close", "mint": "A", "pnl_pct": 40.0, "final": True},
    ]
    assert market.seed_from_log(records) == 1


def test_seed_takes_only_the_freshest():
    market = pulse()                              # память на 5
    records = [{"type": "close", "pnl_pct": float(i), "final": True} for i in range(20)]
    market.seed_from_log(records)
    assert market.snapshot(now=NOW)["медиана_результата_pct"] == pytest.approx(17.0)


def test_seed_survives_broken_records():
    market = pulse()
    records = [{"type": "close", "final": True}]  # без pnl_pct
    assert market.seed_from_log(records) == 1
    assert market.snapshot(now=NOW)["медиана_результата_pct"] == 0.0
