"""Forward session purposes, temporal ordering and risk rules."""

from datetime import date, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quant_rsi.intraday.risk import RiskLimits, RiskState, evaluate
from quant_rsi.intraday.schedule import (
    ForwardSessionSchedule,
    TemporalOrderViolation,
    anchor_for,
    development_cutoff,
    researcher_view,
    validate_temporal_order,
)

ANCHOR = date(2026, 2, 9)


def schedule(**overrides):
    value = {
        "schema_version": 1,
        "rule": "block_4w_dev_4w_reserved_v1",
        "anchor_monday": ANCHOR,
        "development_weeks": 4,
        "reserved_weeks": 4,
        "embargo_sessions": 1,
    }
    value.update(overrides)
    return ForwardSessionSchedule.model_validate(value)


def weekdays(start, count):
    days, day = [], start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def test_purpose_follows_eight_week_epochs():
    s = schedule()
    assert s.purpose(ANCHOR) == "development_observation"
    assert s.purpose(ANCHOR + timedelta(weeks=3, days=4)) == "development_observation"
    assert s.purpose(ANCHOR + timedelta(weeks=4)) == "reserved_protected"
    assert s.purpose(ANCHOR + timedelta(weeks=7, days=4)) == "reserved_protected"
    assert s.purpose(ANCHOR + timedelta(weeks=8)) == "development_observation"
    with pytest.raises(ValueError, match="precedes"):
        s.purpose(ANCHOR - timedelta(days=1))


def test_release_block_overrides_and_anchor_must_be_monday():
    s = schedule(
        release_block={
            "first_session": ANCHOR + timedelta(weeks=1),
            "last_session": ANCHOR + timedelta(weeks=1, days=4),
        }
    )
    assert s.purpose(ANCHOR + timedelta(weeks=1, days=2)) == "reserved_release"
    with pytest.raises(ValidationError):
        schedule(anchor_monday=ANCHOR + timedelta(days=1))
    assert anchor_for(date(2026, 2, 12)) == ANCHOR


def test_reserved_blocks_are_contiguous_runs():
    days = weekdays(ANCHOR, 80)
    blocks = schedule().reserved_blocks(days)
    assert len(blocks) == 2 and len(blocks[0]) == 20
    assert blocks[0][0] == ANCHOR + timedelta(weeks=4)


def test_scoring_must_follow_every_used_session_plus_embargo():
    days = weekdays(ANCHOR, 60)
    validate_temporal_order(days, days[21:25], days[:20], 1)
    with pytest.raises(TemporalOrderViolation, match="embargo"):
        validate_temporal_order(days, days[20:25], days[:20], 1)
    with pytest.raises(TemporalOrderViolation):
        validate_temporal_order(days, days[10:15], days[:5] + days[40:45], 1)


def test_development_cutoff_excludes_later_development_weeks():
    s = schedule()
    days = weekdays(ANCHOR, 80)
    second_block = s.reserved_blocks(days)[1]
    allowed = development_cutoff(s, days, second_block)
    assert max(allowed) < min(second_block)
    assert all(s.purpose(d) == "development_observation" for d in allowed)
    validate_temporal_order(days, second_block, allowed, s.embargo_sessions)
    first_block = s.reserved_blocks(days)[0]
    later_dev = [
        d for d in days if d > max(first_block) and s.purpose(d) == "development_observation"
    ]
    with pytest.raises(TemporalOrderViolation):
        validate_temporal_order(days, first_block, later_dev, s.embargo_sessions)


def test_researcher_view_hides_reserved_outcomes():
    s = schedule()
    records = [
        {"session_id": ANCHOR.isoformat(), "equity": "100100", "integrity_status": "ok"},
        {
            "session_id": (ANCHOR + timedelta(weeks=5)).isoformat(),
            "equity": "99000",
            "integrity_status": "ok",
            "fills": 4,
        },
    ]
    view = researcher_view(s, records)
    assert view[0]["equity"] == "100100"
    assert view[1] == {
        "session_id": records[1]["session_id"],
        "integrity_status": "ok",
        "purpose": "reserved_protected",
    }


def test_risk_rules_report_every_triggered_reason():
    limits = RiskLimits(
        max_gross_notional="1000",
        max_position_notional="500",
        max_orders_per_minute=2,
        max_daily_loss_fraction="0.01",
        max_consecutive_rejects=2,
        max_clock_skew_seconds=2.0,
    )
    state = RiskState(
        gross_notional=Decimal(2000),
        largest_position_notional=Decimal(600),
        orders_last_minute=2,
        session_pnl=Decimal(-200),
        capital=Decimal(10000),
        consecutive_rejects=2,
        clock_skew_seconds=3,
        stale_feed=True,
        model_expired=True,
        in_restart_window=True,
    )
    assert set(evaluate(state, limits)) == {
        "max_gross_notional",
        "max_position_notional",
        "max_orders_per_minute",
        "daily_loss_stop",
        "consecutive_rejects",
        "clock_skew",
        "stale_feed",
        "model_expired",
        "planned_restart_window",
    }
    assert evaluate(RiskState(capital=Decimal(10000)), limits) == []
