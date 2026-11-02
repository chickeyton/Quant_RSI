"""Fundamental (`f_`) intraday leaves: as-of-open values, binding and refusal rules."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from intraday_helpers import build_snapshot, plan

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.intraday.__main__ import main
from quant_rsi.intraday.features import FUNDAMENTAL_LEAVES, build_features, validate_leaves
from quant_rsi.intraday.fundamentals import (
    FundamentalSource,
    load_financial_snapshot,
    session_values,
    with_fundamentals,
)
from quant_rsi.research.study_financials import METRICS

UTC = timezone.utc
# Q4 2025 results are published after the close on 2026-01-20 (a Tuesday), so the first
# session that may use them is 2026-01-21; earlier sessions still see Q3 2025.
RELEASE = datetime(2026, 1, 20, 21, 30, tzinfo=UTC)
QUARTERS = [
    ("2024-07-01", "2024-09-30", "2024-10-22T12:00:00+00:00", 100.0, 20.0),
    ("2024-10-01", "2024-12-31", "2025-02-11T12:00:00+00:00", 110.0, 24.0),
    ("2025-07-01", "2025-09-30", "2025-10-21T12:00:00+00:00", 105.0, 22.0),
    ("2025-10-01", "2025-12-31", RELEASE.isoformat(), 121.0, 30.25),
]


def financial_snapshot(symbol="KO"):
    records = []
    for number, (start, end, available, revenue, income) in enumerate(QUARTERS):
        for field, value in (("revenue", revenue), ("operating_income", income)):
            records.append(
                {
                    "field": field,
                    "start": start,
                    "end": end,
                    "available_at": available,
                    "value": value,
                    "priority": 0,
                    "statement_id": f"release-{number}",
                    "unit": "USD",
                }
            )
    return {
        "symbol": symbol,
        "provider": "issuer_earnings",
        "records": records,
        "sha256": digest(canonical(records)),
        "notice": "Fixture financial releases; engineering only.",
    }


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    store, ref = build_snapshot(tmp_path_factory.mktemp("fundamentals"), count=30)
    return store.load(ref)


def fundamental_plan(financials, leaves=("ret_1", "f_revenue_yoy", "f_days_since_report")):
    return plan(
        leaves=list(leaves),
        fundamentals=[{"instrument": "KO", "snapshot_sha256": financials["sha256"]}],
    )


def test_values_change_only_at_the_first_open_after_publication(snapshot):
    sessions = snapshot.contract.calendar.sessions
    values = session_values(financial_snapshot(), sessions)
    before = values["2026-01-20"]  # opened before the after-close release
    after = values["2026-01-21"]
    assert before["f_revenue_yoy"] == pytest.approx(105 / 100 - 1)
    assert before["f_operating_margin"] == pytest.approx(22 / 105)
    assert before["f_operating_margin_yoy"] == pytest.approx(22 / 105 - 20 / 100)
    assert after["f_revenue_yoy"] == pytest.approx(121 / 110 - 1)
    assert after["f_operating_margin"] == pytest.approx(30.25 / 121)
    opened = next(s.open_at for s in sessions if s.session_id == "2026-01-21")
    assert after["f_days_since_report"] == pytest.approx((opened - RELEASE) / timedelta(days=1))
    # Cash-flow and inventory facts are absent from the fixture, so they stay missing.
    assert after["f_inventory_yoy"] is None and after["f_free_cashflow_yoy"] is None
    assert set(after) == set(FUNDAMENTAL_LEAVES)
    assert FUNDAMENTAL_LEAVES[:-1] == tuple("f_" + metric for metric in METRICS)


def test_leaves_are_session_constant_and_missing_without_fundamentals(snapshot):
    financials = financial_snapshot()
    leaves = ("f_revenue_yoy", "f_operating_margin", "f_days_since_report")
    source = with_fundamentals(snapshot, fundamental_plan(financials, leaves), [financials])
    assert isinstance(source, FundamentalSource) and source.ref == snapshot.ref
    table = build_features(source, leaves)
    sessions = np.array([row[1] for row in table.rows])
    for session_id in ("2026-01-20", "2026-01-21"):
        rows = table.values[sessions == session_id]
        assert np.isfinite(rows).all() and (rows == rows[0]).all()
    first = {s: table.values[sessions == s][0, 0] for s in ("2026-01-20", "2026-01-21")}
    assert first["2026-01-20"] != first["2026-01-21"]
    bare = build_features(snapshot, leaves)
    assert np.isnan(bare.values).all()


def test_binding_refuses_missing_substituted_or_unused_snapshots(snapshot, tmp_path):
    financials = financial_snapshot()
    bound = fundamental_plan(financials)
    with pytest.raises(ValueError, match="no financial snapshot"):
        with_fundamentals(snapshot, bound, [])
    other = financial_snapshot()
    other["records"] = other["records"][:-2]
    other["sha256"] = digest(canonical(other["records"]))
    with pytest.raises(ValueError, match="no financial snapshot"):
        with_fundamentals(snapshot, bound, [other])
    renamed = financial_snapshot("PEP")
    with pytest.raises(ValueError, match="is not KO"):
        with_fundamentals(snapshot, fundamental_plan(renamed), [renamed])
    with pytest.raises(ValueError, match="no f_ leaves"):
        with_fundamentals(snapshot, plan(), [financials])
    assert with_fundamentals(snapshot, plan(), []) is snapshot
    spy = plan(
        leaves=["ret_1", "f_revenue_yoy"],
        fundamentals=[{"instrument": "SPY", "snapshot_sha256": financials["sha256"]}],
    )
    with pytest.raises(ValueError, match="cover exactly the targets"):
        with_fundamentals(snapshot, spy, [financials])
    path = tmp_path / "snapshot.json"
    path.write_bytes(canonical(financials))
    assert load_financial_snapshot(path) == financials
    financials["records"][0]["value"] = 999.0
    path.write_bytes(canonical(financials))
    with pytest.raises(ValueError, match="does not match its digest"):
        load_financial_snapshot(path)


def test_plans_declare_fundamentals_exactly_when_f_leaves_are_used(snapshot):
    validate_leaves(snapshot.contract, FUNDAMENTAL_LEAVES)
    with pytest.raises(ValueError, match="unsupported"):
        validate_leaves(snapshot.contract, ("f_eps_yoy",))
    with pytest.raises(ValueError, match="exactly when an f_ leaf"):
        plan(leaves=["ret_1", "f_revenue_yoy"])
    with pytest.raises(ValueError, match="exactly when an f_ leaf"):
        plan(fundamentals=[{"instrument": "KO", "snapshot_sha256": "0" * 64}])


def test_evaluate_command_binds_the_financial_snapshot(tmp_path, capsys):
    store, ref = build_snapshot(tmp_path, count=32)
    financials = financial_snapshot()
    financial_path = tmp_path / "financials" / "snapshot.json"
    financial_path.parent.mkdir()
    financial_path.write_bytes(canonical(financials))
    plan_path = tmp_path / "plan.json"
    leaves = ["ret_1", "ret_3", "rv_12", "minutes_since_open", "f_revenue_yoy"]
    plan_path.write_text(fundamental_plan(financials, leaves).model_dump_json())
    command = ["evaluate", "--store", str(tmp_path / "store"), "--snapshot", ref]
    assert main([*command, "--plan", str(plan_path), "--financials", str(financial_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["leaves"] == leaves and len(report["folds"]) == 2
    with pytest.raises(ValueError, match="no financial snapshot"):
        main([*command, "--plan", str(plan_path)])
