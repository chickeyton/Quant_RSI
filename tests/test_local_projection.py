"""Real-import projection probes; task assembly tests deliberately do not issue C5 grants."""

import json
from copy import deepcopy
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from market_helpers import account_rules, contract_and_events, write_events

from quant_rsi.artifacts import ArtifactStore, canonical, digest
from quant_rsi.asof import make_feature_view
from quant_rsi.local_evidence import LocalProgramInput, economic_session_keys, register_tasks
from quant_rsi.local_projection import development_projection
from quant_rsi.local_research import (
    LocalResearchData,
    export_local_data,
    fit_predict_local,
    training_view,
)
from quant_rsi.market_contracts import DataContract, trusted_now
from quant_rsi.market_data import ingest
from quant_rsi.researcher import ResearchPlan, package_definition
from quant_rsi.schemas import FactorSpec, LocalDataSpec


def local_input(
    root, monkeypatch, *, name="fixture", count=36, mutate=None, reverse=False, offset=False
):
    root.mkdir()
    contract, events, availability = contract_and_events(instruments=tuple("ABCD"), count=count)
    document = contract.model_dump(mode="json")
    document["source_id"] = name
    if offset:
        document["calendar"]["timezone"] = "America/St_Thomas"
        for row, session in zip(
            document["calendar"]["sessions"], contract.calendar.sessions, strict=True
        ):
            row["open_at"] = session.open_at.astimezone(timezone(timedelta(hours=-4))).isoformat()
            row["close_at"] = session.close_at.astimezone(timezone(timedelta(hours=-4))).isoformat()
    contract = DataContract.model_validate_json(canonical(document))
    if mutate is not None:
        mutate(events, contract)
    path = write_events(root / "input.csv", contract, list(reversed(events)) if reverse else events)
    availability_path = root / "availability.txt"
    availability_path.write_bytes(availability)
    with monkeypatch.context() as patch:
        patch.setenv("QUANT_RSI_WORKSPACE", str(root))
        report = ingest(contract, path, request_id="actual-import", evidence_file=availability_path)
        spec = LocalDataSpec(
            source="local_event_snapshot",
            sessions=count,
            instruments=4,
            train_cutoff=12,
            seed=0,
            snapshot_ref=report["snapshot_ref"],
            decision_delay_seconds=60,
        )
        return LocalResearchData(export_local_data(spec))


def template(data, sessions=24):
    value = json.loads(Path("examples/research_development.json").read_text())
    value["development"]["data"] = data.spec.model_dump(mode="json") | {"sessions": sessions}
    value["development"]["portfolio"] = account_rules(
        max_positions=2, fill_cap_units={s: 100 for s in data.instruments}
    ).model_dump(mode="json")
    return ResearchPlan.model_validate_json(canonical(value))


def test_future_raw_canary_preserves_actual_model_masks_and_development(tmp_path, monkeypatch):
    original = local_input(tmp_path / "original", monkeypatch)

    def canary(events, contract):
        for row in events:
            if row["session_id"] in {s.session_id for s in contract.calendar.sessions[24:]}:
                row["price"] = "987654.321"

    changed = local_input(tmp_path / "canary", monkeypatch, mutate=canary)
    projections = [development_projection(d, 24) for d in (original, changed)]
    assert projections[0]["raw_inputs"] == projections[1]["raw_inputs"]
    assert "987654.321" not in canonical(projections[1]).decode()
    assert (
        projections[0]["projection_lineage"]["original_snapshot_ref"]
        != projections[1]["projection_lineage"]["original_snapshot_ref"]
    )
    actual = []
    for projection in projections:
        data = LocalResearchData(projection)
        plan = template(data).development
        factor = FactorSpec(op="momentum", window=5)
        x, y, metadata, mask = training_view(data, plan, factor)
        fitted = fit_predict_local(data, plan, factor)
        actual.append(
            (
                x,
                y,
                metadata["training_rows"],
                [(r["session_id"], r["instrument"], r["training_eligible"]) for r in mask["rows"]],
                fitted,
            )
        )
    np.testing.assert_array_equal(actual[0][0], actual[1][0])
    np.testing.assert_array_equal(actual[0][1], actual[1][1])
    assert actual[0][2:4] == actual[1][2:4]
    assert actual[0][4]["model_text"] == actual[1][4]["model_text"]
    assert [(p["session_id"], p["rows"]) for p in actual[0][4]["predictions"]] == [
        (p["session_id"], p["rows"]) for p in actual[1][4]["predictions"]
    ]


def test_projection_keeps_original_first_seen_and_separate_creation_time(tmp_path, monkeypatch):
    data = local_input(tmp_path / "input", monkeypatch)
    projected = development_projection(data, 24)
    lineage = projected["projection_lineage"]
    assert lineage["receipts"][0]["first_seen_at"] == data.snapshot["receipts"][0]["first_seen_at"]
    assert lineage["derived_created_at"] >= lineage["receipts"][0]["first_seen_at"]
    assert projected["snapshot"]["kind"] == "DerivedLocalSnapshot"
    assert (
        projected["snapshot"]["receipts"][0]["raw_ref"] != data.snapshot["receipts"][0]["raw_ref"]
    )
    assert projected["availability_evidence_hex"] is None
    bad = deepcopy(projected)
    bad["snapshot"]["receipts"][0]["first_seen_at"] = "2000-01-01T00:00:00+00:00"
    bad["specification"]["snapshot_ref"] = digest(canonical(bad["snapshot"]))
    with pytest.raises(ValueError, match="first-seen"):
        LocalResearchData(bad)


def test_already_known_future_notice_remains_visible(tmp_path, monkeypatch):
    def add_notice(events, contract):
        events.append(
            dict(
                events[0],
                event_id="notice.future",
                kind="CORPORATE_ACTION_NOTICE",
                session_id="s30",
                event_at=contract.calendar.sessions[30].open_at.isoformat(),
                source_available_at=contract.calendar.sessions[20].close_at.isoformat(),
                status=None,
                action="announced split",
            )
        )

    data = local_input(tmp_path / "notice", monkeypatch, mutate=add_notice)
    projected = LocalResearchData(development_projection(data, 24))
    assert any(r["event"]["event_id"] == "notice.future" for r in projected.snapshot["events"])
    view = make_feature_view(
        projected.snapshot,
        projected.contract,
        "s23",
        projected.decision_at(23),
        FactorSpec(op="momentum", window=5),
    )
    assert view["rows"][0]["status"] != "eligible"


@pytest.mark.parametrize("cross_role", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_task_assembly_rejects_economic_alias_and_cross_role_leakage(
    tmp_path, monkeypatch, cross_role, offset
):
    one = local_input(tmp_path / "one", monkeypatch)
    two = local_input(
        tmp_path / "two", monkeypatch, name="renamed_source", reverse=True, offset=offset
    )
    assert one.snapshot["source_id"] != two.snapshot["source_id"]
    assert set(one.value["raw_inputs"]) != set(two.value["raw_inputs"])
    assert economic_session_keys(one, 0, 36) == economic_session_keys(two, 0, 36)
    parent = template(one)
    # Package metadata is a unit fixture. Actual C4 import and issuance are covered separately.
    imported = dict(
        plans={r: parent.model_dump(mode="json") for r in ("parent", "candidate")},
        packages={
            r: digest(canonical(package_definition(parent))) for r in ("parent", "candidate")
        },
    )
    tasks = []
    sources = []
    for i, data in enumerate((one, two)):
        start, end = (24, 30) if cross_role and i == 0 else ((30, 36) if cross_role else (24, 36))
        value = deepcopy(data.value)
        value["specification"]["sessions"] = end
        sources.append(value)
        tasks.append(
            dict(
                task_id=f"task_{i}",
                data=value["specification"],
                development_sessions=start,
                score_session_ids=list(data.dates[start:end]),
                terminal_session_id=data.dates[end - 1],
                weight="0.5",
                economic_group="same_economic_market",
                dependence_cluster="unverified_common_cluster",
                prior_exposure="public_or_previously_exposed_engineering",
            )
        )
    spec = LocalProgramInput.model_validate_json(
        canonical(
            dict(
                name="alias unit",
                source="local_event_snapshot",
                package_campaign="unused",
                tasks=tasks,
            )
        )
    )
    with pytest.raises(
        ValueError, match="another task score rows" if cross_role else "cannot gain task weight"
    ):
        register_tasks(
            spec,
            imported,
            dict(created_at=trusted_now(), sources=sources),
            ArtifactStore(tmp_path / "artifacts"),
            ArtifactStore(tmp_path / "protected"),
        )
