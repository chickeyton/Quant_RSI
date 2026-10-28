"""In-memory engineering envelopes only: no registry ingestion or model calls."""

import csv
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from market_helpers import contract_and_events
from test_forecast_inputs import snapshot_value

from quant_rsi.core.artifacts import canonical, decode, digest
from quant_rsi.evaluation.local_research import LocalResearchData
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.forecast.recursive.forecast_recursive_market import (
    ForecastRecursiveMarketEvidence,
    project_market_target,
)
from quant_rsi.market.market_contracts import DataContract, utc_text
from quant_rsi.market.market_data import parse_csv

FIRST_SEEN = "2026-10-27T00:00:00Z"


def prepared_inputs(
    count=64,
    *,
    future_count=5,
    target_gap=None,
    future_quote=False,
    context_missing=False,
    unknown_action=True,
    with_signals=True,
):
    """Match exported object/receipt shapes using pure CSV parsing, never a registry."""
    evidence_hex, markets, imports = {}, {}, {}
    for sequence, (role, instrument) in enumerate(
        (("target", "A"), ("industry", "I"), ("market", "M")), 1
    ):
        contract, events, availability = contract_and_events(
            instruments=(instrument,), count=count + future_count
        )
        value = contract.model_dump(mode="json")
        value["source_id"] = f"{role}_fixture"
        contract = DataContract.model_validate_json(canonical(value))
        historical = {session.session_id for session in contract.calendar.sessions[:count]}
        events = [
            event
            for event in events
            if event["session_id"] is None
            or event["session_id"] in historical
            or (role == "target" and future_quote and event["session_id"] == f"s{count}")
        ]
        if role == "target" and target_gap is not None:
            events = [
                event
                for event in events
                if not (event["session_id"] == target_gap and event["kind"] == "CLOSE_QUOTE")
            ]
        if role == "market" and context_missing:
            for event in events:
                if event["session_id"] == "s10" and event["kind"] == "CLOSE_QUOTE":
                    event.update(quality="missing", price=None)
        if unknown_action and role == "target":
            events.append(
                dict(
                    event_id="A.unknown_split",
                    revision=0,
                    kind="CORPORATE_ACTION_NOTICE",
                    instrument="A",
                    session_id="s20",
                    event_at=utc_text(contract.calendar.sessions[20].open_at),
                    source_available_at=utc_text(contract.calendar.sessions[20].open_at),
                    price=None,
                    status=None,
                    action='{"action_kind":"split","ratio":"unknown"}',
                    quality="valid",
                )
            )
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(contract.columns.values()))
        writer.writeheader()
        writer.writerows(
            {column: event.get(key) for key, column in contract.columns.items()} for event in events
        )
        raw = stream.getvalue().encode()
        raw_ref = digest(raw)
        parsed_events = []
        for parsed in parse_csv(contract, raw, FIRST_SEEN):
            event = parsed.model_dump(mode="json")
            event.update(
                event_at=utc_text(parsed.event_at),
                source_available_at=(
                    None
                    if parsed.source_available_at is None
                    else utc_text(parsed.source_available_at)
                ),
            )
            parsed_events.append(
                dict(
                    event=event,
                    payload_ref=digest(canonical(event)),
                    first_seen_at=FIRST_SEEN,
                    receipt_seq=sequence,
                )
            )
        snapshot = dict(
            kind="LocalMarketSnapshot",
            schema_version=1,
            registry_id="unit-only-registry",
            source_id=contract.source_id,
            contract_ref=digest(canonical(value)),
            receipt_watermark=sequence,
            events=sorted(
                parsed_events,
                key=lambda item: (item["event"]["event_id"], item["event"]["revision"]),
            ),
            receipts=[dict(seq=sequence, raw_ref=raw_ref, first_seen_at=FIRST_SEEN)],
        )
        snapshot_ref = digest(canonical(snapshot))
        data = dict(
            kind="LocalResearchData",
            specification=dict(
                source="local_event_snapshot",
                sessions=count,
                instruments=1,
                train_cutoff=count - 5,
                seed=7,
                snapshot_ref=snapshot_ref,
                decision_delay_seconds=60,
            ),
            contract=value,
            snapshot=snapshot,
            raw_inputs={raw_ref: raw.decode()},
            availability_evidence_hex=availability.hex(),
        )
        units, calendar = (
            f"Explicit engineering units for {role}".encode(),
            canonical(value["calendar"]),
        )
        for body in (raw, availability, units, calendar):
            evidence_hex[digest(body)] = body.hex()
        markets[role] = dict(
            data=data,
            objective=dict(
                schema_version=1,
                kind="five_session_raw_close",
                horizon_sessions=5,
                target="log_close_ratio",
                currency="TEST",
                price_unit="raw_currency_per_share",
                price_unit_claim="explicit_assumption",
                price_unit_evidence_ref=digest(units),
                calendar_evidence_ref=digest(calendar),
                notices=[],
                unknown_action_policy="affected_or_unknown_interval_abstain",
                prior_exposure="previously_exposed_historical_exploration",
                source_qualifications=["Invented offline input; unknown notices remain unresolved"],
            ),
        )
        imports[role] = dict(
            receipt=dict(
                registry_id="unit-only-registry",
                receipt_seq=sequence,
                request_id=f"context.{role}.v1",
                source_id=contract.source_id,
                snapshot_ref=snapshot_ref,
                raw_ref=raw_ref,
                first_seen_at=FIRST_SEEN,
                fresh_events=len(parsed_events),
                purpose="developer_visible_observation",
                forward_qualification="requires_preexisting_registered_episode_and_actual_receipts",
            ),
            http_received_at="2026-10-26T23:00:00Z",
            unsupported_actions=["A.unknown_split"] if unknown_action and role == "target" else [],
        )
    audit = canonical(dict(kind="offline_engineering_manifest", files=[]))
    evidence_hex[digest(audit)] = audit.hex()
    if with_signals:
        signal_evidence = b"engineering source"
        evidence_hex[digest(signal_evidence)] = signal_evidence.hex()
    inputs = dict(
        kind="ForecastInputs",
        schema_version=1,
        target=markets.pop("target"),
        contexts=markets,
        signals=snapshot_value() if with_signals else None,
        evidence_hex=evidence_hex,
    )
    receipt = dict(
        schema_version=1,
        kind="forecast_context_preparation_receipt",
        status="complete",
        audit_manifest_ref=digest(audit),
        inputs_ref=digest(canonical(inputs)),
        origin_session_id=f"s{count - 1}",
        historical_sessions=count,
        future_sessions=[f"s{index}" for index in range(count, count + future_count)],
        workspace="/unit-only/not-opened/workspace",
        registry_path="/unit-only/not-opened/workspace/.quant_rsi/market",
        imports=imports,
        qualifications=["Pure invented fixture; no HTTP or registry invocation"],
        completed_at="2026-10-27T00:01:00Z",
    )
    return inputs, canonical(receipt)


def rebind(inputs, receipt):
    receipt = decode(receipt)
    receipt["inputs_ref"] = digest(canonical(inputs))
    return canonical(receipt)


def test_projection_changes_only_declared_target_specification_fields():
    inputs, receipt = prepared_inputs()
    before = canonical(inputs)
    result = project_market_target(inputs, receipt)
    expected = deepcopy(inputs["target"]["data"])
    expected["specification"].update(train_cutoff=29, seed=0)
    assert result["target_data"] == expected
    assert canonical(inputs) == before
    assert canonical(result["inputs"]) == before
    assert result["inputs_ref"] == digest(before)
    assert result["preparation_receipt"] == receipt
    assert result["preparation_receipt_ref"] == digest(receipt)
    assert result["target_data_ref"] == digest(canonical(expected))
    assert LocalResearchData.from_json(expected).spec.train_cutoff == 29
    assert result["quoted_sessions"] == 64
    assert len(expected["contract"]["calendar"]["sessions"]) == 69
    assert result["development_origins"] == [29, 34, 39, 44, 49, 54]
    for key in ("contract", "snapshot", "raw_inputs", "availability_evidence_hex"):
        assert canonical(expected[key]) == canonical(inputs["target"]["data"][key])
    assert result["inputs"]["contexts"] == inputs["contexts"]
    assert result["inputs"]["signals"] == inputs["signals"]


def test_evidence_schema_binds_full_input_original_receipt_and_exact_target():
    inputs, receipt = prepared_inputs()
    result = project_market_target(ForecastInputs(inputs), receipt)
    evidence = ForecastRecursiveMarketEvidence.model_validate_json(
        canonical(result["market_evidence"])
    )
    assert evidence.model_dump(mode="json") == dict(
        kind="ForecastRecursiveMarketEvidence",
        schema_version=1,
        inputs_ref=result["inputs_ref"],
        preparation_receipt_ref=digest(receipt),
        target_data_ref=result["target_data_ref"],
        projection="target_only_v1",
    )
    assert result["market_evidence_ref"] == digest(canonical(result["market_evidence"]))
    assert evidence.target_data_ref != digest(canonical(inputs["target"]["data"]))


def test_original_receipt_byte_representation_is_preserved():
    inputs, original = prepared_inputs()
    receipt = json.dumps(decode(original), indent=2).encode() + b"\n"
    canonical_result = project_market_target(inputs, original)
    result = project_market_target(inputs, receipt)
    assert result["preparation_receipt"] == receipt
    assert result["preparation_receipt_ref"] == digest(receipt) != digest(original)
    assert result["target_data_ref"] == canonical_result["target_data_ref"]
    assert result["market_evidence_ref"] != canonical_result["market_evidence_ref"]


def test_projection_and_returned_original_do_not_share_mutable_objects():
    inputs, receipt = prepared_inputs()
    result = project_market_target(inputs, receipt)
    result["target_data"]["contract"]["provider_statement"] = "changed returned copy"
    result["inputs"]["contexts"]["industry"]["data"]["specification"]["seed"] = 999
    assert (
        inputs["target"]["data"]["contract"]["provider_statement"] == "Invented engineering input"
    )
    assert inputs["contexts"]["industry"]["data"]["specification"]["seed"] == 7
    assert (
        result["inputs"]["target"]["data"]["contract"]["provider_statement"]
        == "Invented engineering input"
    )


def test_context_missing_value_is_retained_without_new_model_eligibility_gate():
    inputs, receipt = prepared_inputs(context_missing=True)
    result = project_market_target(inputs, receipt)
    assert result["inputs"]["contexts"] == inputs["contexts"]
    assert any(
        item["event"]["quality"] == "missing"
        for item in result["inputs"]["contexts"]["market"]["data"]["snapshot"]["events"]
    )
    assert set(result["target_data"]["contract"]["instruments"]) == {"A"}
    assert "contexts" not in result["target_data"]


@pytest.mark.parametrize("field", ["train_cutoff", "seed"])
def test_changed_original_specification_rejects_original_receipt(field):
    inputs, receipt = prepared_inputs()
    inputs["target"]["data"]["specification"][field] += -1 if field == "train_cutoff" else 1
    with pytest.raises(ValueError, match="complete original"):
        project_market_target(inputs, receipt)


def test_mutated_prevalidated_object_is_revalidated_against_receipt():
    inputs, receipt = prepared_inputs()
    parsed = ForecastInputs(inputs)
    parsed.value["target"]["data"]["specification"]["seed"] = 55
    with pytest.raises(ValueError, match="complete original"):
        project_market_target(parsed, receipt)


@pytest.mark.parametrize(
    "options",
    [dict(future_count=4), dict(future_count=6), dict(target_gap="s10"), dict(future_quote=True)],
)
def test_recorded_target_grid_not_calendar_length_or_declared_count(options):
    inputs, receipt = prepared_inputs(**options)
    with pytest.raises(ValueError, match="target history"):
        project_market_target(inputs, receipt)


def test_spec_sessions_cannot_relabel_future_calendar_as_quoted_history():
    inputs, receipt = prepared_inputs()
    inputs["target"]["data"]["specification"]["sessions"] += 1
    with pytest.raises(ValueError, match="target history"):
        project_market_target(inputs, rebind(inputs, receipt))


@pytest.mark.parametrize(
    "field,value",
    [
        ("historical_sessions", 65),
        ("historical_sessions", True),
        ("origin_session_id", "s62"),
        ("future_sessions", ["s65", "s66", "s67", "s68", "s69"]),
        ("status", "pending"),
        ("schema_version", 2),
        ("kind", "other"),
    ],
)
def test_changed_preparation_dimension_or_completion_rejects(field, value):
    inputs, receipt = prepared_inputs()
    changed = decode(receipt)
    changed[field] = value
    with pytest.raises(ValueError):
        project_market_target(inputs, canonical(changed))


@pytest.mark.parametrize("role", ["target", "industry", "market"])
def test_original_import_receipt_is_checked_for_every_retained_role(role):
    inputs, receipt = prepared_inputs()
    changed = decode(receipt)
    changed["imports"][role]["receipt"]["first_seen_at"] = "2026-10-27T00:00:01Z"
    with pytest.raises(ValueError, match="preparation import"):
        project_market_target(inputs, canonical(changed))


@pytest.mark.parametrize("role", ["industry", "market"])
def test_missing_unused_context_evidence_still_rejects(role):
    inputs, receipt = prepared_inputs()
    ref = inputs["contexts"][role]["objective"]["price_unit_evidence_ref"]
    inputs["evidence_hex"].pop(ref)
    with pytest.raises(ValueError, match="missing referenced evidence"):
        project_market_target(inputs, rebind(inputs, receipt))


def test_missing_audit_manifest_does_not_become_an_unchecked_path():
    inputs, receipt = prepared_inputs()
    inputs["evidence_hex"].pop(decode(receipt)["audit_manifest_ref"])
    with pytest.raises(ValueError, match="complete original"):
        project_market_target(inputs, rebind(inputs, receipt))


def test_missing_context_role_cannot_claim_complete_original_bundle():
    inputs, receipt = prepared_inputs()
    inputs["contexts"].pop("industry")
    with pytest.raises(ValueError, match="complete original"):
        project_market_target(inputs, rebind(inputs, receipt))


def test_corrupt_raw_or_signal_evidence_is_not_repaired():
    inputs, receipt = prepared_inputs()
    raw_ref = next(iter(inputs["target"]["data"]["raw_inputs"]))
    inputs["target"]["data"]["raw_inputs"][raw_ref] += "\n"
    with pytest.raises(ValueError, match="raw input hash"):
        project_market_target(inputs, rebind(inputs, receipt))
    inputs, receipt = prepared_inputs()
    ref = inputs["signals"]["events"][0]["payload_ref"]
    inputs["evidence_hex"][ref] = b"changed".hex()
    with pytest.raises(ValueError, match="evidence hash"):
        project_market_target(inputs, rebind(inputs, receipt))


def test_short_history_does_not_truncate_or_fallback_to_another_cutoff():
    inputs, receipt = prepared_inputs(count=46)
    with pytest.raises(ValueError, match="train_cutoff"):
        project_market_target(inputs, receipt)


def test_projection_has_no_file_registry_network_or_training_calls(monkeypatch):
    import urllib.request

    from quant_rsi.evaluation import local_research
    from quant_rsi.market import market_data

    inputs, receipt = prepared_inputs()

    def forbidden(*args, **kwargs):
        raise AssertionError("pure projection attempted an external operation")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", forbidden)
        scoped.setattr(market_data.DataRegistry, "__init__", forbidden)
        scoped.setattr(local_research, "market_context", forbidden)
        scoped.setattr(local_research, "training_view", forbidden)
        scoped.setattr(urllib.request, "urlopen", forbidden)
        result = project_market_target(inputs, receipt)
    assert result["quoted_sessions"] == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt.update(imports=None),
        lambda receipt: receipt["imports"].update(target=None),
        lambda receipt: receipt["imports"]["target"]["receipt"].pop("raw_ref"),
        lambda receipt: receipt["imports"]["target"]["receipt"].update(receipt_seq=True),
        lambda receipt: receipt.update(schema_version=True),
    ],
)
def test_incomplete_or_untyped_receipt_rejects_without_external_recovery(mutation):
    inputs, raw = prepared_inputs()
    receipt = decode(raw)
    mutation(receipt)
    with pytest.raises(ValueError):
        project_market_target(inputs, canonical(receipt))
