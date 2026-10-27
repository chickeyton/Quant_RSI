from copy import deepcopy
from datetime import datetime, timedelta

import pytest
from market_helpers import contract_and_events, write_events

from quant_rsi import market_data
from quant_rsi.artifacts import canonical, digest
from quant_rsi.forecast_features import (
    build_common_mask,
    build_feature_view,
    ordered_feature_values,
)
from quant_rsi.forecast_inputs import ForecastInputs, ForecastRecipe
from quant_rsi.forecast_kernel import forecast_feature_view
from quant_rsi.local_research import export_local_data
from quant_rsi.market_contracts import DataContract
from quant_rsi.schemas import FactorSpec, LocalDataSpec


def recipe(group):
    return ForecastRecipe(schema_version=1, kind="forecast_feature_recipe", group=group)


@pytest.fixture
def prepare_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    serial = 0

    def prepare(
        *,
        future_scale=1,
        missing_market=None,
        later_market_availability=False,
        shifted_market_close=False,
    ):
        nonlocal serial
        serial += 1
        evidence_hex = {}
        roles = {}
        for role, instrument in (("target", "A"), ("industry", "I"), ("market", "M")):
            contract, events, availability = contract_and_events(
                instruments=(instrument,), count=100
            )
            value = contract.model_dump(mode="json")
            value["source_id"] = f"{role}_{serial}"
            if role != "target":
                for session in value["calendar"]["sessions"]:
                    session["session_id"] = role + "_" + session["session_id"]
                for event in events:
                    if event["session_id"] is not None:
                        event["session_id"] = role + "_" + event["session_id"]
            if role == "market" and shifted_market_close:
                shifted = datetime.fromisoformat(value["calendar"]["sessions"][70]["close_at"])
                value["calendar"]["sessions"][70]["close_at"] = (
                    shifted + timedelta(hours=1)
                ).isoformat()
            for event in events:
                if event["kind"] not in {"OPEN_QUOTE", "CLOSE_QUOTE"}:
                    continue
                index = int(event["event_id"].rsplit(".", 1)[1])
                if role != "target":
                    event["price"] = str(100 + index * (1 if role == "industry" else 2))
                if index > 70:
                    event["price"] = str(float(event["price"]) * future_scale)
                if role == "market" and event["kind"] == "CLOSE_QUOTE":
                    if index == 70 and shifted_market_close:
                        event["event_at"] = value["calendar"]["sessions"][70]["close_at"]
                        event["source_available_at"] = event["event_at"]
                    if index == missing_market:
                        event.update(quality="missing", price=None)
                    if index == 70 and later_market_availability:
                        event["source_available_at"] = value["calendar"]["sessions"][71]["close_at"]
            contract = DataContract.model_validate_json(canonical(value))
            availability_path = tmp_path / f"availability-{role}-{serial}.txt"
            availability_path.write_bytes(availability)
            imported = market_data.ingest(
                contract,
                write_events(tmp_path / f"{role}-{serial}.csv", contract, events),
                request_id=f"{role}-{serial}",
                evidence_file=availability_path,
            )
            spec = LocalDataSpec(
                source="local_event_snapshot",
                sessions=100,
                instruments=1,
                train_cutoff=80,
                seed=7,
                snapshot_ref=imported["snapshot_ref"],
                decision_delay_seconds=60,
            )
            units = b"Engineering raw quote unit assumption"
            calendar = canonical(value["calendar"])
            for raw in (units, calendar):
                evidence_hex[digest(raw)] = raw.hex()
            objective = dict(
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
            )
            roles[role] = dict(data=export_local_data(spec), objective=objective)
        envelope = dict(
            kind="ForecastInputs",
            schema_version=1,
            target=roles.pop("target"),
            contexts=roles,
            signals=None,
            evidence_hex=evidence_hex,
        )
        return ForecastInputs(envelope), envelope

    return prepare


def test_nested_feature_groups_preserve_old_recipe_and_align_independent_session_ids(
    prepare_inputs,
):
    inputs, _ = prepare_inputs()
    data, objective = inputs.target
    views = {
        group: build_feature_view(inputs, recipe(group), "s70", data.decision_at(70))
        for group in ("B0", "B1", "B2")
    }
    legacy = forecast_feature_view(
        data.snapshot,
        data.contract,
        "s70",
        data.decision_at(70),
        FactorSpec(op="momentum", window=5),
        objective,
    )
    assert views["B0"]["values"] == legacy["rows"][0]["features"]
    assert views["B1"]["values"][:3] == views["B0"]["values"]
    assert views["B2"]["values"][:7] == views["B1"]["values"]
    assert [len(views[group]["values"]) for group in views] == [3, 7, 15]
    expected_industry = 170 / 165 - 1
    assert views["B2"]["values"][7] == expected_industry
    assert views["B2"]["values"][9] == views["B0"]["values"][0] - expected_industry
    assert views["B2"]["target_instrument"] == "A"


def test_future_price_changes_do_not_change_earlier_features(prepare_inputs):
    first, _ = prepare_inputs()
    changed, _ = prepare_inputs(future_scale=10)
    for group in ("B0", "B1", "B2"):
        a = build_feature_view(first, recipe(group), "s70", first.target[0].decision_at(70))
        b = build_feature_view(changed, recipe(group), "s70", changed.target[0].decision_at(70))
        assert a["values"] == b["values"]
        assert a["source_refs"] == b["source_refs"]


def test_context_availability_and_missing_quotes_do_not_fill_forward(prepare_inputs):
    inputs, _ = prepare_inputs(later_market_availability=True)
    at = inputs.target[0].decision_at(70)
    assert build_feature_view(inputs, recipe("B1"), "s70", at)["status"] == "eligible"
    view = build_feature_view(inputs, recipe("B2"), "s70", at)
    assert view["values"] is None
    assert "market:history_quote_missing_invalid_or_conflicting" in view["reasons"]


def test_common_training_mask_excludes_warmup_and_immature_labels_for_every_group(prepare_inputs):
    inputs, _ = prepare_inputs()
    selected = [recipe(group) for group in ("B0", "B1", "B2")]
    result = build_common_mask(
        inputs,
        selected,
        [55, 60, 65, 70],
        inputs.target[0].decision_at(70),
        purpose="training",
    )
    mask = result["mask"]
    assert mask["eligible_indices"] == [60, 65]
    assert len(mask["rows"]) == 4 and mask["selection_uses_prediction_errors"] is False
    assert len(mask["rows"][1]["feature_view_refs"]) == 3
    assert result["mask_ref"] == digest(canonical(mask))
    assert mask["rows"][-1]["label"]["status"] == "immature"
    assert mask["rows"][2]["label"]["target_session"] == "s70"


def test_missing_context_changes_common_input_mask_before_any_fit(prepare_inputs):
    inputs, _ = prepare_inputs(missing_market=65)
    selected = [recipe(group) for group in ("B0", "B1", "B2")]
    mask = build_common_mask(
        inputs,
        selected,
        [60, 65, 70],
        inputs.target[0].decision_at(80),
        purpose="evaluation",
    )["mask"]
    assert mask["eligible_indices"] == [60]
    assert len(mask["rows"]) == 3
    assert all(not row["eligible"] for row in mask["rows"][1:])


def test_ordered_model_input_rejects_reordered_or_wrong_dimension_features(prepare_inputs):
    inputs, _ = prepare_inputs()
    spec = recipe("B2")
    view = build_feature_view(inputs, spec, "s70", inputs.target[0].decision_at(70))
    assert ordered_feature_values(view, spec, input_ref=inputs.ref) == view["values"]
    bad = deepcopy(view)
    bad["feature_names"][0:2] = reversed(bad["feature_names"][0:2])
    with pytest.raises(ValueError, match="binding"):
        ordered_feature_values(bad, spec, input_ref=inputs.ref)
    bad = deepcopy(view)
    bad["values"].pop()
    with pytest.raises(ValueError, match="dimensions"):
        ordered_feature_values(bad, spec, input_ref=inputs.ref)
    with pytest.raises(ValueError, match="binding"):
        ordered_feature_values(view, recipe("B0"), input_ref=inputs.ref)


def test_input_envelope_rejects_tampered_evidence_and_unknown_fields(prepare_inputs):
    _, envelope = prepare_inputs()
    broken = deepcopy(envelope)
    ref = next(iter(broken["evidence_hex"]))
    broken["evidence_hex"][ref] = b"changed".hex()
    with pytest.raises(ValueError, match="hash"):
        ForecastInputs(broken)
    broken = deepcopy(envelope)
    broken["historical_news_is_PIT"] = True
    with pytest.raises(ValueError, match="envelope"):
        ForecastInputs(broken)


def test_context_calendar_cannot_borrow_a_later_closing_session(prepare_inputs):
    inputs, _ = prepare_inputs(shifted_market_close=True)
    view = build_feature_view(
        inputs,
        recipe("B2"),
        "s70",
        inputs.target[0].decision_at(70),
    )
    assert view["values"] is None
    assert "market:context_calendar_close_missing" in view["reasons"]


def test_admitted_input_is_detached_from_callers_mutable_envelope(prepare_inputs):
    inputs, envelope = prepare_inputs()
    at = inputs.target[0].decision_at(70)
    before = build_feature_view(inputs, recipe("B2"), "s70", at)
    envelope["target"]["data"]["snapshot"]["events"].clear()
    assert build_feature_view(inputs, recipe("B2"), "s70", at) == before
