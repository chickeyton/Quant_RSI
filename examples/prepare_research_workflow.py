"""Prepare an offline RFC #39 profile; --fixture uses invented prices, never market evidence."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, decode, digest
from quant_rsi.core.schemas import LocalDataSpec
from quant_rsi.evaluation.local_research import export_local_data
from quant_rsi.factors.discovery import DiscoveryPlan, bind_catalog
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.market import market_data
from quant_rsi.market.market_contracts import CSV_FIELDS, AccountRules, DataContract


def fixture(output, campaign_id):
    """Register a finite invented calendar and 105 quotes; train on its first 100 rows."""
    output.mkdir(parents=True, exist_ok=True)
    start = datetime(2020, 1, 1, 14, tzinfo=timezone.utc)
    sessions = [
        dict(
            session_id=f"s{i}",
            open_at=(start + timedelta(days=i)).isoformat(),
            close_at=(start + timedelta(days=i, hours=7)).isoformat(),
        )
        for i in range(105)
    ]
    contract = DataContract.model_validate_json(
        canonical(
            dict(
                schema_version=1,
                source_id="workflow-fixture-" + campaign_id,
                provider_statement="Invented KO prices and calendar; no real market evidence.",
                currency="USD",
                price_basis="raw",
                frequency="daily_equities",
                instruments=["KO"],
                calendar=dict(calendar_id="invented-daily", timezone="UTC", sessions=sessions),
                knowledge_basis="assumed_historical_availability",
                availability_evidence_ref=None,
                availability_claim="explicit_assumption",
                historical_assumption="available_at_event_time",
                universe_provenance="unknown_exploration",
                corporate_action_coverage="source_declares_complete_notices",
                columns={key: key for key in sorted(CSV_FIELDS)},
            )
        )
    )
    events = [
        dict(
            event_id="KO.membership",
            revision=0,
            kind="UNIVERSE_STATUS",
            instrument="KO",
            session_id=None,
            event_at=sessions[0]["open_at"],
            source_available_at=sessions[0]["open_at"],
            price=None,
            status="active",
            action=None,
            quality="valid",
        )
    ]
    for i, session in enumerate(sessions):
        for kind, phase, price in (
            ("OPEN_QUOTE", "open_at", 100 + i // 3 + i % 7),
            ("CLOSE_QUOTE", "close_at", 101 + i // 3 + i % 5),
        ):
            events.append(
                dict(
                    event_id=f"KO.{kind}.{i}",
                    revision=0,
                    kind=kind,
                    instrument="KO",
                    session_id=session["session_id"],
                    event_at=session[phase],
                    source_available_at=session[phase],
                    price=str(price),
                    status=None,
                    action=None,
                    quality="valid",
                )
            )
    csv_path = output / "invented-market.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(events)
    imported = market_data.ingest(contract, csv_path, request_id="workflow-fixture-" + campaign_id)
    data = export_local_data(
        LocalDataSpec(
            source="local_event_snapshot",
            sessions=100,
            instruments=1,
            train_cutoff=80,
            seed=7,
            snapshot_ref=imported["snapshot_ref"],
            decision_delay_seconds=60,
        )
    )
    units = b"Invented raw USD per share fixture; no splits or dividends; not predictive evidence."
    calendar = canonical(contract.calendar.model_dump(mode="json"))
    objective = dict(
        schema_version=1,
        kind="five_session_raw_close",
        horizon_sessions=5,
        target="log_close_ratio",
        currency="USD",
        price_unit="raw_currency_per_share",
        price_unit_claim="explicit_assumption",
        price_unit_evidence_ref=digest(units),
        calendar_evidence_ref=digest(calendar),
        notices=[],
        unknown_action_policy="affected_or_unknown_interval_abstain",
        prior_exposure="previously_exposed_historical_exploration",
    )
    return dict(
        kind="ForecastInputs",
        schema_version=1,
        target=dict(data=data, objective=objective),
        contexts={},
        signals=None,
        evidence_hex={digest(raw): raw.hex() for raw in (units, calendar)},
    )


def account(currency):
    costs = dict(commission_bps="5", tax_bps="0", slippage_bps="5")
    return AccountRules.model_validate(
        dict(
            currency=currency,
            currency_precision=2,
            rounding_policy="half_even_per_component",
            initial_cash="10000",
            cash_settlement="immediate_simulation",
            inventory_unlock="immediate_simulation",
            sizing="long_only_topk",
            allocation="1",
            max_positions=1,
            lot_size=1,
            cost_policy="linear_side_bps",
            buy_costs=costs,
            sell_costs=costs,
            cost_claim="explicit_observation_assumption",
            fill_policy="next_registered_open_market",
            fill_cap_units={"KO": 10000},
            funding_policy="pre_funded_no_unfilled_sale_credit",
            valuation="close_fresh_quote_only",
            benchmark="cash_zero_interest",
            action_policy="notice_suspend_unresolved",
            termination="preserve_inventory",
            stop_policy="cancel_unelapsed_targets",
            max_drawdown="0.2",
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", type=Path, nargs="?", help="Existing ForecastInputs JSON envelope"
    )
    parser.add_argument(
        "--fixture", action="store_true", help="Invent data locally; no network or real quotes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-id", default="rfc39-demo")
    parser.add_argument("--folds", default="[[80],[90]]")
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()
    if bool(args.inputs) == args.fixture:
        parser.error("provide either an inputs envelope or --fixture")
    if args.max_rounds < 2:
        parser.error("the complete workflow requires at least two research rounds")
    envelope = (
        fixture(args.output, args.campaign_id) if args.fixture else decode(args.inputs.read_bytes())
    )
    envelope, leaves = bind_catalog(
        envelope,
        Path(__file__).resolve().parents[1] / "instruments",
        "US",
        "KO",
        {"qlib_roc": "target", "qlib_ma": "target"},
    )
    inputs = ForecastInputs(envelope)
    plan = DiscoveryPlan.model_validate_json(
        canonical(
            dict(
                campaign_id=args.campaign_id,
                instrument="KO",
                exposure="engineering_fixture" if args.fixture else "historical_market_exploration",
                leaves=[leaf.model_dump(mode="json") for leaf in leaves],
                fold_origins=decode(args.folds),
                max_rounds=args.max_rounds,
            )
        )
    )
    profile = dict(
        schema_version=1,
        inputs=envelope,
        discovery=plan.model_dump(mode="json"),
        account=account(inputs.target[0].contract.currency).model_dump(mode="json"),
        strategy_threshold=0.0,
    )
    path, raw = args.output / "profile.json", canonical(profile)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"existing preparation differs: {path}")
    atomic_write(path, raw)
    print(path)
    print(
        "Engineering fixture: invented prices; publication is historical, not prospective."
        if args.fixture
        else "Historical exploration: validate and freeze account assumptions before running."
    )


if __name__ == "__main__":
    main()
