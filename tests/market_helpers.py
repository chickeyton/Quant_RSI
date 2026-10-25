"""Explicit engineering-only market fixtures; no selected real market or provider."""

import csv
from datetime import datetime, timedelta, timezone

from quant_rsi.artifacts import canonical, digest
from quant_rsi.market_contracts import CSV_FIELDS, AccountRules, DataContract


def contract_and_events(*, basis="documented_historical_availability", instruments=("A",), count=7):
    start = datetime(2020, 10, 19, 9, tzinfo=timezone.utc)
    sessions = [
        dict(
            session_id=f"s{i}",
            open_at=(start + timedelta(days=i)).isoformat(),
            close_at=(start + timedelta(days=i, hours=7)).isoformat(),
        )
        for i in range(count)
    ]
    evidence = b"Engineering fixture explicitly publishes quotes at the supplied source timestamp."
    value = dict(
        schema_version=1,
        source_id="fixture",
        provider_statement="Invented engineering input",
        currency="TEST",
        price_basis="raw",
        frequency="daily_equities",
        calendar=dict(calendar_id="fixture", timezone="UTC", sessions=sessions),
        instruments=list(instruments),
        knowledge_basis=basis,
        availability_evidence_ref=digest(evidence) if basis.startswith("documented") else None,
        availability_claim={
            "documented_historical_availability": "source_documented_not_independently_verified",
            "assumed_historical_availability": "explicit_assumption",
            "observed_receipts": "local_receipt",
        }[basis],
        historical_assumption="available_at_event_time" if basis.startswith("assumed") else None,
        universe_provenance="unknown_exploration",
        corporate_action_coverage="unknown",
        columns={key: key for key in sorted(CSV_FIELDS)},
    )
    events = []
    for instrument in instruments:
        events.append(
            dict(
                event_id=f"{instrument}.membership",
                revision=0,
                kind="UNIVERSE_STATUS",
                instrument=instrument,
                session_id=None,
                event_at=sessions[0]["open_at"],
                source_available_at=sessions[0]["open_at"],
                price=None,
                status="active",
                action=None,
                quality="valid",
            )
        )
        for i, session in enumerate(sessions):
            for kind, field, price in (
                (
                    "OPEN_QUOTE",
                    "open_at",
                    ([100, 100, 110, 121, 120, 119, 118][i] if i < 7 else 100 + i + i % 3),
                ),
                (
                    "CLOSE_QUOTE",
                    "close_at",
                    ([100, 110, 121, 122, 120, 118, 117][i] if i < 7 else 101 + i + i % 4),
                ),
            ):
                events.append(
                    dict(
                        event_id=f"{instrument}.{kind}.{i}",
                        revision=0,
                        kind=kind,
                        instrument=instrument,
                        session_id=session["session_id"],
                        event_at=session[field],
                        source_available_at=session[field],
                        price=str(price),
                        status=None,
                        action=None,
                        quality="valid",
                    )
                )
    return DataContract.model_validate_json(canonical(value)), events, evidence


def write_events(path, contract, events):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(contract.columns.values()))
        writer.writeheader()
        writer.writerows(
            {column: event.get(key) for key, column in contract.columns.items()} for event in events
        )
    return path


def account_rules(**changes):
    value = dict(
        currency="TEST",
        currency_precision=2,
        rounding_policy="half_even_per_component",
        initial_cash="1000",
        cash_settlement="immediate_simulation",
        inventory_unlock="immediate_simulation",
        sizing="long_only_topk",
        allocation="1",
        max_positions=2,
        lot_size=1,
        cost_policy="linear_side_bps",
        buy_costs=dict(commission_bps="100", tax_bps="0", slippage_bps="0"),
        sell_costs=dict(commission_bps="200", tax_bps="100", slippage_bps="0"),
        cost_claim="explicit_observation_assumption",
        fill_policy="next_registered_open_market",
        fill_cap_units={"A": 3, "B": 4},
        funding_policy="pre_funded_no_unfilled_sale_credit",
        valuation="close_fresh_quote_only",
        benchmark="cash_zero_interest",
        action_policy="notice_suspend_unresolved",
        termination="preserve_inventory",
        stop_policy="cancel_unelapsed_targets",
        max_drawdown="1",
    )
    return AccountRules.model_validate_json(canonical(value | changes))
