"""Cross-run study receipts using the existing journal, artifact and account helpers."""

import math
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from quant_rsi.core.artifacts import ArtifactStore, canonical, decode, digest
from quant_rsi.core.controller import controller_lock
from quant_rsi.core.data import MarketData
from quant_rsi.core.journal import Journal
from quant_rsi.forecast.forecast_kernel import forecast_errors
from quant_rsi.forecast.forecast_publication import confirmed_report
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research.demo_prices import load_daily_prices
from quant_rsi.research.demo_strategy import evaluate_strategy

IDENTITY = digest(canonical({"kind": "ConnectedStudyHistory", "schema_version": 1}))
NOTICE = (
    "Individual forecast receipts, not a preregistered cohort or independent confirmation. "
    "Later public collections are matched by provider, instrument and frozen calendar; "
    "they are not receipt-prefix extensions of a market registry. "
    "Strategy metrics replay the frozen signal on later received bars with explicit costs; "
    "they are hypothetical, not durable paper orders or broker fills."
)


@contextmanager
def _ledger(directory):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with controller_lock(root):
        journal = Journal(root, IDENTITY)
        try:
            metadata = dict(journal.connection.execute("SELECT key,value FROM metadata"))
            if metadata != {"manifest_digest": IDENTITY}:
                raise ValueError("Directory belongs to another journal")
            yield journal, ArtifactStore(root / "artifacts")
        finally:
            journal.close()


def _entries(journal, store):
    entries = {}
    for event in journal.connection.execute("SELECT kind,payload FROM events ORDER BY sequence"):
        payload = decode(event["payload"].encode())
        if event["kind"] == "study_registered":
            entry = store.json(payload["ref"])
            entries[entry["study_id"]] = entry | {"observations": []}
        elif event["kind"] == "study_confirmed":
            confirmation = store.json(payload["ref"])
            entry = entries[confirmation["study_id"]]
            publication = confirmed_report(entry["publication"], confirmation)
            entry["publication"] = publication
            entry["status"] = "pending" if publication["forward"] else "not_prospective"
            entry["reason"] = (
                "target_not_observed"
                if publication["forward"]
                else "issued_or_confirmed_outside_origin_close_next_open"
            )
        elif event["kind"] == "study_observed":
            outcome = store.json(payload["ref"])
            entries[outcome["study_id"]]["observations"].append(outcome)
    return entries


def _append(journal, store, kind, value):
    reference = store.put_json(value)
    with journal.connection:
        journal.event(None, kind, {"ref": reference})


def _confirm(journal, store, entry):
    publication = entry.get("publication")
    if publication and "post_commit_confirmed_at" not in publication:
        _append(
            journal,
            store,
            "study_confirmed",
            {
                "study_id": entry["study_id"],
                "forecast_ref": digest(canonical(publication)),
                "confirmed_at": market_contracts.trusted_now(),
            },
        )


def collection_snapshot(symbol, directory, *, horizon_sessions=5, historical_sessions=None):
    data, info = load_daily_prices(
        symbol,
        directory,
        horizon_sessions=horizon_sessions,
        historical_sessions=historical_sessions,
    )
    root = Path(directory)
    return {
        "info": {key: value for key, value in info.items() if key != "collection_directory"},
        "dates": list(data.dates),
        "opens": data.opens[:, 0].tolist(),
        "closes": data.closes[:, 0].tolist(),
        "contract": decode((root / "data_contract.json").read_bytes()),
        "source_files": {
            name: (root / name).read_text()
            for name in ("data_contract.json", "events.csv", "collection_receipt.json")
        },
    }


def register_study(directory, run_root, result, snapshot=None):
    """Freeze the actually selected forecast and parameters after research finishes."""
    run_root = str(Path(run_root).resolve())
    study_id = digest(run_root.encode())
    with _ledger(directory) as (journal, store):
        existing = _entries(journal, store).get(study_id)
        result_ref = store.put_json(result)
        if existing:
            if existing["result_ref"] != result_ref:
                raise ValueError("Study directory already binds a different completed result")
            _confirm(journal, store, existing)
            return _entries(journal, store)[study_id]
        now = market_contracts.trusted_now()
        selected = result["iterations"]["selected_round"]
        attempt = result["iterations"]["attempts"][selected - 1]
        forecast = attempt["prediction"].get("latest_forecast")
        sessions, publication = [], None
        status, reason = "unavailable", "selected_latest_forecast_unavailable"
        if forecast and snapshot:
            if timestamp(snapshot["info"]["received_at"]) > timestamp(now):
                raise ValueError("Publication precedes its price receipt")
            calendar = snapshot["contract"]["calendar"]["sessions"]
            index = next(
                i for i, s in enumerate(calendar) if s["session_id"] == forecast["origin_session"]
            )
            horizon = forecast["horizon_sessions"]
            if type(horizon) is not int or horizon not in (5, 20, 60):
                raise ValueError("Unsupported forecast horizon")
            sessions = calendar[index : index + horizon + 1]
            if (
                len(sessions) != horizon + 1
                or sessions[-1]["session_id"] != forecast["target_session"]
            ):
                raise ValueError("Forecast differs from its retained horizon calendar")
            publication = {
                "status": "forecast",
                "issued_at": now,
                "origin_close_at": sessions[0]["close_at"],
                "forward_deadline": sessions[1]["open_at"],
                "forecast": forecast,
                "result_ref": result_ref,
            }
            status, reason = "not_prospective", "publication_confirmation_missing"
        from quant_rsi.research.study_factor_ledger import attempt_records, attempted_definitions
        from quant_rsi.research.study_factor_memory import selected_definitions

        entry = {
            "study_id": study_id,
            "run_directory": run_root,
            "symbol": result["configuration"]["symbol"],
            "question": result["configuration"].get("question", ""),
            "horizon_sessions": result["configuration"].get(
                "horizon_sessions", (forecast or {}).get("horizon_sessions", 5)
            ),
            "issued_at": now,
            "selected_round": selected,
            "factor_ids": attempt["prediction"]["factor_ids"],
            "factor_definitions": selected_definitions(result),
            "attempts": attempt_records(result),
            "attempted_definitions": attempted_definitions(result),
            "input_mode": result["configuration"].get("prices", "unknown"),
            "forecast": forecast,
            "publication": publication,
            "sessions": sessions,
            "status": status,
            "reason": reason,
            "result_ref": result_ref,
            "source_ref": store.put_json(snapshot) if snapshot else None,
            "strategy_parameters": {
                key: result["configuration"][key]
                for key in ("initial_cash", "commission_bps", "slippage_bps", "threshold")
            },
            "notice": NOTICE,
        }
        _append(journal, store, "study_registered", entry)
        _confirm(journal, store, entry)
        return _entries(journal, store)[study_id]


def _assessment(entry, original, received, now):
    outcome = {
        "status": "pending",
        "reason": "target_not_observed",
        "metrics": None,
        "strategy": None,
    }
    if entry["status"] != "pending":
        return outcome | {"status": entry["status"], "reason": entry["reason"]}
    sessions, forecast = entry["sessions"], entry["forecast"]
    for key in (
        "instruments",
        "currency",
        "price_basis",
        "frequency",
        "provider_statement",
        "knowledge_basis",
    ):
        if original["contract"][key] != received["contract"][key]:
            raise ValueError("Observation changed the frozen source contract")
    for source in (original, received):
        if not re.fullmatch(
            rf"yahoo_{re.escape(entry['symbol'])}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}}-\d{{2}}-\d{{2}}_[a-f0-9]{{64}}",
            source["contract"]["source_id"],
        ):
            raise ValueError("Observation changed the frozen source provider")

    def endpoints(source):
        from quant_rsi.market.daily_data import NASDAQ_PAST_CALENDAR_URL, NYSE_PAST_CALENDAR_URL

        receipt = decode(source["source_files"]["collection_receipt.json"].encode())

        # A longer study requests two years instead of one. Only that bounded
        # range parameter may differ; hosts, paths and every other parameter stay bound.
        def window_url(url):
            return re.sub(r"(?<=[?&])range=(?:1y|2y)(?=&|$)", "range=retained_window", url)

        return [
            (window_url(row["url"]), window_url(row["response_url"]))
            for row in receipt["downloads"]
            # A later short receipt need not download the older calendar again.
            # The full published prediction interval is still matched below.
            if not (
                row["url"] in {NASDAQ_PAST_CALENDAR_URL, NYSE_PAST_CALENDAR_URL}
                and row["response_url"] == row["url"]
            )
        ]

    if endpoints(original) != endpoints(received):
        raise ValueError("Observation changed the frozen source endpoints")
    if timestamp(now) < timestamp(sessions[-1]["close_at"]):
        return outcome
    calendar = {s["session_id"]: s for s in received["contract"]["calendar"]["sessions"]}
    if any(calendar.get(s["session_id"]) != s for s in sessions):
        return outcome | {
            "status": "insufficient_evidence",
            "reason": "frozen_calendar_unavailable_or_changed",
        }
    positions = {day: i for i, day in enumerate(received["dates"])}
    if any(s["session_id"] not in positions for s in sessions):
        return outcome | {"reason": "missing_target_period_prices"}
    indices = [positions[s["session_id"]] for s in sessions]
    prices = [received["closes"][i] for i in indices]
    if prices[0] != forecast["reference_price"]:
        return outcome | {
            "status": "insufficient_evidence",
            "reason": "published_origin_price_revised",
        }
    actual_return = math.log(prices[-1] / prices[0])
    predicted_return = math.log(forecast["predicted_price"] / prices[0])
    data = MarketData(
        tuple(s["session_id"] for s in sessions),
        (entry["symbol"],),
        np.array([[received["opens"][i]] for i in indices]),
        np.array(prices).reshape(-1, 1),
    )
    row = forecast | {"observed_price": prices[-1], "predicted_log_return": predicted_return}
    strategy = evaluate_strategy(
        data,
        entry["symbol"],
        [row],
        horizon_sessions=forecast["horizon_sessions"],
        **entry["strategy_parameters"],
    )
    return outcome | {
        "status": "settled",
        "reason": None,
        "metrics": {
            "actual_price": prices[-1],
            "price_error": forecast["predicted_price"] - prices[-1],
            "absolute_price_error": abs(forecast["predicted_price"] - prices[-1]),
            "forecast": forecast_errors([predicted_return], [actual_return]),
            "unchanged_price": forecast_errors([0.0], [actual_return]),
        },
        "strategy": {key: strategy[key]["metrics"] for key in ("strategy", "buy_and_hold")},
    }


def observe_prices(directory, symbol, snapshot):
    """Assess all prior forecasts for one symbol; identical receipts are idempotent."""
    now = market_contracts.trusted_now()
    if timestamp(snapshot["info"]["received_at"]) > timestamp(now):
        raise ValueError("Observation precedes price receipt")
    with _ledger(directory) as (journal, store):
        reference = store.put_json(snapshot)
        for entry in _entries(journal, store).values():
            if entry["symbol"] != symbol or not entry["source_ref"]:
                continue
            original = store.json(entry["source_ref"])
            if timestamp(now) < timestamp(entry["issued_at"]) or timestamp(
                snapshot["info"]["received_at"]
            ) < timestamp(original["info"]["received_at"]):
                raise ValueError("Observation cannot roll back the publication or initial receipt")
            previous = entry["observations"][-1] if entry["observations"] else None
            if previous and previous["source_ref"] == reference:
                continue
            if previous and (
                timestamp(now) < timestamp(previous["observed_at"])
                or timestamp(snapshot["info"]["received_at"]) < timestamp(previous["received_at"])
            ):
                raise ValueError("Observation cannot roll back its clock or price receipt")
            outcome = _assessment(entry, original, snapshot, now)
            _append(
                journal,
                store,
                "study_observed",
                outcome
                | {
                    "study_id": entry["study_id"],
                    "observed_at": now,
                    "received_at": snapshot["info"]["received_at"],
                    "source_ref": reference,
                    "previous_ref": store.put_json(previous) if previous else None,
                    "notice": NOTICE,
                },
            )
    return history_view(directory, symbol)


def history_view(directory, symbol=None):
    if not (Path(directory) / "journal.sqlite").exists():
        return {"entries": [], "notice": NOTICE}
    with _ledger(directory) as (journal, store):
        entries = []
        for entry in _entries(journal, store).values():
            if symbol and entry["symbol"] != symbol:
                continue
            latest = entry["observations"][-1] if entry["observations"] else entry
            entries.append(
                {
                    key: entry[key]
                    for key in (
                        "study_id",
                        "run_directory",
                        "symbol",
                        "issued_at",
                        "selected_round",
                        "factor_ids",
                        "forecast",
                    )
                }
                | {
                    key: latest.get(key)
                    for key in ("status", "reason", "metrics", "strategy", "observed_at")
                }
                | {
                    "assessment_count": len(entry["observations"]),
                    "horizon_sessions": entry.get("horizon_sessions", 5),
                    "question": entry.get("question", ""),
                }
            )
        return {"entries": entries[-50:], "notice": NOTICE}


def mature_feedback(directory, symbol, *, horizon_sessions=5, question=""):
    view = history_view(directory, symbol)
    view["entries"] = [
        e
        for e in view["entries"]
        if e.get("horizon_sessions", 5) == horizon_sessions and e.get("question", "") == question
    ]
    return {
        "scope": "previously_issued_forecasts_observed_after_maturity; now_research_feedback",
        "notice": NOTICE,
        "horizon_sessions": horizon_sessions,
        "question": question,
        "factor_memory": recent_factor_recipes(
            directory, symbol, horizon_sessions=horizon_sessions, question=question
        ),
        "factor_ledger": factor_ledger(
            directory, symbol, horizon_sessions=horizon_sessions, limit=40
        ),
        "entries": [e for e in view["entries"] if e["status"] == "settled"][-20:],
        "history": {
            "entries": [
                {key: e[key] for key in ("study_id", "factor_ids", "forecast", "status", "reason")}
                for e in view["entries"]
            ],
            "notice": "Status and chosen factors only; outcome metrics are in entries above.",
        },
    }


def recent_factor_recipes(directory, symbol, *, horizon_sessions=5, question=""):
    """Read selected recipes through retained artifact references, never mutable run paths."""
    if not (Path(directory) / "journal.sqlite").exists():
        return []
    from quant_rsi.research.study_factor_memory import selected_definitions

    with _ledger(directory) as (journal, store):
        entries = [
            e
            for e in _entries(journal, store).values()
            if e["symbol"] == symbol
            and e.get("horizon_sessions", 5) == horizon_sessions
            and e.get("question", "") == question
        ]
        result = []
        for entry in reversed(entries[-5:]):
            definitions = entry.get("factor_definitions")
            input_mode = entry.get("input_mode", "unknown")
            if definitions is None:
                retained = store.json(entry["result_ref"])
                definitions = selected_definitions(retained)
                input_mode = retained.get("configuration", {}).get("prices", "unknown")
            latest = entry["observations"][-1] if entry["observations"] else entry
            result.append(
                {
                    "study_id": entry["study_id"],
                    "result_ref": entry["result_ref"],
                    "issued_at": entry["issued_at"],
                    "outcome_status": latest["status"],
                    "input_mode": input_mode,
                    "definitions": definitions,
                }
            )
        return result


def ledger_studies(directory, symbol, *, horizon_sessions=None):
    """Attempt records for every retained study; older entries read their frozen result."""
    if not (Path(directory) / "journal.sqlite").exists():
        return []
    from quant_rsi.research.study_factor_ledger import attempt_records, attempted_definitions

    with _ledger(directory) as (journal, store):
        studies = []
        for entry in _entries(journal, store).values():
            if entry["symbol"] != symbol or (
                horizon_sessions is not None
                and entry.get("horizon_sessions", 5) != horizon_sessions
            ):
                continue
            attempts = entry.get("attempts")
            definitions = entry.get("attempted_definitions")
            if attempts is None or definitions is None:
                retained = store.json(entry["result_ref"])
                attempts = attempt_records(retained)
                definitions = attempted_definitions(retained)
            latest = entry["observations"][-1] if entry["observations"] else entry
            studies.append(
                {
                    "study_id": entry["study_id"],
                    "run_directory": entry.get("run_directory"),
                    "issued_at": entry["issued_at"],
                    "question": entry.get("question", ""),
                    "horizon_sessions": entry.get("horizon_sessions", 5),
                    "outcome_status": latest["status"],
                    "attempts": attempts,
                    "definitions": definitions,
                }
            )
        return studies


def factor_ledger(directory, symbol, *, horizon_sessions=None, limit=None):
    from quant_rsi.research.study_factor_ledger import summarize

    studies = ledger_studies(directory, symbol, horizon_sessions=horizon_sessions)
    return summarize(studies, limit=limit) | {
        "symbol": symbol,
        "horizon_sessions": horizon_sessions,
        "study_count": len(studies),
    }
