"""Cheap evidence refresh before paid research; receipts alone are not new evidence."""

import json

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.research.demo_news import issuer_news
from quant_rsi.research.demo_related import profile_related
from quant_rsi.research.study_financials import collect_financials, reported_records
from quant_rsi.research.study_history import collection_snapshot, mature_feedback, observe_prices
from quant_rsi.research.study_resume import binding


def refresh_evidence(task, manifest, *, on_stage=None):
    root = task.parent.parent
    symbol, mode = manifest["symbol"], manifest["data"]
    history = root / "history" / mode
    options = {
        k: manifest.get(k)
        for k in (
            "symbol",
            "data",
            "llm",
            "llm_model",
            "model",
            "rounds",
            "max_fits",
            "patience",
            "selection_metric",
            "horizon_sessions",
            "historical_sessions",
            "walk_forward_folds",
            "question",
            "news_research",
            "financials",
            "financials_source",
            "external_drivers",
        )
    }
    arguments = list(manifest["arguments"])
    profile_root = arguments[arguments.index("--factor-root") + 1]
    evidence = {
        "configuration": options,
        "runtime": binding(
            {"factor_root": profile_root, "financials": manifest.get("financials", False)}
        ),
    }
    if mode == "daily":
        from quant_rsi.research.study_walk_forward import price_history_sessions

        if on_stage:
            on_stage("collecting_prices")
        prices = task / "preflight/prices"
        horizon = manifest.get("horizon_sessions", 5)
        history_sessions = price_history_sessions(
            horizon, 20, manifest.get("walk_forward_folds", 0), manifest.get("historical_sessions")
        )
        snapshot = collection_snapshot(
            symbol,
            prices,
            horizon_sessions=horizon,
            historical_sessions=history_sessions,
        )
        observe_prices(history, symbol, snapshot)
        evidence["prices"] = {k: snapshot[k] for k in ("dates", "opens", "closes")}
        arguments += ["--prices-dir", str(prices)]
        if on_stage:
            on_stage("collecting_news")
        news = issuer_news({"symbol": symbol, "related": profile_related(symbol)})
        atomic_write(task / "preflight/news.json", canonical(news))
        arguments += ["--news-snapshot", str(task / "preflight/news.json")]
        evidence["news"] = sorted(
            [
                {
                    k: row.get(k)
                    for k in ("symbol", "title", "summary", "url", "published_at", "published_date")
                }
                for row in news["items"]
            ],
            key=lambda row: canonical(row),
        )
        evidence["news_coverage"] = [
            {k: row.get(k) for k in ("symbol", "status")} for row in news.get("coverage", [])
        ]
        if manifest.get("financials"):
            if on_stage:
                on_stage("collecting_financials")
            directory = task / "preflight/financials"
            snapshot = collect_financials(
                symbol, directory, profile_root, manifest.get("financials_source", "auto")
            )
            evidence["financials"] = sorted(
                [
                    {k: v for k, v in row.items() if k != "page_modified_date"}
                    for row in reported_records(snapshot)
                ],
                key=canonical,
            )
            arguments += ["--financials-dir", str(directory)]
        if manifest.get("external_drivers"):
            from quant_rsi.research.eia_petroleum import collect_petroleum

            if on_stage:
                on_stage("collecting_external_drivers")
            directory = task / "preflight/external_drivers"
            dates = evidence["prices"]["dates"]
            cache_flag = "--external-drivers-cache-dir"
            shared = arguments[arguments.index(cache_flag) + 1] if cache_flag in arguments else None
            external = collect_petroleum(directory, dates[0], dates[-1], cache_directory=shared)
            evidence["external_drivers"] = {
                "records": external["records"],
                "failures": external["failures"],
            }
            arguments += ["--external-drivers-dir", str(directory)]
    if on_stage:
        on_stage("checking_new_evidence")
    evidence["matured"] = [
        {k: entry[k] for k in ("study_id", "status", "metrics")}
        for entry in mature_feedback(
            history,
            symbol,
            horizon_sessions=manifest.get("horizon_sessions", 5),
            question=manifest.get("question", ""),
        )["entries"]
    ]
    fingerprint = digest(canonical(evidence))
    previous = None
    for receipt in root.glob("tasks/*/trigger.json"):
        state = receipt.parent / "worker.json"
        if receipt.parent == task or not state.exists():
            continue
        if json.loads(state.read_bytes()).get("status") != "completed":
            continue
        old = json.loads(receipt.read_bytes())
        if old["fingerprint"] == fingerprint:
            previous = receipt.parent.name
            break
    return {
        "fingerprint": fingerprint,
        "previous_completed_task": previous,
        "reason": "unchanged_evidence" if previous else "new_evidence_or_configuration",
        "source_digests": {k: digest(canonical(v)) for k, v in evidence.items()},
    }, arguments
