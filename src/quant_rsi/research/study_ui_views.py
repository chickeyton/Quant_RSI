"""Small visual projections of retained study outputs; never used by research selection."""

import math
import re

from quant_rsi.research.study_explanation import (
    continuity,
    evidence_path,
    explain_prediction,
    question_coverage,
)


def _pick(value, *keys):
    return {key: value.get(key) for key in keys}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _news(items):
    return [
        {
            **_pick(
                item,
                "id",
                "title",
                "url",
                "published_at",
                "published_date",
                "received_at",
                "status",
                "reason",
                "final_url",
                "fetch_attempted",
                "content_extent",
                "symbol",
            ),
            "excerpt": str(item.get("text") or item.get("summary") or "")[:600],
            "excerpt_shortened": len(str(item.get("text") or item.get("summary") or "")) > 600,
        }
        for item in items[:24]
    ]


def _evidence(ids, previous):
    available = {}
    for stage, value in previous.items():
        items = (
            value
            if stage.startswith("article_reads_")
            else value.get("items", [])
            if stage == "news"
            else []
        )
        for item in items:
            if item.get("id"):
                available[item["id"]] = item
    return _news(
        [
            available.get(key, {"id": key, "title": "Evidence not present in this trace"})
            for key in ids
        ]
    )


def _factors(items):
    return [
        _pick(
            item,
            "factor_id",
            "kind",
            "description",
            "input_kind",
            "recipe",
            "reuse_lifetime",
            "runnable",
            "availability_reason",
            "reasons",
            "definition_id",
            "definition_scope",
            "definition_owner",
            "binding_id",
            "source",
            "input_factor_ids",
        )
        for item in items[:64]
    ]


def factor_changes(before, after):
    """IDs define identity. Pair unique same-operator replacements only for recipe comparison."""
    old = {item["factor_id"]: item for item in _factors(before)}
    new = {item["factor_id"]: item for item in _factors(after)}
    changes = []
    for key in sorted(old.keys() & new.keys()):
        changes.append(
            {
                "type": "retained" if old[key]["recipe"] == new[key]["recipe"] else "changed",
                "before": old.pop(key),
                "after": new.pop(key),
            }
        )

    def operator(item):
        return item["kind"], item["input_kind"], (item["recipe"] or {}).get("op")

    for key, item in list(old.items()):
        same_old = [v for v in old.values() if operator(v) == operator(item)]
        same_new = [v for v in new.values() if operator(v) == operator(item)]
        if len(same_old) == len(same_new) == 1 and operator(item)[-1]:
            replacement = same_new[0]
            changes.append(
                {
                    "type": "replaced",
                    "before": old.pop(key),
                    "after": new.pop(replacement["factor_id"]),
                }
            )
    changes.extend({"type": "removed", "before": item, "after": None} for item in old.values())
    changes.extend({"type": "added", "before": None, "after": item} for item in new.values())
    return changes


def prediction_rows(current, initial=None):
    old = {
        (row["origin_session"], row["target_session"]): row
        for row in (initial or {}).get("development", {}).get("rows", [])
    }
    rows = []
    for row in current.get("development", {}).get("rows", []):
        prior = old.get((row["origin_session"], row["target_session"]), {})
        # Never join by row position or compare attempts with different observed outcomes.
        comparable = prior.get("observed_price") == row.get("observed_price")
        rows.append(
            {
                "origin": row["origin_session"],
                "date": row["target_session"],
                "observed": row.get("observed_price"),
                "prediction": row.get("predicted_price"),
                "initial": prior.get("predicted_price") if comparable else None,
                "baseline": row.get("reference_price"),
            }
        )
    return rows


def _accounts(previous, selected_round=None):
    first = previous.get("first_prediction", {}).get("strategy", {})
    second = previous.get("second_prediction", {}).get("strategy", {})
    accounts = [
        ("Initial", first.get("strategy", {})),
        ("Revised", second.get("strategy", {})),
        ("Buy & hold", first.get("buy_and_hold", {})),
    ]
    if selected_round is not None and selected_round > 2:
        selected = previous.get(f"round_{selected_round}_prediction", {}).get("strategy", {})
        accounts.insert(2, ("Selected", selected.get("strategy", {})))
    return _account_series(accounts)


def _account_series(accounts):
    metrics = [
        {
            "label": label,
            **_pick(
                account.get("metrics", {}),
                "net_return",
                "sharpe_ratio",
                "max_drawdown",
                "risk_statistics",
            ),
        }
        for label, account in accounts
    ]
    by_date = {}
    for label, account in accounts:
        for valuation in account.get("valuations", []):
            row = by_date.setdefault(valuation["session_id"], {"date": valuation["session_id"]})
            row[label] = valuation.get("nav") if valuation.get("status") == "complete" else None
    return metrics, [by_date[key] for key in sorted(by_date)]


def _terminal_view(output):
    evaluation = output.get("evaluation", {})
    strategy = output.get("strategy", {})
    accounts, nav = _account_series(
        [
            ("Selected", strategy.get("strategy", {})),
            ("Buy & hold", strategy.get("buy_and_hold", {})),
        ]
    )
    return {
        "kind": "terminal",
        **_pick(output, "status", "selected_round", "sample_count", "limitation", "missing_inputs"),
        "metrics": evaluation.get("metrics", {}),
        "baselines": {k: v["metrics"] for k, v in evaluation.get("baselines", {}).items()},
        "rows": prediction_rows({"development": evaluation}),
        "accounts": accounts,
        "nav": nav,
    }


def visual_output(stage, output, previous, symbol):
    if stage == "restore_factor_recipes":
        return {
            "kind": "facts",
            "facts": {
                e["factor_id"]: e["status"].replace("_", " ")
                + (
                    " · reuse until " + e["reuse_lifetime"]["expires_at"]
                    if e.get("reuse_lifetime") and "expires_at" in e["reuse_lifetime"]
                    else ""
                )
                for e in output["entries"]
            },
            "note": output["notice"],
        }
    if stage == "check_inputs":
        financials = output["financials"]
        external = output.get("external_drivers", {})
        return {
            "kind": "facts",
            "facts": {
                "price_source": output["prices"]["provider"],
                "first_session": output["prices"]["first_session"],
                "last_session": output["prices"]["last_session"],
                "historical_sessions": output["prices"].get("historical_sessions"),
                "training_rows": output["prices"]["training_rows"],
                "development_rows": output["prices"]["development_rows"],
                "news_source": output["news"]["provider"],
                "news_items": output["news"]["items"],
                "financial_source": financials["provider"],
                **{
                    d["factor_id"]: "Available"
                    if d["runnable"]
                    else f"Unavailable: {d['missing_required_rows']} missing required rows"
                    for d in financials["definitions"] + external.get("definitions", [])
                },
            },
            "coverage_gaps": financials.get("coverage_gaps", []),
            "sample_coverage": financials.get("sample_coverage", [])
            + external.get("sample_coverage", []),
            "note": output["notice"]
            + " "
            + financials["notice"]
            + " "
            + external.get("notice", ""),
        }
    if stage in {"load_financials", "load_external_drivers"}:
        return {
            "kind": "factors",
            "factors": _factors(output.get("definitions", [])),
            "coverage_gaps": output.get("coverage_gaps", []),
            "sample_coverage": output.get("sample_coverage", []),
            "catalog": True,
            "note": output["notice"]
            + " Source: "
            + output.get("provider", "unknown")
            + ". "
            + "; ".join(
                a["provider"] + ": " + a["status"] + " " + a.get("reason", "")
                for a in output.get("source_attempts", [])
            ),
        }
    if stage in {"financial_comparison", "external_comparison"}:
        family = "financials" if stage == "financial_comparison" else "external_drivers"
        label = "financials" if family == "financials" else "external drivers"
        return {
            "kind": "engine_comparison",
            "rows": [
                {
                    "window": "development",
                    "engine": label,
                    "mae": output[key]["forecast"]["mae_price"],
                    **_pick(output[key]["strategy"], "net_return", "sharpe_ratio", "max_drawdown"),
                }
                for key, label in (
                    (f"with_{family}", f"Selected + {label}"),
                    (f"without_{family}", f"{label.capitalize()} removed"),
                )
                if output["status"] == "completed"
            ],
            "note": output["notice"] + f" Additional fits: {output['extra_fits']}.",
        }
    if stage == "walk_forward":
        rows = []
        for fold in output["folds"]:
            for account in ("strategy", "buy_and_hold"):
                completed = fold["status"] == "completed"
                metrics = fold["strategy"][account]["metrics"] if completed else {}
                rows.append(
                    {
                        "window": f"{fold['first_origin']} → {fold['last_target']}",
                        "engine": account if completed else "unavailable",
                        "mae": fold["evaluation"]["metrics"]["mae_price"]
                        if completed and account == "strategy"
                        else None,
                        **_pick(metrics, "net_return", "sharpe_ratio", "max_drawdown"),
                    }
                )
                if not completed:
                    break
        return {
            "kind": "engine_comparison",
            "rows": rows,
            "extra_fits": output["extra_fits"],
            "note": output["notice"] + f" Additional fits: {output['extra_fits']}.",
        }
    if stage == "qlib_comparison":
        rows = []
        for segment, values in output["segments"].items():
            if values["status"] != "completed":
                continue
            for engine in ("native", "qlib"):
                value = values[engine]
                metrics = value["strategy"]["strategy"]["metrics"]
                rows.append(
                    {
                        "window": segment,
                        "engine": engine,
                        "mae": value["metrics"]["mae_price"],
                        **_pick(metrics, "net_return", "sharpe_ratio", "max_drawdown"),
                    }
                )
        return {"kind": "engine_comparison", "rows": rows, "note": output["notice"]}
    if stage == "observe_prior_forecasts":
        if "prior_outcomes" in output:
            return {"kind": "continuity", **continuity(output["prior_outcomes"])}
        return {
            "kind": "facts",
            "facts": {key: value for key, value in output.items() if key != "notice"},
            "note": output.get("notice"),
        }
    if stage == "terminal_evaluation":
        return _terminal_view(output)
    match = re.fullmatch(
        r"round_(\d+)_(hypothesis|factor_library|factors|prediction|feedback|decision)", stage
    )
    if match:
        number, suffix = int(match[1]), match[2]
        if suffix == "decision":
            return {"kind": "iteration", **output}
        previous = dict(previous)
        feedback = previous.get(f"round_{number}_feedback", {})
        incumbent = feedback.get(
            "incumbent_round", output.get("incumbent_round", 1) if isinstance(output, dict) else 1
        )
        prefix = {1: "initial", 2: "revised"}.get(incumbent, f"round_{incumbent}")
        prediction_stage = {1: "first_prediction", 2: "second_prediction"}.get(
            incumbent, f"round_{incumbent}_prediction"
        )
        previous["initial_factors"] = previous.get(f"{prefix}_factors", [])
        previous["revised_hypothesis"] = previous.get(f"round_{number}_hypothesis", {})
        previous["revised_factors"] = previous.get(f"round_{number}_factors", [])
        previous["revised_factor_library"] = previous.get(f"round_{number}_factor_library", {})
        if suffix == "feedback":
            previous["first_prediction"] = previous.get(prediction_stage, {})
        mapped = {"prediction": "second_prediction", "feedback": "feedback"}.get(
            suffix, f"revised_{suffix}"
        )
        view = visual_output(mapped, output, previous, symbol)
        view["attempt"] = f"round_{number}"
        return view
    if stage == "load_prices":
        return {"kind": "facts", "facts": _pick(output, "symbol", "provider", "sessions")}
    if stage == "extract_guidance":
        extraction = output.get("latest_extraction") or {}
        items = {item["id"]: item for item in extraction.get("items", [])}
        return {
            "kind": "findings",
            "findings": [
                {
                    "text": f["status"] + ": " + f["quote"],
                    "evidence": _news([items.get(f["news_id"], {"id": f["news_id"]})]),
                }
                for f in extraction.get("facts", [])
            ],
            "related": [],
            "followup": output["notice"] + " Status: " + output["status"],
        }
    if stage in {"collect_news_context", "load_related_prices"}:
        return {"kind": "relationships", "target": symbol, **output["related"]}
    if stage == "target":
        return {
            "kind": "facts",
            "facts": _pick(
                output,
                "symbol",
                "reference_price",
                "origin_session",
                "evaluation_sessions",
                "model_mode",
            ),
            "note": output.get("model_note"),
        }
    if stage == "related_stocks":
        return {"kind": "relationships", "target": symbol, **output}
    if stage == "news" or stage.startswith(("news_search_", "article_reads_")):
        items = (
            output if isinstance(output, list) else output.get("items", output.get("results", []))
        )
        return {
            "kind": "news",
            "items": _news(items),
            "count": len(items),
            "query": output.get("query") if isinstance(output, dict) else None,
            "note": output.get("rationale") if isinstance(output, dict) else None,
            "query_origin": output.get("query_origin") if isinstance(output, dict) else None,
        }
    if stage.startswith("news_review_"):
        return {
            "kind": "findings",
            "findings": [
                {
                    "text": item["text"],
                    "evidence": _evidence(item.get("evidence_ids", []), previous),
                }
                for item in output.get("findings", [])
            ],
            "related": output.get("related_companies", []),
            "followup": output.get("followup_query"),
            "recovery_query": output.get("recovery_query"),
            "recovery_reason": output.get("recovery_reason"),
        }
    if stage in {"initial_hypothesis", "revised_hypothesis"}:
        return {
            "kind": "hypothesis",
            "text": output.get("hypothesis"),
            "question_map": question_coverage(output),
            "factors": output.get("factor_ids", []),
            "evidence": _evidence(output.get("evidence_news_ids", []), previous),
            "feedback_used": output.get("feedback_used"),
        }
    if stage in {"factor_library", "initial_factor_library", "revised_factor_library"}:
        return {
            "kind": "factors",
            "factors": _factors(output.get("definitions", [])),
            "count": len(output.get("definitions", [])),
            "note": output.get("execution_note"),
            "catalog": True,
        }
    if stage in {"initial_factors", "revised_factors"}:
        revised = stage == "revised_factors"
        hypothesis = previous.get("revised_hypothesis" if revised else "initial_hypothesis", {})
        return {
            "kind": "factor_changes" if revised else "factors",
            "factors": _factors(output),
            "changes": factor_changes(previous.get("initial_factors", []), output)
            if revised
            else [],
            "note": hypothesis.get("hypothesis"),
            "count": len(output),
        }
    if stage in {"first_prediction", "second_prediction"}:
        initial = previous.get("first_prediction") if stage == "second_prediction" else None
        prefix = "revised" if stage == "second_prediction" else "initial"
        development = output.get("development", {})
        return {
            "kind": "prediction",
            "attempt": "revised" if stage == "second_prediction" else "initial",
            "rows": prediction_rows(output, initial),
            "metrics": development.get("metrics", {}),
            "initial_metrics": (initial or {}).get("development", {}).get("metrics"),
            "baseline_metrics": development.get("baselines", {})
            .get("unchanged_price", {})
            .get("metrics", {}),
            "currency": development.get("currency"),
            "horizon": output.get("horizon_sessions"),
            "note": development.get("evaluation_note"),
            "input_usage": output.get("input_usage", []),
            "evidence_path": explain_prediction(
                output,
                previous.get(f"{prefix}_hypothesis", {}),
                previous.get(f"{prefix}_factors", []),
                previous.get(f"{prefix}_factor_library", previous.get("factor_library", {})),
                previous.get("news", {}),
            ),
        }
    if stage == "feedback":
        rows = prediction_rows(previous.get("first_prediction", {}))
        errors = [
            {**row, "error": row["prediction"] - row["observed"]}
            for row in rows
            if _number(row["prediction"]) and _number(row["observed"])
        ]
        return {
            "kind": "feedback",
            "metrics": output.get("development", {}).get("metrics", {}),
            "errors": sorted(errors, key=lambda row: abs(row["error"]), reverse=True)[:5],
            "latest": _pick(
                output, "predicted_price", "observed_price", "error", "observed_session"
            ),
            "note": output.get("development", {}).get("evaluation_note"),
        }
    if stage == "summary":
        accounts, nav = _accounts(previous, (output.get("selection") or {}).get("selected_round"))
        return {
            "kind": "summary",
            "text": output.get("text"),
            "selection": output.get("selection"),
            "accounts": accounts,
            "nav": nav,
            "rounds": output.get("research_rounds", []),
            "stop_reason": output.get("stop_reason"),
            "evidence_paths": evidence_path(previous),
            "continuity": continuity(
                previous.get("target", {}).get("prior_outcomes"),
                (output.get("selection") or {}).get("selected_factor_ids"),
            ),
            "terminal": _terminal_view(previous["terminal_evaluation"])
            if "terminal_evaluation" in previous
            else None,
        }
    return {"kind": "unknown"}
