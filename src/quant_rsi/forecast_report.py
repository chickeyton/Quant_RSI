"""Offline HTML presentation of already verified enhanced-forecast reports.

The caller must obtain status/replay reports through the existing verification APIs.
Rendering performs no verification, I/O, fitting, metric recomputation or publication.
"""

import json
import math
from html import escape


def _text(value):
    if value is None:
        return "Unavailable"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
    return escape(str(value), quote=True)


def _number(value, *, places=4, scale=1, suffix=""):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not math.isfinite(value * scale):
        return None
    return f"{value * scale:,.{places}f}{suffix}"


def _table(headers, rows):
    head = "".join(f'<th scope="col">{_text(value)}</th>' for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _fields(rows):
    return (
        "<dl>"
        + "".join(f"<dt>{_text(label)}</dt><dd>{_text(value)}</dd>" for label, value in rows)
        + "</dl>"
    )


def _section(title, content):
    return f"<section><h2>{_text(title)}</h2>{content}</section>"


def _campaign_budget(workflow, campaign):
    if campaign is None:
        reason = (workflow.get("failure") or {}).get("budget_unavailable_reason")
        return (
            "<p>Parent campaign budget report unavailable. No remaining allowance is inferred.</p>"
            + (_fields([("Budget unavailable reason", reason)]) if reason is not None else "")
        )
    binding = workflow.get("managed_run")
    if (
        not isinstance(binding, dict)
        or not binding
        or campaign.get("kind") != "ForecastCampaignReport"
        or binding.get("run_id") != workflow.get("workflow_id")
        or binding.get("workspace") != workflow.get("workspace")
        or any(
            key not in binding or campaign.get(key) != binding[key]
            for key in ("authority_ref", "campaign_ref", "campaign_id", "workspace")
        )
    ):
        raise ValueError("campaign report does not match the workflow's managed identity")
    runs = [row for row in campaign.get("runs", []) if row.get("run_id") == binding["run_id"]]
    if len(runs) != 1 or runs[0].get("binding") != binding:
        raise ValueError("campaign report does not bind exactly this managed run")
    run = runs[0]
    budget, allocated = campaign.get("budget", {}), campaign.get("allocated", {})
    actual, allocation, run_actual = (
        campaign.get("actual", {}),
        run.get("allocation", {}),
        run.get("actual", {}),
    )
    rows = [
        (
            "Registered attempt ceiling",
            budget.get("max_fit_reservations"),
            allocation.get("max_attempts"),
        ),
        (
            "Registered worker-slot seconds ceiling",
            budget.get("max_worker_seconds"),
            allocation.get("max_worker_seconds"),
        ),
        (
            "Attempt reservations",
            actual.get("fit_reservations"),
            run_actual.get("fit_reservations"),
        ),
        ("Charged fit slots", actual.get("charged_fits"), run_actual.get("charged_fits")),
        (
            "Charged worker-slot seconds",
            actual.get("reserved_worker_seconds"),
            run_actual.get("reserved_worker_seconds"),
        ),
    ]
    return (
        _fields(
            [
                ("Campaign", campaign.get("campaign_id")),
                ("Campaign allocated attempt ceiling", allocated.get("max_attempts")),
                (
                    "Campaign allocated worker-slot seconds ceiling",
                    allocated.get("max_worker_seconds"),
                ),
            ]
        )
        + _table(("Measure", "Whole campaign", "This run"), rows)
        + "<p>Worker-slot seconds are registered timeout charges, not measured runtime. "
        "Charged fit slots are distinct from all attempt reservations. "
        "Unused allocations are not presented as transferable or refunded.</p>"
    )


def render_forecast_report(workflow_report: dict, *, campaign_report: dict | None = None) -> str:
    """Return standalone HTML without reading or mutating any external or input state."""
    experiment = workflow_report.get("experiment") or {}
    forecast = experiment.get("forecast") or {}
    comparison = workflow_report.get("development_comparison") or {}
    selection = experiment.get("selection") or {}
    config = workflow_report.get("config") or {}
    target = config.get("target") or forecast.get("instrument")
    sections = [
        _section(
            "Workflow",
            _fields(
                [
                    ("Workflow ID", workflow_report.get("workflow_id")),
                    ("Status", workflow_report.get("status")),
                    ("Target", target),
                    ("Industry proxy", config.get("industry")),
                    ("Market proxy", config.get("market")),
                    ("Workspace", workflow_report.get("workspace")),
                ]
            ),
        )
    ]
    failure = workflow_report.get("failure")
    if failure:
        details = [
            ("Category", failure.get("category")),
            ("Failure stage", failure.get("stage")),
            ("Message", failure.get("message")),
            ("Recoverable", failure.get("recoverable")),
            ("Budget unavailable reason", failure.get("budget_unavailable_reason")),
        ]
        command = failure.get("recovery_command")
        if failure.get("recoverable") is True and isinstance(command, str) and command:
            details.append(("Supplied safe recovery command", command))
        sections.append(_section("Reported failure and recovery", _fields(details)))

    forecast_content = ""
    if not forecast:
        forecast_content += "<p>No committed publication forecast is available in this report.</p>"
    forecast_content += _fields(
        [
            ("Publication status", forecast.get("status")),
            ("Observed origin session", forecast.get("origin_session")),
            ("Observed origin close (UTC)", forecast.get("origin_close_at")),
            ("Observed origin price", _number(forecast.get("origin_price"))),
            ("Forecast target session", forecast.get("target_session")),
            ("Forecast target close (UTC)", forecast.get("target_close_at")),
            ("Horizon (registered sessions)", forecast.get("horizon_sessions")),
            ("Forecast price", _number(forecast.get("forecast_price"))),
            (
                "Forecast simple return",
                _number(forecast.get("forecast_return"), scale=100, suffix="%"),
            ),
            ("Currency", forecast.get("currency")),
            ("Price unit", forecast.get("price_unit")),
            ("Price-unit qualification", forecast.get("price_unit_claim")),
            ("Abstention or failure reasons", forecast.get("reasons")),
        ]
    )
    sections.append(_section("Observed price and forecast", forecast_content))

    sections.append(
        _section(
            "Timing and eligibility",
            _fields(
                [
                    ("Recorded mode", forecast.get("mode")),
                    ("Confirmed forward qualification", forecast.get("forward")),
                    ("Frozen input as of (UTC)", forecast.get("frozen_input_asof")),
                    (
                        "Training information cutoff (UTC)",
                        forecast.get("training_information_cutoff"),
                    ),
                    ("Model effective at (UTC)", forecast.get("model_effective_at")),
                    ("Forecast issued at (UTC)", forecast.get("issued_at")),
                    ("Post-commit confirmation (UTC)", forecast.get("post_commit_confirmed_at")),
                    ("Forward deadline (UTC)", forecast.get("forward_deadline")),
                ]
            )
            + "<p>Timing qualifications are reported as supplied. A late, retrospective, "
            "unconfirmed or abstained result is not presented as a timely forward forecast.</p>",
        )
    )

    sections.append(
        _section(
            "Selected recipe and actual model inputs",
            _fields(
                [
                    ("Development-selected recipe", selection.get("selected_recipe")),
                    ("Publication recipe", forecast.get("recipe_group")),
                    ("Actual feature names", forecast.get("feature_names")),
                    ("Actual model input groups", forecast.get("model_inputs")),
                    (
                        "Sidecar inputs used in training",
                        forecast.get("sidecar_inputs_used_in_training"),
                    ),
                    ("Publication training rows", forecast.get("training_rows")),
                    ("Model reference", forecast.get("model_ref")),
                    ("Feature-view reference", forecast.get("feature_view_ref")),
                ]
            )
            + "<p>Feature names and input groups identify what this model used. Numeric feature "
            "values are not included in the status report and are not retrieved by this page.</p>",
        )
    )

    selected = comparison.get("selected_development_metrics") or {}
    rows = [
        (
            "Selected development model: "
            + str(comparison.get("selected_recipe") or "Unavailable"),
            _number(selected.get("mae_pp")),
            _number(selected.get("rmse_pp")),
            selected.get("observations"),
            "Not applicable",
            "Not applicable",
        )
    ]
    controls = comparison.get("naive_controls") or {}
    verdicts = comparison.get("selected_beats_each_control") or {}
    for key, label in (
        ("unchanged_price", "Unchanged-price control"),
        ("training_mature_label_mean", "Same-training-sample mean control"),
    ):
        metrics, checks = controls.get(key) or {}, verdicts.get(key) or {}
        verdict = [
            "Yes"
            if checks.get(metric) is True
            else "No — model did not beat this control"
            if checks.get(metric) is False
            else None
            for metric in ("mae_pp", "rmse_pp")
        ]
        rows.append(
            (
                label,
                _number(metrics.get("mae_pp")),
                _number(metrics.get("rmse_pp")),
                metrics.get("observations"),
                *verdict,
            )
        )
    sections.append(
        _section(
            "Development errors and same-row naive controls",
            _table(
                (
                    "Model or control",
                    "MAE (pp)",
                    "RMSE (pp)",
                    "Eligible rows",
                    "Model beats control MAE?",
                    "Model beats control RMSE?",
                ),
                rows,
            )
            + "<p>Errors are simple-return percentage points on the selected recipe's "
            "same eligible evaluation rows. Missing metrics and counts remain unavailable. "
            "Recipe selection "
            "does not establish out-of-sample predictive advantage.</p>"
            + _fields([("Comparison qualification", comparison.get("qualification"))]),
        )
    )

    sections.append(
        _section(
            "Separate publication-model validity",
            _fields(
                [
                    (
                        "Publication-model validation metrics",
                        forecast.get("publication_model_validation_metrics"),
                    ),
                    (
                        "Publication model inherits development scores",
                        forecast.get("publication_model_inherits_evaluation_scores"),
                    ),
                    (
                        "Predictive validity established",
                        comparison.get("predictive_validity_established"),
                    ),
                ]
            )
            + "<p>When available, the publication model comes from a separate fit. "
            "Development errors above do not validate that model. This is an uncalibrated "
            "point forecast; no confidence interval or economic advantage is inferred.</p>",
        )
    )
    sections.append(
        _section(
            "Coverage and qualifications",
            _fields(
                [
                    ("Coverage", workflow_report.get("coverage")),
                    ("Input/protocol qualifications", experiment.get("qualifications")),
                    ("Publication qualifications", forecast.get("qualifications")),
                    ("Corporate-action evidence", forecast.get("action_evidence")),
                ]
            ),
        )
    )
    sections.append(
        _section("Registered allowance", _campaign_budget(workflow_report, campaign_report))
    )

    attempts = experiment.get("attempts") or []
    sections.append(
        _section(
            "Attempt outcomes",
            _table(
                (
                    "Ordinal",
                    "Name",
                    "Status",
                    "Worker-slot seconds",
                    "Measured elapsed seconds",
                    "Error",
                ),
                [
                    (
                        row.get("ordinal"),
                        row.get("name"),
                        row.get("status"),
                        row.get("worker_seconds"),
                        _number(row.get("elapsed_seconds")),
                        row.get("error"),
                    )
                    for row in attempts
                ],
            )
            if attempts
            else "<p>No attempt rows are available in this report.</p>",
        )
    )
    sections.append(
        _section(
            "Report identity",
            _fields(
                [
                    ("Config reference", workflow_report.get("config_ref")),
                    ("Protocol reference", workflow_report.get("protocol_ref")),
                    ("Input reference", experiment.get("input_ref")),
                    ("Forecast reference", forecast.get("forecast_ref")),
                    ("Source references", forecast.get("source_refs")),
                ]
            ),
        )
    )
    title = "Forecast report — " + str(target or "Unavailable")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_text(title)}</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem auto;padding:0 1rem;max-width:72rem;"
        "line-height:1.5;color:#18202b;background:#fff}h1,h2{line-height:1.2}"
        "section{margin:2rem 0;border-top:1px solid #ccd3db;padding-top:1rem}"
        "dl{display:grid;grid-template-columns:minmax(10rem,1fr) 3fr;gap:.4rem 1rem}"
        "dt{font-weight:600}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}"
        ".table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccd3db;padding:.5rem;text-align:left;vertical-align:top;"
        "white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf2f7}"
        "@media(max-width:40rem){dl{display:block}dd{margin-bottom:.8rem}}"
        "</style></head><body><main>"
        f"<h1>{_text(title)}</h1>"
        "<p>Historical comparisons and publication qualifications are shown separately.</p>"
        + "".join(sections)
        + "</main></body></html>"
    )
