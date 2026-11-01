"""Portable research reports: local HTML/Markdown plus the complete JSON evidence."""

import json
from html import escape

from quant_rsi.market.market_contracts import timestamp
from quant_rsi.research.study_explanation import continuity, evidence_path


def _cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def _number(value, percent=False):
    if value is None:
        return "N/A"
    return f"{float(value) * (100 if percent else 1):.4f}" + ("%" if percent else "")


def _reuse_deadline(lifetime):
    if not lifetime or not lifetime.get("expires_at"):
        return "No expiry declared"
    try:
        return timestamp(lifetime["expires_at"]).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return "Invalid recorded deadline"


def render_study(result):
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = (steps[key] for key in ("first_prediction", "second_prediction"))
    iterations = result.get("iterations")
    if iterations:
        second = iterations["attempts"][-1]["prediction"]
    md, html = [], []

    def section(title, text):
        md.extend([f"## {title}", "", text, ""])
        html.append(f"<section><h2>{escape(title)}</h2><p>{escape(text)}</p></section>")

    def table(headers, rows):
        md.extend(
            ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        )
        md.extend("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows)
        md.append("")
        html.append(
            "<div class='table'><table><thead><tr>"
            + "".join(f"<th>{escape(str(x))}</th>" for x in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table></div>"
        )

    title = result["target"] + " · Connected research study"
    md.extend([f"# {title}", "", result["notice"], ""])
    html.append(
        f"<header><p class='eyebrow'>QUANT RSI / RESEARCH STUDY</p><h1>{escape(title)}</h1>"
        f"<p>{escape(result['notice'])}</p></header>"
    )
    section("Conclusion", result["study"]["conclusion"])
    selection = result["study"]["selection"]
    section(
        "Attempt selection",
        selection["reason"] + " " + selection["forecast_note"] + " "
        "This is development selection after feedback, not independent validation.",
    )
    table(
        [
            "Preselected metric",
            "Initial",
            "Selected score" if iterations else "Revised",
            "Selected attempt",
            "Selected factors",
        ],
        [
            [
                selection["metric"] + " (" + selection["direction"] + " is better)",
                _number(selection["initial_score"]),
                _number(selection["selected_score"] if iterations else selection["revised_score"]),
                selection["selected_attempt"],
                ", ".join(selection["selected_factor_ids"]),
            ]
        ],
    )
    section(
        "What ran",
        f"{iterations['completed_fits'] if iterations else 2} research model fits "
        "and a fixed development "
        "window. Results include failed or missing source coverage. No orders were sent.",
    )
    if iterations:
        section(
            "Research rounds",
            f"Stopped: {iterations['stop_reason']}. "
            f"Fit allowance: {iterations['fit_budget']}; selected round: "
            f"{iterations['selected_round']}. Every trial uses development data only.",
        )
        table(
            ["Round", "Factors", "Price MAE", "Net return", "Sharpe", "Accepted"],
            [
                [
                    item["round"],
                    ", ".join(item["factor_ids"]),
                    _number(item["development_metrics"].get("mae_price")),
                    _number(
                        item["prediction"]["strategy"]["strategy"]["metrics"]["net_return"], True
                    ),
                    _number(item["prediction"]["strategy"]["strategy"]["metrics"]["sharpe_ratio"]),
                    item["accepted"],
                ]
                for item in iterations["attempts"]
            ],
        )
    profile = result["study"].get("factor_profile", [])
    previous = continuity(
        steps.get("target", {}).get("prior_outcomes"), selection["selected_factor_ids"]
    )
    if previous:
        section("Continuing the previous research", previous["notice"])
        if previous["entries"]:
            table(
                ["Target", "Status / reason", "Predicted", "Actual", "Error", "Feedback supplied"],
                [
                    [
                        e["target_session"],
                        e["status"] + " / " + (e["reason"] or ""),
                        _number(e["predicted_price"]),
                        _number(e["actual_price"]),
                        _number(e["price_error"]),
                        "Yes" if e["provided_as_feedback"] else "No",
                    ]
                    for e in previous["entries"]
                ],
            )
            table(
                ["Selected model", "Factors"],
                [
                    ["Previous run", ", ".join(previous["previous_factor_ids"])],
                    ["This run", ", ".join(previous["selected_factor_ids"])],
                    ["Added", ", ".join(previous["added"]) or "None"],
                    ["Removed", ", ".join(previous["removed"]) or "None"],
                ],
            )
        else:
            section("First run", "No previous outcome feedback in this history.")
    history = result["study"].get("history")
    if history:
        section("Forecast history and later observations", history["notice"])
        table(
            [
                "Issued",
                "Target",
                "Round",
                "Status",
                "Predicted",
                "Actual",
                "Price error",
                "Net return",
                "Sharpe",
                "Buy-and-hold",
            ],
            [
                [
                    entry["issued_at"],
                    (entry.get("forecast") or {}).get("target_session", "N/A"),
                    entry["selected_round"],
                    entry["status"],
                    _number((entry.get("forecast") or {}).get("predicted_price")),
                    _number((entry.get("metrics") or {}).get("actual_price")),
                    _number((entry.get("metrics") or {}).get("price_error")),
                    _number(
                        (entry.get("strategy") or {}).get("strategy", {}).get("net_return"), True
                    ),
                    _number((entry.get("strategy") or {}).get("strategy", {}).get("sharpe_ratio")),
                    _number(
                        (entry.get("strategy") or {}).get("buy_and_hold", {}).get("net_return"),
                        True,
                    ),
                ]
                for entry in history["entries"]
            ],
        )
    if profile:
        section(
            "Instrument factor profile",
            "Ownership and exact source bindings are retained. "
            "Availability is fixed on training/development rows; inclusion is not selection. "
            "Unsupported business/text sources remain pending, never target-price substitutes.",
        )
        table(
            ["Factor", "Definition", "Scope / owner", "Source / inputs", "Availability"],
            [
                [
                    item["factor_id"],
                    item["definition_id"],
                    f"{item['definition_scope']} / {item['definition_owner']}",
                    " / ".join(item["source"][key] for key in ("source_id", "entity", "field"))
                    if item.get("source")
                    else ", ".join(item["input_factor_ids"]),
                    item["availability_reason"],
                ]
                for item in profile
            ],
        )
        for item in profile:
            if item.get("recipe", {}).get("op") == "factor_expression":
                section(item["factor_id"], item["description"])
                table(
                    ["Node", "Operation", "Inputs"],
                    [
                        [node["name"], node["op"], node.get("leaf") or ", ".join(node["inputs"])]
                        for node in item["recipe"]["expression"]["nodes"]
                    ],
                )
    terminal = result.get("terminal_evaluation")
    if terminal:
        section("Reserved terminal evaluation", terminal["limitation"])
        section(
            "Frozen selection",
            f"Round {terminal['selected_round']}; "
            f"{terminal['sample_count']} reserved observations; "
            f"status: {terminal['status']}. The same fitted model is reused without refitting.",
        )
        if terminal["status"] == "completed":
            evaluation = terminal["evaluation"]
            forecasts = [("Selected model", evaluation["metrics"])] + [
                (name, value["metrics"]) for name, value in evaluation["baselines"].items()
            ]
            table(
                ["Terminal forecast", "Price MAE", "Price RMSE", "Observations"],
                [
                    [name, _number(m["mae_price"]), _number(m["rmse_price"]), m["sample_count"]]
                    for name, m in forecasts
                ],
            )
            accounts = [(name, terminal["strategy"][name]) for name in ("strategy", "buy_and_hold")]
            table(
                ["Terminal account", "Net return", "Sharpe", "Max drawdown", "Daily returns"],
                [
                    [
                        name,
                        _number(a["metrics"]["net_return"], True),
                        _number(a["metrics"]["sharpe_ratio"]),
                        _number(a["metrics"]["max_drawdown"], True),
                        a["metrics"]["risk_statistics"]["daily_return_count"],
                    ]
                    for name, a in accounts
                ],
            )
            section(
                "Terminal account period",
                str(terminal["strategy"]["period"]) + ". "
                "Same account costs and policy as development, independently initialized. "
                "Sharpe is annualized using 252 sessions/year and zero risk-free rate; "
                "these short samples are descriptive. Do not join these NAV curves.",
            )
            table(
                ["Terminal origin", "Target", "Observed", "Predicted", "Unchanged price"],
                [
                    [
                        r["origin_session"],
                        r["target_session"],
                        _number(r["observed_price"]),
                        _number(r["predicted_price"]),
                        _number(r["reference_price"]),
                    ]
                    for r in evaluation["rows"]
                ],
            )
        else:
            section("Terminal inputs unavailable", "No model substitution or row dropping.")
            table(
                ["Origin", "Missing factors"],
                [
                    [r["origin_session"], ", ".join(r["factor_ids"])]
                    for r in terminal["missing_inputs"]
                ],
            )
    table(
        ["Component", "Provider"],
        [
            ["Relationships", steps["related_stocks"]["provider"]],
            ["News", steps["news"]["provider"]],
            ["Hypotheses", steps["initial_hypothesis"]["provider"]],
            ["Prices", first["price_provider"]],
            ["Model", first["provider"]],
            ["Execution", first["strategy"]["evidence_mode"]],
        ],
    )
    model_config = first.get("model_configuration")
    if model_config is not None:
        section(
            "Model configuration",
            "Chosen before research and shared by both attempts. Preprocessing is fitted only "
            "on training rows; neither model choice nor hyperparameters are searched in this run.",
        )
        table(
            ["Setting", "Value"], [[key, json.dumps(value)] for key, value in model_config.items()]
        )
    section("News and relationships", result["study"]["news_use"])
    for path in evidence_path(steps):
        section(
            f"Round {path['round']}: news → hypothesis → fitted inputs",
            (path["hypothesis"] or "No hypothesis text") + " " + path["notice"],
        )
        table(
            ["Cited source ID", "Title", "Role"],
            [[c["id"], c["title"], "Hypothesis context"] for c in path["citations"]],
        )
        table(
            ["Factor", "Input", "Execution", "Reason"],
            [[f["factor_id"], f["input_kind"], f["status"], f["reason"]] for f in path["factors"]],
        )
    related = steps["related_stocks"]
    if related.get("relationships"):
        table(
            ["Company", "Relationship", "Source"],
            [
                [item["symbol"], item["relationship"], "; ".join(item["source_urls"])]
                for item in related["relationships"]
            ],
        )
    else:
        section("Relationship context", related["reason"])
    news = steps["news"]
    if news.get("coverage"):
        table(
            ["Issuer", "Coverage", "Selected items"],
            [[item["symbol"], item["status"], item["selected_items"]] for item in news["coverage"]],
        )
    table(
        ["Symbol / scope", "News", "Published", "Source"],
        [
            [
                item["symbol"] + (" research context" if item.get("scope") else ""),
                item.get("title", item["text"]),
                item.get("published_at") or item.get("published_date") or "unknown / simulated",
                item.get("url", "simulated"),
            ]
            for item in news["items"]
        ],
    )
    research = news.get("research")
    if research:
        section("Active news research", research["provider"] + ". " + research["notice"])
        table(
            ["Search request", "Proposed query", "Origin", "Reason", "Recorded tool actions"],
            [
                [
                    index,
                    item["query"],
                    item.get("query_origin", "not recorded"),
                    item["rationale"],
                    json.dumps(item["search_actions"]),
                ]
                for index, item in enumerate(research["searches"], 1)
            ],
        )
        table(
            ["Article ID", "Title", "Read status", "Received at", "Text extent"],
            [
                [
                    item["id"],
                    item["title"],
                    item["status"] + (": " + item["reason"] if item.get("reason") else ""),
                    item["received_at"] or "not fetched",
                    item["content_extent"] + (" (truncated)" if item["text_truncated"] else ""),
                ]
                for item in research["articles"]
            ],
        )
        for article in research["articles"]:
            if article["status"] == "available":
                url, identity = article["final_url"], article["id"]
                md.extend([f"[{identity}](<{url}>)", ""])
                html.append(
                    f"<p><a href='{escape(url, quote=True)}' rel='noreferrer'>"
                    f"{escape(identity)}: {escape(article['title'])}</a></p>"
                )
        table(
            ["Research finding (LLM interpretation)", "Article IDs"],
            [[item["text"], ", ".join(item["evidence_ids"])] for item in research["findings"]],
        )
        section(
            "Suggested related companies",
            "Source-cited LLM candidates only. Business claims and ticker mappings are not "
            "independently verified; they do not replace the profile or add peer price features.",
        )
        table(
            ["Symbol / name", "Suggested relationship", "Article IDs"],
            [
                [
                    item["symbol"] + " / " + item["name"],
                    item["relationship"],
                    ", ".join(item["evidence_ids"]),
                ]
                for item in research["related_candidates"]
            ],
        )
        table(
            ["Research round", "Chosen factors", "Cited news / article IDs"],
            [
                [
                    label,
                    ", ".join(steps[key]["factor_ids"]),
                    ", ".join(steps[key]["evidence_news_ids"]),
                ]
                for label, key in (
                    ("Initial", "initial_hypothesis"),
                    ("Revised", "revised_hypothesis"),
                )
            ],
        )
        section(
            "Article evidence file",
            "news_research.json retains bounded extracted page text, hashes, receipt times and "
            "final findings. trace.jsonl retains every search/read/review stage, including "
            "the intermediate follow-up decision. Publication time is unknown unless separately "
            "established; search snippets do not establish that an article was read.",
        )
    section("Initial hypothesis", steps["initial_hypothesis"]["hypothesis"])
    section("Revision after feedback", steps["revised_hypothesis"]["hypothesis"])
    memory = result["study"].get("factor_memory", {})
    if memory.get("entries"):
        section("Recipes from previous studies", memory["notice"])
        table(
            [
                "Factor",
                "Current availability",
                "Reuse until",
                "Selected again",
                "Previous inputs",
                "Source study",
            ],
            [
                [
                    e["factor_id"],
                    e["status"].replace("_", " "),
                    _reuse_deadline(e.get("reuse_lifetime")),
                    "Yes" if e["selected_in_this_study"] else "No",
                    e["source_input_mode"],
                    e["source_study_id"][:12],
                ]
                for e in memory["entries"]
            ],
        )
    changes = result["study"]["factor_changes"]
    table(
        ["Factor change", "IDs"],
        [
            [key, ", ".join(changes[key]) or "none"]
            for key in ("activated", "deactivated", "retained")
        ],
    )
    table(
        ["Round", "Executed factor", "Recipe"],
        [
            [label, factor["factor_id"], json.dumps(factor["recipe"], sort_keys=True)]
            for label, key in (("Initial", "initial_factors"), ("Revised", "revised_factors"))
            for factor in steps[key]
        ],
    )
    dev = second["development"]
    if first.get("input_usage"):
        section(
            "Model input usage",
            "Used means an input entered this attempt's numerical path. "
            "Available but unselected peers and context-only news are distinct from executed "
            "features. Missing peer windows do not drop samples or become zero. Related "
            "prices retain the source adapter's retrospective availability qualifications; "
            "current relationship discovery is not historical or causal evidence.",
        )
        table(
            ["Attempt", "Source", "Status", "Executed factors", "Reason"],
            [
                [
                    label,
                    item["source"],
                    item["status"],
                    ", ".join(item["factor_ids"]) or "none",
                    item["reason"],
                ]
                for label, prediction in (("Initial", first), ("Revised", second))
                for item in prediction["input_usage"]
            ],
        )
    guidance = result["study"].get("guidance")
    if guidance:
        section("Issuer guidance business factors", guidance["notice"] + " " + guidance["status"])
        extraction = guidance["latest_extraction"] or {}
        table(
            ["News ID", "Classification", "Source quote", "Extraction completed"],
            [
                [f["news_id"], f["status"], f["quote"], extraction.get("extracted_at")]
                for f in extraction.get("facts", [])
            ],
        )
        table(
            ["Factor", "Days", "State", "Reason"],
            [
                [
                    d["factor_id"],
                    d["recipe"]["window_days"],
                    "eligible" if d["runnable"] else "pending",
                    ", ".join(d["reasons"]),
                ]
                for d in guidance["definitions"]
            ],
        )
        md.extend(["[Guidance source and extraction evidence](guidance_extractions.json)", ""])
        html.append(
            "<p><a href='guidance_extractions.json'>Guidance source and extraction evidence</a></p>"
        )
    news_factors = result["study"].get("news_factors")
    if news_factors:
        support = news_factors["support"]
        section(
            "News factor execution",
            (
                "Historical news inputs are usable. "
                if support["runnable"]
                else "News proposals are pending: too few usable historical snapshots. "
            )
            + result["study"]["news_use"]
            + " "
            + news_factors["timing_note"]
            + " Both attempts and prediction baselines share the same preselected sample mask. "
            "Keyword windows use calendar days; zero means no matching retained title, "
            "not absence of all issuer news.",
        )
        table(
            ["Coverage", "Count"],
            [[key, value] for key, value in support.items() if key.endswith("rows")],
        )
        table(
            ["Proposed factor", "Keywords", "Days", "State", "Reuse until"],
            [
                [
                    item["factor_id"],
                    ", ".join(item["recipe"]["keywords"]),
                    item["recipe"]["window_days"],
                    "eligible" if item["runnable"] else "pending",
                    _reuse_deadline(item.get("reuse_lifetime")),
                ]
                for item in news_factors["definitions"]
            ],
        )
        evidence_rows = []
        for label, prediction in (("Initial", first), ("Revised", second)):
            for identity, evidence in prediction.get("news_feature_evidence", {}).items():
                evidence_rows.append(
                    [
                        label,
                        identity,
                        evidence["value"],
                        evidence["snapshot_received_at"],
                        ", ".join(evidence["evidence_news_ids"]) or "no matching retained title",
                    ]
                )
        if evidence_rows:
            table(
                [
                    "Round",
                    "Executed factor",
                    "Feature value at final origin",
                    "Snapshot receipt",
                    "Evidence",
                ],
                evidence_rows,
            )
    section(
        "Prediction comparison",
        f"{dev['first_evaluation_session']} to "
        f"{dev['last_evaluation_session']}: overlapping five-session labels, "
        "retrospective development observations. MAPE is price error percent; "
        "direction hit rate is a sample frequency, not confidence or independent accuracy. "
        "Current news and feedback influenced recipe selection. No independent holdout. "
        "Where news coverage removes dates, accounts hold prior units until the next "
        "supplied signal; missing dates do not imply a fresh daily forecast.",
    )
    rows = []
    comparisons = [
        ("Initial", first["development"]["metrics"]),
        ("Revised", second["development"]["metrics"]),
    ]
    comparisons += [(key, value["metrics"]) for key, value in dev["baselines"].items()]
    for label, metrics in comparisons:
        rows.append(
            [
                label,
                metrics["sample_count"],
                _number(metrics["mae_price"]),
                _number(metrics["rmse_price"]),
                _number(metrics["mape_percent"]) + "%",
                _number(metrics["direction_hit_rate"], True),
            ]
        )
    table(["Arm", "N", "Price MAE", "Price RMSE", "MAPE", "Direction hits"], rows)
    section(
        "Costed strategy comparison",
        "Hypothetical long/cash policy: a forecast log return "
        "above the configured threshold targets the next open; unchanged long signals "
        "keep existing units. Buy-and-hold uses the same capital, dates and costs. "
        "Inventory is marked at the final close without assumed exit fees. "
        "Five-session forecast and execution intervals differ. Returns exclude dividends. "
        "Sharpe and volatility use consecutive costed daily NAV returns, including cash days "
        "and excluding the initial pre-trade mark. Annualization assumes 252 sessions/year, "
        "a 0% risk-free rate and sample standard deviation (ddof=1). "
        "These short development samples do not establish reliable long-term Sharpe. "
        "N/A means the statistic cannot be calculated; inspect the status below.",
    )
    rows = []
    cost_rows = []
    accounts = []
    for label, prediction in (("Initial", first), ("Revised", second)):
        for name in ("strategy", "buy_and_hold"):
            if name == "buy_and_hold" and label == "Initial":
                continue
            account = prediction["strategy"][name]
            metrics = account["metrics"]
            account_label = label + " strategy" if name == "strategy" else "Buy-and-hold"
            rows.append(
                [
                    account_label,
                    _number(metrics["net_return"], True),
                    _number(metrics["sharpe_ratio"]),
                    _number(metrics["annualized_volatility"], True),
                    _number(metrics["max_drawdown"], True),
                    metrics["risk_statistics"]["daily_return_count"],
                ]
            )
            cost_rows.append(
                [
                    account_label,
                    metrics["fees"]["commission"],
                    metrics["fees"]["slippage"],
                    _number(metrics["turnover"]),
                    metrics["risk_statistics"]["status"],
                ]
            )
            accounts.append((account_label, account))
    table(
        ["Account", "Net return", "Sharpe (ann.)", "Volatility (ann.)", "Max drawdown", "Daily N"],
        rows,
    )
    table(["Account", "Commission", "Slippage", "Turnover", "Risk statistic status"], cost_rows)
    html.append(_equity_chart(accounts))
    selected = first if selection["selected_attempt"] == "initial" else second
    if iterations:
        selected = iterations["attempts"][iterations["selected_round"] - 1]["prediction"]
    section(
        "Latest forecast",
        f"This forecast reuses the selected {selection['selected_attempt']} fitted model with its "
        "disclosed "
        "training cutoff. It is not an observed outcome and has no accuracy score.",
    )
    latest = selected.get("latest_forecast")
    if latest:
        table(
            ["Origin", "Target", "Reference price", "Predicted price", "Training cutoff"],
            [
                [
                    latest["origin_session"],
                    latest["target_session"],
                    _number(latest["reference_price"]),
                    _number(latest["predicted_price"]),
                    latest["training_information_cutoff"],
                ]
            ],
        )
    elif selected.get("latest_forecast_unavailable"):
        section(
            "Forecast unavailable", "; ".join(selected["latest_forecast_unavailable"]["reasons"])
        )
    else:
        section("Forecast availability", "Synthetic run: no real-market latest forecast.")
    financial = result["study"].get("financial_comparison")
    if financial:
        section("Financial business factors", result["study"]["financial_data"]["notice"])
        table(
            ["Factor", "Available", "Missing fixed rows", "Definition"],
            [
                [d["factor_id"], d["runnable"], d["missing_required_rows"], d["description"]]
                for d in result["study"]["financial_data"]["definitions"]
            ],
        )
        data = result["study"]["financial_data"]
        section("Financial source", data.get("provider", "unknown"))
        table(
            ["Source", "Status", "Reason"],
            [
                [a["provider"], a["status"], a.get("reason", "")]
                for a in data.get("source_attempts", [])
            ],
        )
        table(
            ["Unavailable statement", "Reason"],
            [[f["release_url"], f["error"]] for f in data.get("source_failures", [])],
        )
        section("With / without financial factors", financial["notice"])
        section("Financial diagnostic cost", f"{financial['extra_fits']} additional fits.")
        if financial["status"] == "completed":
            table(
                ["Model inputs", "Price MAE", "Net return", "Sharpe"],
                [
                    [
                        label,
                        financial[key]["forecast"]["mae_price"],
                        financial[key]["strategy"]["net_return"],
                        financial[key]["strategy"]["sharpe_ratio"],
                    ]
                    for key, label in (
                        ("with_financials", "Selected + financials"),
                        ("without_financials", "Same model settings, financials removed"),
                    )
                ],
            )
        else:
            section("Financial comparison availability", financial["reason"])
    comparison = result["study"].get("qlib_comparison")
    if comparison:
        section("Actual Qlib workflow comparison", comparison["notice"])
        for segment, values in comparison["segments"].items():
            if values["status"] != "completed":
                section(f"Qlib {segment}", values["reason"])
                continue
            rows = []
            for engine in ("native", "qlib"):
                value = values[engine]
                metrics = value["strategy"]["strategy"]["metrics"]
                rows.append(
                    [
                        segment,
                        engine,
                        _number(value["metrics"]["mae_price"]),
                        _number(metrics["net_return"], percent=True),
                        _number(metrics["sharpe_ratio"]),
                        _number(metrics["max_drawdown"], percent=True),
                    ]
                )
            table(["Window", "Engine", "Price MAE", "Net return", "Sharpe", "Max drawdown"], rows)
            section(
                f"{segment.capitalize()} prediction agreement",
                f"Within declared tolerance: {values['prediction_parity']}. "
                f"Maximum absolute price difference: "
                f"{values['max_absolute_difference']['predicted_price']:.12g}. "
                "Matching the same fitted algorithm is not evidence of superiority.",
            )
        section(
            "Qlib runtime", json.dumps(comparison["versions"]) + "; " + comparison["inputs_sha256"]
        )
    section(
        "Engine and limits",
        result["study"]["external_engine"] + " "
        "This is a research run, not a broker deployment or a registered prospective study. "
        "Missing partner news is not zero news. "
        "Definitions, full quotes, predictions, orders and valuations are retained locally.",
    )
    table(["Setting", "Value"], [[key, value] for key, value in result["configuration"].items()])
    runtime = result["runtime"]
    section(
        "Run time and usage",
        f"Measured wall time: {runtime['total_elapsed_seconds']:.2f} seconds. "
        "Counts describe the current execution attempt, excluding reused operations. "
        "Earlier attempts are listed separately. Token counts are only what "
        "the provider reported; missing usage is unknown, not zero. Cached input tokens are "
        "included in input totals. No monetary cost is estimated.",
    )
    table(["Operation count", "Value"], [[key, value] for key, value in runtime["counts"].items()])
    if runtime.get("previous_attempts"):
        table(
            [
                "Earlier attempt",
                "Status",
                "Seconds",
                "Model stages started",
                "LLM requests",
                "Input tokens",
                "Output tokens",
            ],
            [
                [
                    index,
                    previous["status"],
                    _number(previous["total_elapsed_seconds"]),
                    previous["counts"].get("model_stages_started", 0),
                    previous["counts"].get("llm_requests_started", 0),
                    *[
                        previous["token_usage"].get(key)
                        if previous["token_usage"].get(key) is not None
                        else "not reported"
                        for key in ("input_tokens", "output_tokens")
                    ],
                ]
                for index, previous in enumerate(runtime["previous_attempts"], 1)
            ],
        )
    usage = runtime["token_usage"]
    table(
        ["Usage coverage", "Input tokens", "Cached input tokens", "Output tokens"],
        [
            [
                usage["status"],
                *[
                    usage[key] if usage[key] is not None else "not reported"
                    for key in ("input_tokens", "cached_input_tokens", "output_tokens")
                ],
            ]
        ],
    )
    table(
        ["Stage", "Status", "Seconds"],
        [
            [row["stage"], row["status"], _number(row["duration_seconds"])]
            for row in runtime["stages"]
        ],
    )
    md.extend(
        [
            "## Full evidence",
            "",
            "See `result.json`, `inputs.json`, `news_history.json`, `config.json` and "
            "`trace.jsonl` in this directory. Reading these files makes no API calls.",
            "",
        ]
    )
    html.append(
        "<p><a href='result.json'>Full result</a> · <a href='trace.jsonl'>Stage trace</a> · "
        "<a href='news_history.json'>News snapshots</a> · "
        "<a href='inputs.json'>Retained inputs</a> · <a href='config.json'>Configuration</a> · "
        "<a href='report.md'>Markdown report</a></p>"
    )
    if research:
        md.extend(["[Active news evidence](news_research.json)", ""])
        html.append("<p><a href='news_research.json'>Active news evidence</a></p>")
    md.extend(["[Run time and usage](runtime.json)", ""])
    html.append("<p><a href='runtime.json'>Run time and usage</a></p>")
    return "\n".join(md), (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>"
        "body{margin:0;background:#f3f5f7;color:#172735;font:16px/1.6 system-ui,sans-serif}"
        "main{max-width:1100px;margin:auto;padding:32px}header{border-bottom:3px solid #147d79}"
        "h1{font-size:32px;line-height:1.2}h2{font-size:21px;margin-top:32px}"
        ".eyebrow{color:#147d79;letter-spacing:.1em;font-size:12px;font-weight:bold}"
        ".table{overflow:auto;margin:18px 0}"
        "table{width:100%;border-collapse:collapse;background:white}"
        "td,th{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #d8e1e7;"
        "overflow-wrap:anywhere;min-width:80px}th{background:#e4eeed}a{color:#086b79}"
        "svg{width:100%;height:auto;background:white}section p{max-width:100ch}"
        "</style><main>" + "".join(html) + "</main></html>"
    )


def _equity_chart(accounts):
    series = [
        (name, [(row["session_id"], float(row["nav"])) for row in account["valuations"]])
        for name, account in accounts
    ]
    values = [value for _, rows in series for _, value in rows]
    low, high = min(values), max(values)
    span = high - low or 1
    palette = ("#246bce", "#13827b", "#d18b18")
    svg = [
        "<h2>Hypothetical account NAV</h2><svg viewBox='0 0 900 340' role='img' "
        "aria-label='Initial and revised strategy NAV versus buy and hold'>"
    ]
    for name, rows in series:
        color = palette[len(svg) - 1]
        points = " ".join(
            f"{70 + i * 790 / max(1, len(rows) - 1):.2f},{260 - (v - low) * 210 / span:.2f}"
            for i, (_, v) in enumerate(rows)
        )
        svg.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2'/>"
            f"<text x='{70 + (len(svg) - 1) * 270}' y='325' fill='{color}' font-size='13'>"
            f"{escape(name)}</text>"
        )
    svg.append(
        f"<text x='8' y='55' font-size='12'>{high:.0f}</text>"
        f"<text x='8' y='265' font-size='12'>{low:.0f}</text>"
        f"<text x='70' y='290' font-size='12'>{escape(series[0][1][0][0])}</text>"
        f"<text x='775' y='290' font-size='12'>{escape(series[0][1][-1][0])}</text></svg>"
    )
    return "".join(svg)
