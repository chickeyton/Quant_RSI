"""Portable research reports: local HTML/Markdown plus the complete JSON evidence."""

import json
from html import escape


def _cell(value):
    return str(value).replace("\n", " ").replace("|", "\\|")


def _number(value, percent=False):
    if value is None:
        return "N/A"
    return f"{float(value) * (100 if percent else 1):.4f}" + ("%" if percent else "")


def render_study(result):
    steps = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = (steps[key] for key in ("first_prediction", "second_prediction"))
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
    section(
        "What ran",
        "Two model fits, one feedback-driven revision, and a fixed development "
        "window. Results include failed or missing source coverage. No orders were sent.",
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
    section("News and relationships", result["study"]["news_use"])
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
        ["Issuer", "News", "Published", "Source"],
        [
            [
                item["symbol"],
                item.get("title", item["text"]),
                item.get("published_at") or item.get("published_date") or "unknown / simulated",
                item.get("url", "simulated"),
            ]
            for item in news["items"]
        ],
    )
    section("Initial hypothesis", steps["initial_hypothesis"]["hypothesis"])
    section("Revision after feedback", steps["revised_hypothesis"]["hypothesis"])
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
            ["Proposed factor", "Keywords", "Days", "State"],
            [
                [
                    item["factor_id"],
                    ", ".join(item["recipe"]["keywords"]),
                    item["recipe"]["window_days"],
                    "eligible" if item["runnable"] else "pending",
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
                    "Count at final origin",
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
    section(
        "Latest forecast",
        "This forecast reuses the revised fitted model with its disclosed "
        "training cutoff. It is not an observed outcome and has no accuracy score.",
    )
    latest = second.get("latest_forecast")
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
    elif second.get("latest_forecast_unavailable"):
        section("Forecast unavailable", "; ".join(second["latest_forecast_unavailable"]["reasons"]))
    else:
        section("Forecast availability", "Synthetic run: no real-market latest forecast.")
    section(
        "Engine and limits",
        result["study"]["external_engine"] + " "
        "This is a research run, not a broker deployment or a registered prospective study. "
        "Missing partner news is not zero news. "
        "Definitions, full quotes, predictions, orders and valuations are retained locally.",
    )
    table(["Setting", "Value"], [[key, value] for key, value in result["configuration"].items()])
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
