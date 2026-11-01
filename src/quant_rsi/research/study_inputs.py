"""Human-readable input coverage before any paid research or model fitting."""

from html import escape

from quant_rsi.core.artifacts import atomic_write, canonical


def input_report(root, symbol, model, news, financials):
    financials = financials or {
        "provider": "disabled",
        "definitions": [],
        "notice": "Not requested",
    }
    report = {
        "kind": "input_check",
        "symbol": symbol,
        "prices": {
            "provider": model.price_provider,
            "first_session": str(model.data.dates[0]),
            "last_session": str(model.data.dates[-1]),
            "training_rows": len(model.train),
            "development_rows": len(model.origins),
        },
        "news": {
            "provider": news.get("provider"),
            "items": len(news.get("items", [])),
            "coverage": news.get("coverage", []),
        },
        "financials": {k: v for k, v in financials.items() if k not in {"columns", "evidence"}},
        "notice": "Input coverage only: no LLM calls or model fits at this point. "
        "Peer prices, active news searches, article reading and LLM availability are not checked. "
        "Current news does not establish historical news coverage. "
        "Unavailable financial factors are excluded; "
        "available price factors can still be researched.",
    }
    atomic_write(root / "input_report.json", canonical(report))

    def table(headers, rows):
        def cells(row, tag):
            return "".join(f"<{tag}>{escape(str(x))}</{tag}>" for x in row)

        if not rows:
            return "<p>None recorded.</p>"
        return (
            "<table><thead><tr>"
            + cells(headers, "th")
            + "</tr></thead><tbody>"
            + "".join("<tr>" + cells(row, "td") + "</tr>" for row in rows)
            + "</tbody></table>"
        )

    prices = report["prices"]
    periods = financials.get("coverage", {}).get("reported_periods", [])
    body = table(
        ["Input", "Source", "Coverage"],
        [
            [
                "Target prices",
                prices["provider"],
                f"{prices['first_session']} → {prices['last_session']}",
            ],
            [
                "Current news",
                report["news"]["provider"],
                f"{report['news']['items']} retained items",
            ],
            [
                "Financial statements",
                financials["provider"],
                f"{len(periods)} retained periods · {periods[0]} → {periods[-1]}"
                if periods
                else "No retained quarterly values",
            ],
        ],
    )
    body += "<h2>Executable financial factors</h2>" + table(
        ["Factor", "Available", "Missing required rows", "Definition"],
        [
            [
                d["factor_id"],
                "Available" if d["runnable"] else "Unavailable",
                d["missing_required_rows"],
                d["description"],
            ]
            for d in financials["definitions"]
        ],
    )
    body += "<p>" + escape(financials["notice"]) + "</p>"
    body += "<h2>Data source attempts</h2>" + table(
        ["Provider", "Status", "Reason"],
        [
            [a["provider"], a["status"], a.get("reason", "")]
            for a in financials.get("source_attempts", [])
        ],
    )
    body += "<h2>Unavailable statements</h2>" + table(
        ["Release", "Reason"],
        [[f["release_url"], f["error"]] for f in financials.get("source_failures", [])],
    )
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Research input check</title><style>
body{background:#0b1220;color:#e5edf9;font:16px system-ui;
max-width:1100px;margin:40px auto;padding:20px}
h1,h2{color:#72e3d1}p{line-height:1.6}table{width:100%;border-collapse:collapse;display:block;overflow:auto}
th,td{text-align:left;padding:12px;border-bottom:1px solid #334155;overflow-wrap:anywhere}
th:first-child,td:first-child{min-width:150px}th:nth-child(2),td:nth-child(2){min-width:180px}
th{color:#a7bbd8}h2{margin-top:32px;font-size:20px}</style><main>"""
    document += f"<h1>{escape(symbol)} · Input check</h1><p>{escape(report['notice'])}</p>" + body
    atomic_write(root / "input_report.html", (document + "</main></html>").encode())
    return report
