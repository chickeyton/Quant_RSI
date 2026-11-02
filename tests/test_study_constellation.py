"""The factor star map reads the retained ledger and shows only recorded relationships."""

import json
import re
import threading
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest
from test_study_factor_ledger import run

from quant_rsi.cli import main
from quant_rsi.research import study_history
from quant_rsi.research.study_constellation import constellation, render
from quant_rsi.research.study_workbench import Workbench, make_server


def embedded(html):
    match = re.search(r"const DATA = (.*);\nconst PALETTE", html)
    return json.loads(match.group(1))


def attempt(number, ids, mae):
    return {
        "round": number,
        "prediction": {
            "factor_ids": ids,
            "development": {
                "metrics": {"mae_price": mae},
                "baselines": {"unchanged_price": {"metrics": {"mae_price": 1.5}}},
            },
        },
    }


def register(directory, study_id, issued_at, attempts, selected, definitions):
    result = {
        "configuration": {"symbol": "KO"},
        "iterations": {"selected_round": selected, "attempts": attempts},
        "trace": [{"stage": "revised_factor_library", "output": {"definitions": definitions}}],
    }
    with study_history._ledger(directory) as (journal, store):
        study_history._append(
            journal,
            store,
            "study_registered",
            {
                "study_id": study_id,
                "symbol": "KO",
                "result_ref": store.put_json(result),
                "issued_at": issued_at,
                "status": "unavailable",
            },
        )


def test_relations_come_from_definitions_selection_and_revisions(tmp_path):
    business = {
        "factor_id": "business.inventory_gap",
        "kind": "numeric",
        "description": "Inventory minus revenue growth </script><b>not markup</b>",
        "recipe": {"op": "financial_expression"},
        "input_factor_ids": ["financial.inventory_yoy", "financial.revenue_yoy"],
    }
    momentum = {"factor_id": "research.momentum_5", "kind": "numeric", "recipe": {}}
    register(
        tmp_path,
        "s1",
        "2026-01-01T21:00:00Z",
        [
            attempt(1, ["research.momentum_5"], 2.0),
            attempt(2, ["business.inventory_gap", "peer.KOF.peer_return_5"], 1.0),
        ],
        2,
        [business, momentum],
    )
    data = constellation(tmp_path, "KO")
    stars = {s["factor_id"]: s for s in data["stars"]}
    assert stars["financial.inventory_yoy"]["input_only"]
    assert stars["financial.inventory_yoy"]["category"] == "financial"
    assert stars["peer.KOF.peer_return_5"]["category"] == "peer"
    assert stars["business.inventory_gap"]["importance"] == 2  # selected and beat baseline
    assert stars["research.momentum_5"]["importance"] == 0
    kinds = {(r["source"], r["target"], r["kind"]) for r in data["relations"]}
    assert ("financial.inventory_yoy", "business.inventory_gap", "derived") in kinds
    assert ("business.inventory_gap", "peer.KOF.peer_return_5", "co_selected") in kinds
    assert ("research.momentum_5", "business.inventory_gap", "replaced") in kinds
    assert not any(
        r["kind"] == "co_selected" and "momentum" in r["source"] for r in data["relations"]
    )

    html = render(data)
    assert "<b>not markup</b>" not in html
    assert embedded(html)["stars"] == json.loads(json.dumps(data["stars"]))


def test_cli_writes_the_map_for_default_history(tmp_path, default_history, capsys):
    assert run(tmp_path / "study") == 0
    history = default_history / "synthetic"
    assert (
        main(["research-history", "constellation", "KO", "--history-dir", str(history), "--no-ui"])
        == 0
    )
    page = (history / "constellation/KO.html").read_text()
    data = embedded(page)
    assert data["symbol"] == "KO" and len(data["studies"]) == 1 and data["stars"]
    assert "factor stars" in capsys.readouterr().out
    output = tmp_path / "map.html"
    arguments = ["--history-dir", str(history), "--output", str(output), "--no-ui"]
    assert main(["research-history", "constellation", "KO", *arguments]) == 0
    assert output.is_file()
    assert main(["research-history", "status", "KO", *arguments]) != 0
    assert "only used by" in capsys.readouterr().err


def test_workbench_serves_the_map_from_its_shared_history(tmp_path):
    register(
        tmp_path / "history/daily",
        "s1",
        "2026-01-01T21:00:00Z",
        [attempt(1, ["research.momentum_5"], 1.0)],
        1,
        [{"factor_id": "research.momentum_5", "kind": "numeric", "recipe": {}}],
    )
    with make_server(Workbench(tmp_path), "test-token") as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}/test-token/"
        try:
            with urlopen(base + "constellation/daily/KO") as response:
                data = embedded(response.read().decode())
            assert [s["factor_id"] for s in data["stars"]] == ["research.momentum_5"]
            with urlopen(base + "constellation/synthetic/KO") as response:
                assert embedded(response.read().decode())["stars"] == []
            with pytest.raises(HTTPError):
                urlopen(base + "constellation/daily/..%2Fescape")
        finally:
            server.shutdown()
