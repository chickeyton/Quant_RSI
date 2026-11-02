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
from quant_rsi.research.study_estimators import column_share
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


def test_column_share_is_the_within_round_weight():
    assert column_share([1, 3], ["a", "b"]) == {"a": 0.25, "b": 0.75}
    assert column_share([0, 0], ["a", "b"]) == {}
    assert column_share([1], ["a", "b"]) == {}


def test_round_stack_keeps_each_rounds_column_share(tmp_path):
    register(
        tmp_path,
        "s1",
        "2026-01-01T21:00:00Z",
        [
            {
                "round": 1,
                "accepted": True,
                "hypothesis": {"hypothesis": "Start from a slow price recipe."},
                "prediction": {
                    "factor_ids": ["research.momentum_5"],
                    "column_share": {"research.momentum_5": 1, "ignored": 9},
                    "development": {
                        "metrics": {"mae_price": 1.2},
                        "baselines": {"unchanged_price": {"metrics": {"mae_price": 1.5}}},
                    },
                },
            },
            {
                "round": 2,
                "accepted": False,
                "hypothesis": {"hypothesis": "Add the peer and keep the price recipe."},
                "prediction": {
                    "factor_ids": ["research.momentum_5", "peer.KOF.peer_return_5"],
                    "column_share": {"research.momentum_5": 0.25, "peer.KOF.peer_return_5": 0.75},
                    "development": {
                        "metrics": {"mae_price": 1.4},
                        "baselines": {"unchanged_price": {"metrics": {"mae_price": 1.5}}},
                    },
                },
            },
        ],
        1,
        [
            {"factor_id": "research.momentum_5", "kind": "numeric", "recipe": {}},
            {"factor_id": "peer.KOF.peer_return_5", "kind": "numeric", "recipe": {}},
        ],
    )
    stack = constellation(tmp_path, "KO")["studies"][0]["round_stack"]
    assert stack[0]["column_share"] == {"research.momentum_5": 1}
    assert stack[0]["hypothesis"] == "Start from a slow price recipe."
    assert stack[0]["accepted"] is True and stack[0]["selected"] is True
    assert stack[1]["column_share"]["peer.KOF.peer_return_5"] == 0.75
    assert stack[1]["accepted"] is False and stack[1]["selected"] is False
    assert "Round stack" in render(constellation(tmp_path, "KO"))


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
    names = {s["ledger_id"]: s["factor_id"] for s in data["stars"]}
    kinds = {(names[r["source"]], names[r["target"]], r["kind"]) for r in data["relations"]}
    assert ("financial.inventory_yoy", "business.inventory_gap", "derived") in kinds
    assert any(
        kind == "co_selected"
        and {source, target} == {"business.inventory_gap", "peer.KOF.peer_return_5"}
        for source, target, kind in kinds
    )
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


def test_same_name_versions_keep_separate_stars_rounds_and_edges(tmp_path):
    for index, keyword in enumerate(("earnings", "litigation"), 1):
        register(
            tmp_path,
            f"s{index}",
            f"2026-01-0{index}T21:00:00Z",
            [attempt(1, ["news.event_7", "research.momentum_5"], float(index))],
            1,
            [
                {"factor_id": "news.event_7", "recipe": {"keywords": [keyword]}},
                {"factor_id": "research.momentum_5", "recipe": {"window": 5}},
            ],
        )
    data = constellation(tmp_path, "KO")
    news = [star for star in data["stars"] if star["factor_id"] == "news.event_7"]
    assert len(news) == 2
    assert {star["best_mae_price"] for star in news} == {1.0, 2.0}
    assert all(star["rounds_tried"] == 1 for star in news)
    nodes = {star["ledger_id"] for star in news}
    assert {s["round_stack"][0]["node_ids"][0] for s in data["studies"]} == nodes
    assert len(data["relations"]) == 2
    assert all(r["first_seen"] == r["last_seen"] for r in data["relations"])


def test_constellation_rejects_symbol_path_escape(tmp_path, capsys):
    assert (
        main(
            [
                "research-history",
                "constellation",
                "../../outside",
                "--history-dir",
                str(tmp_path / "history"),
                "--no-ui",
            ]
        )
        != 0
    )
    assert not list(tmp_path.rglob("*.html"))
