"""Every attempted factor round survives studies, including rejected revisions."""

import json
from copy import deepcopy

from test_research_study import command

from quant_rsi.cli import main
from quant_rsi.research import study, study_history
from quant_rsi.research.demo import simulated_hypothesis
from quant_rsi.research.study_factor_ledger import attempt_records, summarize


def run(root, *extra):
    return main(command(root, "--llm", "simulated", "--model", "ridge", "--quiet", *extra))


def test_default_history_records_all_rounds_and_feeds_the_next_hypothesis(
    tmp_path, monkeypatch, default_history, capsys
):
    contexts = []

    def hypothesis(context, library, **kwargs):
        contexts.append(deepcopy(context))
        return simulated_hypothesis(context, library, **kwargs)

    monkeypatch.setattr(study, "simulated_hypothesis", hypothesis)
    assert run(tmp_path / "first") == 0
    assert contexts[0]["factor_ledger"]["factors"] == []
    first = json.loads((tmp_path / "first/result.json").read_text())
    tried = {key for a in first["iterations"]["attempts"] for key in a["prediction"]["factor_ids"]}
    selected = set(first["study"]["selection"]["selected_factor_ids"])
    assert tried - selected, "the fixture needs one factor that was only tried"

    history = default_history / "synthetic"
    assert run(tmp_path / "second") == 0
    ledger = {item["factor_id"]: item for item in contexts[-1]["factor_ledger"]["factors"]}
    assert tried <= set(ledger)
    for key in tried - selected:
        assert ledger[key]["rounds_selected"] == 0 and ledger[key]["rounds_tried"] >= 1
        assert ledger[key]["recipe"] is not None
    assert all(ledger[key]["rounds_selected"] == 1 for key in selected)
    assert "factor_ledger" not in contexts[-1]["prior_outcomes"]

    # The ledger is read through retained artifacts, not the mutable run directories.
    (tmp_path / "first/result.json").unlink()
    view = study_history.factor_ledger(history, "KO")
    assert view["study_count"] == 2
    assert main(["research-history", "factors", "KO", "--history-dir", str(history)]) == 0
    assert "remembered factors from 2 studies" in capsys.readouterr().out


def test_no_history_keeps_a_run_self_contained(tmp_path, default_history, capsys):
    assert run(tmp_path / "isolated", "--no-history") == 0
    assert not default_history.exists()
    assert run(tmp_path / "both", "--no-history", "--history-dir", str(tmp_path / "h")) != 0
    assert "cannot be combined" in capsys.readouterr().err
    assert not (tmp_path / "both").exists()


def test_entries_registered_before_the_ledger_recover_attempts_from_their_result(tmp_path):
    result = {
        "configuration": {"symbol": "KO"},
        "iterations": {
            "selected_round": 2,
            "attempts": [
                {
                    "round": 1,
                    "prediction": {
                        "factor_ids": ["research.momentum_5"],
                        "development": {
                            "metrics": {"mae_price": 2.0},
                            "baselines": {"unchanged_price": {"metrics": {"mae_price": 1.5}}},
                        },
                    },
                },
                {
                    "round": 2,
                    "prediction": {
                        "factor_ids": ["research.momentum_11"],
                        "development": {
                            "metrics": {"mae_price": 1.0},
                            "baselines": {"unchanged_price": {"metrics": {"mae_price": 1.5}}},
                        },
                    },
                },
            ],
        },
        "trace": [
            {
                "stage": "revised_factor_library",
                "output": {
                    "definitions": [
                        {"factor_id": f"research.momentum_{w}", "kind": "numeric", "recipe": {}}
                        for w in (5, 11)
                    ]
                },
            }
        ],
    }
    with study_history._ledger(tmp_path) as (journal, store):
        study_history._append(
            journal,
            store,
            "study_registered",
            {
                "study_id": "legacy",
                "symbol": "KO",
                "result_ref": store.put_json(result),
                "issued_at": "2026-01-01",
                "status": "unavailable",
            },
        )
    factors = {f["factor_id"]: f for f in study_history.factor_ledger(tmp_path, "KO")["factors"]}
    assert factors["research.momentum_5"]["rounds_selected"] == 0
    assert factors["research.momentum_5"]["rounds_beating_unchanged_price"] == 0
    assert factors["research.momentum_11"]["rounds_selected"] == 1
    assert factors["research.momentum_11"]["rounds_beating_unchanged_price"] == 1
    assert factors["research.momentum_11"]["kind"] == "numeric"


def test_summary_keeps_the_latest_definition_and_counts_each_study_once():
    def study_record(study_id, issued_at, maes, description):
        # The factor appears in every round; the final round is the selected one.
        result = {
            "iterations": {
                "selected_round": len(maes),
                "attempts": [
                    {
                        "round": number,
                        "prediction": {
                            "factor_ids": ["a"],
                            "development": {"metrics": {"mae_price": mae}},
                        },
                    }
                    for number, mae in enumerate(maes, start=1)
                ],
            }
        }
        return {
            "study_id": study_id,
            "issued_at": issued_at,
            "question": "",
            "outcome_status": "settled",
            "attempts": attempt_records(result),
            "definitions": [{"factor_id": "a", "description": description}],
        }

    (item,) = summarize(
        [
            study_record("s1", "2026-01-01", [2.0], "old"),
            study_record("s2", "2026-02-01", [4.0, 3.0], "new"),
        ]
    )["factors"]
    assert item["study_count"] == 2 and item["settled_selected_forecasts"] == 2
    assert item["rounds_tried"] == 3 and item["rounds_selected"] == 2
    assert item["best_mae_price"] == 2.0 and item["last_mae_price"] == 3.0
    assert item["description"] == "new" and item["first_seen"] == "2026-01-01"
