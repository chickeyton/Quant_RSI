"""A configurable input window reaches actual fits without rewriting retained evidence."""

import json

import pytest
from test_study_horizon import command, long_collection

from quant_rsi.cli import main
from quant_rsi.research.study_queue import study_options
from quant_rsi.research.study_walk_forward import price_history_sessions


@pytest.mark.parametrize("requested", [True, 119, 446, 252.0, "252"])
def test_history_options_reject_invalid_types_and_bounds(requested):
    with pytest.raises(ValueError, match="Historical sessions"):
        study_options({"historical_sessions": requested})


def test_history_requirement_accounts_for_horizon_and_diagnostic():
    assert price_history_sessions(5, 20, 0) == 120
    assert price_history_sessions(60, 20, 3) == 445
    assert price_history_sessions(5, 20, 3, 252) == 252
    with pytest.raises(ValueError, match="445 through 445"):
        study_options({"horizon_sessions": 60, "walk_forward_folds": 3, "historical_sessions": 252})
    with pytest.raises(ValueError, match="requires daily prices"):
        study_options({"data": "synthetic", "historical_sessions": 252})


def test_requested_history_drives_preflight_real_fits_and_preserves_inputs(tmp_path, monkeypatch):
    prices = tmp_path / "prices"
    long_collection(prices, monkeypatch, "2026-06-30", 5, historical_sessions=252)
    receipt = (prices / "collection_receipt.json").read_bytes()
    root = tmp_path / "study"
    args = command(root, 5)
    args[args.index("--prices") + 1] = "daily"
    args += ["--prices-dir", str(prices), "--historical-sessions", "252", "--no-ui"]
    assert main(args) == 0
    result = json.loads((root / "result.json").read_bytes())
    assert result["configuration"]["historical_sessions"] == 252
    report = json.loads((root / "input_report.json").read_bytes())
    assert report["prices"]["historical_sessions"] == 252
    assert report["prices"]["training_rows"] == 177
    assert report["prices"]["development_rows"] == 20
    for event in result["trace"]:
        if event["stage"] in {"first_prediction", "second_prediction"}:
            assert event["output"]["training_rows"] == 177
    assert result["iterations"]["completed_fits"] == 2
    assert (prices / "collection_receipt.json").read_bytes() == receipt
    # The normal CLI also requests this length when it owns a fresh acquisition.
    args[args.index("--output") + 1] = str(tmp_path / "fresh-check")
    args[args.index("--prices-dir") + 1] = str(tmp_path / "fresh-prices")
    assert main(args + ["--check-inputs"]) == 0
    fresh = json.loads((tmp_path / "fresh-prices/collection_receipt.json").read_bytes())
    assert fresh["historical_sessions"] == 252
    assert not (tmp_path / "fresh-check/result.json").exists()
    args[args.index("--prices-dir") + 1] = str(prices)
    # The request is a minimum: a retained longer collection remains intact and is disclosed.
    args[args.index("--output") + 1] = str(tmp_path / "shorter-request")
    args[args.index("--historical-sessions") + 1] = "120"
    assert main(args + ["--check-inputs"]) == 0
    report = json.loads((tmp_path / "shorter-request/input_report.json").read_bytes())
    assert report["prices"]["historical_sessions"] == 252
    args[args.index("--output") + 1] = str(tmp_path / "too-long")
    args[args.index("--historical-sessions") + 1] = "445"
    assert main(args + ["--check-inputs"]) == 1
    assert (prices / "collection_receipt.json").read_bytes() == receipt


def test_cli_rejects_explicit_history_for_synthetic_before_creating_study(tmp_path):
    root = tmp_path / "invalid"
    assert main(command(root, 5) + ["--historical-sessions", "252"]) == 1
    assert not root.exists()
