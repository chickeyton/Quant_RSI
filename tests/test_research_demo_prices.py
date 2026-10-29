"""Offline daily-source fixtures with actual feature calculation and LightGBM fits."""

import json
from pathlib import Path

import pytest
from test_daily_data import chart, install

from quant_rsi.cli import main
from quant_rsi.core.artifacts import digest
from quant_rsi.research import demo, demo_prices, llm
from quant_rsi.research.demo_model import LightGBMDemo

CATALOG = Path(__file__).resolve().parents[1] / "instruments/shared/equities.json"


@pytest.fixture
def collection(tmp_path, monkeypatch):
    raw = chart()
    source = raw["chart"]["result"][0]
    source["meta"].update(symbol="KO", instrumentType="EQUITY", exchangeName="NYQ")
    source["events"] = {}
    install(monkeypatch, data=raw)
    root = tmp_path / "prices"
    data, info = demo_prices.load_daily_prices("KO", root)
    return root, data, info


def test_retained_collection_is_reused_and_bound_to_symbol_and_bytes(collection, monkeypatch):
    root, data, info = collection

    def unexpected(*args):
        pytest.fail("A nonempty retained collection must not be downloaded again")

    monkeypatch.setattr(demo_prices, "collect_daily", unexpected)
    loaded, metadata = demo_prices.load_daily_prices("KO", root)
    assert loaded.dates == data.dates and metadata == info
    assert len(loaded.dates) == 120 and metadata["price_basis"] == "raw"
    with pytest.raises(ValueError, match="match the requested symbol"):
        demo_prices.load_daily_prices("SOXL", root)
    (root / "events.csv").write_bytes((root / "events.csv").read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input hashes differ"):
        demo_prices.load_daily_prices("KO", root)


def test_failed_receipt_is_preserved_without_overwrite(tmp_path, monkeypatch):
    (tmp_path / "collection_receipt.json").write_text('{"status":"failed"}')
    before = (tmp_path / "collection_receipt.json").read_bytes()

    def unexpected(*args):
        pytest.fail("Failed evidence must be preserved")

    monkeypatch.setattr(demo_prices, "collect_daily", unexpected)
    with pytest.raises(ValueError, match="successful"):
        demo_prices.load_daily_prices("KO", tmp_path)
    assert (tmp_path / "collection_receipt.json").read_bytes() == before


def test_cli_real_price_path_separates_development_feedback_from_latest_forecast(
    collection,
    monkeypatch,
    capsys,
):
    root, data, info = collection
    calls = []

    def codex(self, prompt, schema):
        value = json.loads(prompt.split("\nINPUT_JSON:\n")[1])
        calls.append(value)
        assert value["context"]["provider"] == "public_daily_prices"
        assert "not historically available" in value["context"]["model_note"]
        definitions = value["catalog"]["definitions"]
        return json.dumps(
            {
                "hypothesis": "Explore a numeric recipe and revise after development feedback.",
                "factor_ids": [definitions[1 if value["feedback"] else 0]["factor_id"]],
                "evidence_news_ids": [value["context"]["news"]["items"][0]["id"]],
                "event_signal": 0,
            }
        )

    monkeypatch.setattr(llm.LLMHypothesizer, "_codex", codex)
    assert (
        main(
            [
                "research-demo",
                "ko",
                "--catalog",
                str(CATALOG),
                "--model",
                "lightgbm",
                "--prices",
                "daily",
                "--prices-dir",
                str(root),
                "--llm",
                "codex",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    stages = {event["stage"]: event["output"] for event in result["trace"]}
    first, second = stages["first_prediction"], stages["second_prediction"]
    assert first["origin_session"] == second["origin_session"] == data.dates[-6]
    assert first["training_rows"] == second["training_rows"] == 105
    assert first["last_training_label_session"] == data.dates[-6]
    assert stages["feedback"]["observed_price"] == data.closes[-1, 0]
    assert calls[1]["feedback"] == stages["feedback"]
    latest = result["summary"]["latest_forecast"]
    assert latest == second["latest_forecast"]
    assert latest["origin_session"] == data.dates[-1]
    assert latest["target_session"] == info["latest_target_session"] > data.dates[-1]
    assert latest["outcome"] == "not_observed"
    assert latest["training_information_cutoff"] == data.dates[-6]
    assert "Real daily prices" in result["notice"] and result["mode"] == "mixed"


def test_historical_estimate_ignores_later_prices_but_observation_and_latest_use_them(collection):
    _, data, info = collection
    model = LightGBMDemo(demo.load_catalog(CATALOG), prices=(data, info))
    factors = model.library["definitions"]
    before = model.predict({}, factors, {})
    data.closes[model.origin + 1 :] *= 1.1
    after = model.predict({}, factors, {})
    assert before["predicted_price"] == after["predicted_price"]
    assert before["feature_values"] == after["feature_values"]
    assert (
        before["latest_forecast"]["reference_price"] != after["latest_forecast"]["reference_price"]
    )


def test_missing_session_rejected_even_if_receipt_hash_is_updated(collection):
    root, _, _ = collection
    path = root / "events.csv"
    lines = path.read_bytes().splitlines(keepends=True)
    raw = b"".join(lines[:2] + lines[3:])
    path.write_bytes(raw)
    receipt = json.loads((root / "collection_receipt.json").read_bytes())
    receipt["normalized_inputs"]["events_file_sha256"] = digest(raw)
    (root / "collection_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="incomplete or unexpected price sessions"):
        demo_prices.load_daily_prices("KO", root)


def test_known_split_stops_before_fitting(tmp_path, monkeypatch):
    install(monkeypatch)  # Existing SOXL fixture includes a split inside the price window.
    with pytest.raises(ValueError, match="Unresolved corporate action"):
        demo_prices.load_daily_prices("SOXL", tmp_path)


def test_daily_mode_requires_explicit_model_and_storage(capsys):
    assert main(["research-demo", "--prices", "daily"]) == 1
    assert "requires --model lightgbm and --prices-dir" in capsys.readouterr().err
