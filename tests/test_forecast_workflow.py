import json

import pytest
from market_helpers import contract_and_events, write_events

from quant_rsi import cli
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.forecast.forecast_workflow import run_forecast
from quant_rsi.market import daily_data, market_contracts
from quant_rsi.research import research


def test_one_command_baseline_collects_once_and_replays_without_fit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    collections = []

    def collect(ticker, output):
        collections.append(ticker)
        output.mkdir(parents=True)
        contract, events, availability = contract_and_events(count=75, instruments=(ticker,))
        events = [
            event
            for event in events
            if event["session_id"] is None or int(event["session_id"][1:]) < 70
        ]
        raw = output / "raw.json"
        raw.write_bytes(b'{"engineering_fixture":true}')
        receipt = output / "receipt.json"
        receipt.write_bytes(canonical(dict(received_at=market_contracts.trusted_now())))
        units = output / "units.txt"
        units.write_bytes(b"Engineering units; no real provider assertion")
        calendar = output / "calendar.txt"
        calendar.write_bytes(b"Engineering daily calendar")
        available = output / "availability.txt"
        available.write_bytes(availability)
        return dict(
            contract=contract.model_dump(mode="json"),
            events_path=str(write_events(output / "events.csv", contract, events)),
            raw_path=str(raw),
            receipt_path=str(receipt),
            origin_session_id="s69",
            evidence_paths={
                digest(path.read_bytes()): str(path) for path in (units, calendar, available)
            },
            price_unit_evidence_ref=digest(units.read_bytes()),
            calendar_evidence_ref=digest(calendar.read_bytes()),
            cash_actions={},
            qualifications=["engineering_fixture"],
        )

    monkeypatch.setattr(daily_data, "collect_daily", collect)
    root = tmp_path / "baseline"
    argv = ["forecast", "run", "A", "--request-id", "one-command", "--output", str(root)]
    assert cli.main([*argv, "--prepare-only"]) == 0
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["status"] == "prepared" and prepared["actual_fits"] == 0
    assert not (root / "research").exists()
    original_write = research.atomic_write
    with monkeypatch.context() as interrupted:

        def after_manifest(path, body):
            original_write(path, body)
            if path.name == "manifest.json":
                raise OSError("injected initialization interruption")

        interrupted.setattr(research, "atomic_write", after_manifest)
        assert cli.main(argv) == 1
        assert "injected initialization" in capsys.readouterr().err
    assert (root / "research" / "manifest.json").exists()
    assert (root / "research" / "research_initialization.json").exists()
    assert cli.main(argv) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["rows"][0]["forecast_price"] > 0
    assert first["target_session"] == "s74"
    assert research.inspect_research(root / "research")["kernel_attempts"] == 1
    frozen = json.loads((root / "protocol.json").read_text())["protocol"]
    assert frozen["evaluation_origin_indices"] == [35, 40, 45, 50, 55, 60]
    assert frozen["actual_fits_reserved"] == 1

    def forbidden(*args, **kwargs):
        pytest.fail("duplicate forecast request attempted another fit")

    monkeypatch.setattr(research, "resume_research", forbidden)
    assert cli.main(argv) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert collections == ["A"]
    assert cli.main(["forecast", "replay", "one-command"]) == 0
    assert json.loads(capsys.readouterr().out) == first
    with pytest.raises(ValueError, match="another ticker/output"):
        run_forecast("A", tmp_path / "different-output", request_id="one-command")


def test_completed_collection_recovers_without_new_download_or_fit(tmp_path, monkeypatch):
    from test_daily_data import install

    from quant_rsi.forecast import forecast_workflow

    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(tmp_path))
    _, calls = install(monkeypatch)
    root = tmp_path / "interrupted-collection"
    original_write = forecast_workflow.atomic_write
    with monkeypatch.context() as interrupted:

        def missing_pointer(path, body):
            if path.name == "collection.json":
                raise OSError("injected collection pointer interruption")
            return original_write(path, body)

        interrupted.setattr(forecast_workflow, "atomic_write", missing_pointer)
        with pytest.raises(OSError, match="collection pointer"):
            run_forecast("SOXL", root, request_id="collection-recovery", prepare_only=True)
    assert not (root / "collection.json").exists()
    assert len(calls) == 2

    def forbidden(*args, **kwargs):
        pytest.fail("completed source was silently downloaded again")

    monkeypatch.setattr(daily_data, "urlopen", forbidden)
    prepared = run_forecast("SOXL", root, request_id="collection-recovery", prepare_only=True)
    assert prepared["status"] == "prepared"
    assert (root / "collection.json").exists()
    assert not (root / "research").exists()
    assert len(list((root / "collections").iterdir())) == 1
