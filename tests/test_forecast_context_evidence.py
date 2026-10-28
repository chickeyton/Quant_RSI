"""Offline preparation-to-sealing evidence regression; no HTTP or model fits."""

from pathlib import Path

import pytest
from test_forecast_context_data import audit_package

from quant_rsi import forecast_context_data as context
from quant_rsi.artifacts import digest
from quant_rsi.baseline import validate_forecast_evidence
from quant_rsi.forecast_inputs import ForecastInputs
from quant_rsi.market_data import DataRegistry


@pytest.fixture
def prepared(tmp_path):
    audit = audit_package(tmp_path)
    out = tmp_path / "prepared"
    result = context.prepare_context_data(audit, out, history_sessions=24)
    return audit, out, result


def evidence_ref(result, kind):
    objective = result["inputs"]["target"]["objective"]
    if kind == "notice":
        return objective["notices"][0]["evidence_ref"]
    return objective[kind + "_evidence_ref"]


def registry_files(root):
    return {
        str(path.relative_to(root)): digest(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_preparation_satisfies_forecast_sealing_evidence(prepared, monkeypatch):
    _, out, result = prepared
    inputs = ForecastInputs(result["inputs"])
    registry_root = Path(result["meta"]["registry_path"])
    with DataRegistry(registry_root) as registry:
        for data, objective in [inputs.target, *inputs.contexts.values()]:
            # Invoke the real sealing gate, including its action-payload checks.
            validate_forecast_evidence(registry, objective, data.snapshot)
            assert objective.notices
            assert objective.price_unit_claim == "explicit_assumption"
            assert data.contract.knowledge_basis == "assumed_historical_availability"
        for ref, raw in inputs.evidence.items():
            assert registry.store.get(ref) == raw
    before = registry_files(registry_root)
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(out / "workspace"))
    context._ingest_prepared(out / "prepared.json")
    assert registry_files(registry_root) == before


@pytest.mark.parametrize("kind", ["price_unit", "calendar", "notice"])
def test_ingestion_rejects_existing_corrupt_evidence(prepared, monkeypatch, kind):
    _, out, result = prepared
    registry_root = Path(result["meta"]["registry_path"])
    ref = evidence_ref(result, kind)
    artifact = registry_root / "artifacts" / ref
    assert artifact.exists()
    artifact.write_bytes(b"corrupt registered evidence")
    prior_inputs = (out / "inputs.json").read_bytes()
    prior_receipt = (out / "preparation.receipt.json").read_bytes()
    monkeypatch.setenv("QUANT_RSI_WORKSPACE", str(out / "workspace"))
    with pytest.raises(ValueError, match="artifact integrity failure"):
        context._ingest_prepared(out / "prepared.json")
    assert artifact.read_bytes() == b"corrupt registered evidence"
    assert (out / "inputs.json").read_bytes() == prior_inputs
    assert (out / "preparation.receipt.json").read_bytes() == prior_receipt


@pytest.mark.parametrize("defect", ["missing", "corrupt"])
def test_resume_does_not_repair_prepared_evidence(prepared, defect):
    audit, out, result = prepared
    ref = evidence_ref(result, "price_unit")
    artifact = Path(result["evidence_paths"][ref])
    registry_root = Path(result["meta"]["registry_path"])
    before = registry_files(registry_root)
    if defect == "missing":
        artifact.unlink()
        expected = FileNotFoundError
    else:
        artifact.write_bytes(b"corrupt original evidence")
        expected = ValueError
    with pytest.raises(expected):
        context.prepare_context_data(audit, out, history_sessions=24, resume=True)
    assert registry_files(registry_root) == before
    if defect == "missing":
        assert not artifact.exists()
    else:
        assert artifact.read_bytes() == b"corrupt original evidence"
