from pathlib import Path

import pytest

from quant_rsi.core.schemas import DevelopmentPlan


@pytest.fixture(autouse=True)
def default_history(tmp_path, monkeypatch):
    """CLI studies record shared history by default; never write a real home directory."""
    root = tmp_path / "default-history"
    monkeypatch.setenv("QUANT_RSI_HISTORY_DIR", str(root))
    return root


@pytest.fixture
def plan():
    path = Path(__file__).parents[1] / "examples" / "development.json"
    return DevelopmentPlan.model_validate_json(path.read_bytes())
