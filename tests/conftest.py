from pathlib import Path

import pytest

from quant_rsi.schemas import DevelopmentPlan


@pytest.fixture
def plan():
    path = Path(__file__).parents[1] / "examples" / "development.json"
    return DevelopmentPlan.model_validate_json(path.read_bytes())
