"""The offline example must prepare a real registered envelope without fitting."""

import os
import subprocess
import sys
from pathlib import Path

from quant_rsi.core.artifacts import canonical, decode
from quant_rsi.factors.discovery import DiscoveryPlan
from quant_rsi.forecast.forecast_inputs import ForecastInputs
from quant_rsi.market.market_contracts import AccountRules


def test_offline_workflow_example_is_runnable_and_repeated_preparation_is_stable(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "prepare_research_workflow.py"
    output = tmp_path / "prepared"
    command = [sys.executable, str(script), "--fixture", "--output", str(output)]
    env = os.environ | {"QUANT_RSI_WORKSPACE": str(tmp_path)}
    first = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert "invented prices" in first.stdout
    frozen = (output / "profile.json").read_bytes()
    profile = decode(frozen)
    inputs = ForecastInputs(profile["inputs"])
    plan = DiscoveryPlan.model_validate_json(canonical(profile["discovery"]))
    rules = AccountRules.model_validate_json(canonical(profile["account"]))
    assert inputs.target[0].instruments == ("KO",)
    assert inputs.target[0].spec.sessions == 100
    assert len(inputs.target[0].contract.calendar.sessions) == 105
    assert plan.exposure == "engineering_fixture" and plan.max_rounds == 2
    assert plan.fold_origins == ((80,), (90,))
    assert rules.max_positions == 1 and rules.termination == "preserve_inventory"
    subprocess.run(command, env=env, capture_output=True, text=True, check=True)
    assert (output / "profile.json").read_bytes() == frozen
