"""Exercise the real profiling command with tiny offline pytest modules."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/profile_validation.py"


def run_profile(output, *selectors):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(output), "--shards", "2", *selectors],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result, json.loads((output / "summary.json").read_text())


def test_profile_keeps_selected_module_fixture_together_and_records_all_results(tmp_path):
    counter = tmp_path / "setup-count.txt"
    module = tmp_path / "test_shared.py"
    module.write_text(
        "import pytest\nfrom pathlib import Path\n"
        "@pytest.fixture(scope='module', autouse=True)\n"
        "def shared():\n"
        f"    with Path({str(counter)!r}).open('a') as handle: handle.write('setup\\n')\n"
        "def test_first(): pass\ndef test_second(): pass\n"
    )
    other = tmp_path / "test_other.py"
    other.write_text("def test_other(): pass\n")
    result, summary = run_profile(
        tmp_path / "profile", f"{module}::test_first", str(other), f"{module}::test_second"
    )
    assert result.returncode == summary["exit_code"] == 0, result.stdout + result.stderr
    assert len(summary["shards"]) == 2
    assert all(shard["exit_code"] == 0 for shard in summary["shards"])
    assert counter.read_text().splitlines() == ["setup"]
    metadata = json.loads((tmp_path / "profile/environment.json").read_text())
    assert metadata["shard_selectors"][0] == [f"{module}::test_first", f"{module}::test_second"]
    assert metadata["source"]["head"]
    assert metadata["source"]["files"]["examples/profile_validation.py"]
    assert metadata["environment"]["OMP_NUM_THREADS"] == "1"
    assert metadata["loaded_package"] == str(ROOT / "src/quant_rsi/__init__.py")
    assert summary["source_changed"] is False
    for index in range(2):
        assert (tmp_path / f"profile/shard-{index}.xml").exists()
        assert (tmp_path / f"profile/shard-{index}.command.json").exists()


@pytest.mark.parametrize("source, code", [("def test_failure(): assert False\n", 1), ("", 5)])
def test_profile_propagates_pytest_failure_or_empty_selection(tmp_path, source, code):
    module = tmp_path / "test_failure.py"
    module.write_text(source)
    result, summary = run_profile(tmp_path / "profile", str(module))
    assert result.returncode == summary["exit_code"] == code
    assert summary["shards"][0]["exit_code"] == code
