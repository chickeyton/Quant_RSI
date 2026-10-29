"""Audit research artifacts, not factor execution or predictive performance."""

import ast
import hashlib
import json
import os
import re
import statistics
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "experimental" / "candidate_factors"


def read_json(path):
    return json.loads(path.read_text())


SOURCES = {s["id"]: s for s in read_json(ROOT / "sources.json")["sources"]}
CANDIDATES = [
    candidate
    for path in sorted((ROOT / "definitions").glob("*.json"))
    for candidate in read_json(path)["candidates"]
]
ARITHMETIC = read_json(ROOT / "fixtures" / "arithmetic.json")["cases"]


def test_sources_preserve_license_bytes_and_pinned_identities():
    for source in SOURCES.values():
        assert re.fullmatch(r"[0-9a-f]{40}", source["commit"])
        assert source["license_spdx"] in {"MIT", "Apache-2.0"}
        assert source["repository"].startswith("https://github.com/")
        assert len({f["path"] for f in source["files"]}) == len(source["files"])
        for file in source["files"]:
            assert re.fullmatch(r"[0-9a-f]{64}", file["sha256"])
        for artifact in source["license_artifacts"]:
            path = (ROOT / artifact["local_path"]).resolve()
            assert path.is_relative_to(ROOT / "licenses")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    gs_notices = {a["upstream_path"] for a in SOURCES["gs-quant"]["license_artifacts"]}
    assert {"LICENSE", "NOTICE", "NOTICE.txt"} <= gs_notices
    qd_notices = {a["upstream_path"] for a in SOURCES["QuantDinger"]["license_artifacts"]}
    assert {"REUSE.toml", "LICENSES/Apache-2.0.txt"} <= qd_notices


def test_candidate_references_and_evidence_status():
    assert CANDIDATES
    assert len({c["id"] for c in CANDIDATES}) == len(CANDIDATES)
    documented_modes = set(re.findall(r"`([a-z_]+)`", (ROOT / "execution_modes.md").read_text()))
    for candidate in CANDIDATES:
        ref = candidate["source"]
        source = SOURCES[ref["source_id"]]
        assert ref["path"] in {f["path"] for f in source["files"]}
        start, end = ref["lines"]
        assert 1 <= start <= end
        assert ref["url"] == (
            f"{source['repository']}/blob/{source['commit']}/{ref['path']}#L{start}-L{end}"
        )
        assert candidate["execution_modes"]
        assert set(candidate["execution_modes"]) <= documented_modes
        for field in (
            "inputs",
            "formula_summary",
            "output_units",
            "warmup_and_state",
            "semantic_edges",
            "integration_requirements",
        ):
            assert candidate[field], (candidate["id"], field)
        assert candidate["upstream_runtime_conformance"] == "not_run"
        assert candidate["predictive_evidence"] == "not_assessed"
        if "centered_noncausal" in candidate["execution_modes"]:
            assert candidate["review_status"] == "reference_only_noncausal"


def test_inventory_and_fixture_references():
    for path in (ROOT / "inventories").glob("*.json"):
        inventory = read_json(path)
        source = SOURCES[inventory["source_id"]]
        assert inventory["scope"] and inventory["entries"]
        for entry in inventory["entries"]:
            if source["id"] == "Lean":
                assert re.fullmatch(r"Indicators/[^/]+\.cs", entry["path"])
            else:
                assert entry["path"] in {f["path"] for f in source["files"]}
                assert 1 <= entry["lines"][0] <= entry["lines"][1]
    ids = {c["id"] for c in CANDIDATES}
    for path in (ROOT / "fixtures").glob("*.json"):
        cases = read_json(path)["cases"]
        assert len({c["id"] for c in cases}) == len(cases)
        for case in cases:
            assert case["candidate_ids"] and set(case["candidate_ids"]) <= ids


@pytest.mark.parametrize("case", ARITHMETIC, ids=lambda case: case["id"])
def test_synthetic_oracle_arithmetic(case):
    """Recalculate tiny examples only; these are not upstream conformance tests."""
    x = case["inputs"]
    match case["id"]:
        case "reciprocal_roc":
            actual = x["close"][-1 - x["d"]] / x["close"][-1]
        case "bar_shape":
            actual = [(x["close"] - x["open"]) / x["open"], (x["high"] - x["low"]) / x["open"]]
        case "cross_section_ties":
            values = x["values"]
            ranks = [
                statistics.mean(i + 1 for i, v in enumerate(sorted(values)) if v == value)
                for value in values
            ]
            actual = [(rank / len(values) - 0.5) * 3.46 for rank in ranks]
        case "ema_seed":
            actual = [x["values"][0]]
            for value in x["values"][1:]:
                actual.append(x["beta"] * actual[-1] + (1 - x["beta"]) * value)
        case "market_beta":
            actual = [
                None,
                None,
                None,
                statistics.covariance(x["asset_returns"], x["benchmark_returns"])
                / statistics.variance(x["benchmark_returns"]),
            ]
            assert actual[:3] == case["expected"][:3]
            assert actual[3] == pytest.approx(case["expected"][3])
            return
        case "obv_seed_and_flat":
            actual = [x["volume"][0]]
            for i in range(1, len(x["close"])):
                delta = x["close"][i] - x["close"][i - 1]
                sign = (delta > 0) - (delta < 0)
                actual.append(actual[-1] + sign * x["volume"][i])
        case "vwap_price_basis":
            actual = {}
            for source, fields in (
                ("lean", ("open", "high", "low", "close")),
                ("quantdinger", ("high", "low", "close")),
            ):
                amount = sum(
                    statistics.mean(x[f][i] for f in fields) * volume
                    for i, volume in enumerate(x["volume"])
                )
                actual[source] = amount / sum(x["volume"])
        case "earnings_yield_ratio":
            actual = x["net_income"] / x["market_cap"]
        case _:
            pytest.fail(f"No arithmetic check registered for {case['id']}")
    assert actual == pytest.approx(case["expected"])


def test_optional_pinned_upstream_audit():
    cache = os.environ.get("QUANT_RSI_UPSTREAM_CACHE")
    if not cache:
        pytest.skip("Set QUANT_RSI_UPSTREAM_CACHE to audit pinned local upstream snapshots")
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"}
    for source in SOURCES.values():
        repo = Path(cache) / source["id"]

        def git(*args, repo=repo):
            return subprocess.check_output(["git", "-C", str(repo), *args], env=env)

        assert git("rev-parse", "HEAD").decode().strip() == source["commit"]
        blobs = {}
        for file in source["files"]:
            blob = git("show", f"HEAD:{file['path']}")
            assert hashlib.sha256(blob).hexdigest() == file["sha256"], file["path"]
            blobs[file["path"]] = blob.decode().splitlines()
        for artifact in source["license_artifacts"]:
            blob = git("show", f"HEAD:{artifact['upstream_path']}")
            assert hashlib.sha256(blob).hexdigest() == artifact["sha256"]
        for candidate in CANDIDATES:
            ref = candidate["source"]
            if ref["source_id"] != source["id"]:
                continue
            lines = blobs[ref["path"]]
            start, end = ref["lines"]
            assert end <= len(lines)
            assert ref["symbol"].split(".")[-1] in "\n".join(lines[start - 1 : end])
        inventory = read_json(ROOT / "inventories" / f"{source['id']}.json")
        if source["id"] == "Lean":
            paths = set(
                git("ls-tree", "-r", "--name-only", "HEAD", "Indicators/").decode().splitlines()
            )
            assert {e["path"] for e in inventory["entries"]} <= paths
        else:
            for entry in inventory["entries"]:
                start, end = entry["lines"]
                lines = blobs[entry["path"]]
                assert end <= len(lines)
                if "expression_template" in entry:
                    # Adjacent Python string literals can span multiple lines.
                    literals = {
                        (node.lineno, node.end_lineno, node.value)
                        for node in ast.walk(ast.parse("\n".join(lines)))
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    }
                    assert (start, end, entry["expression_template"]) in literals
                else:
                    key = "symbol" if "symbol" in entry else "id"
                    assert entry[key] in "\n".join(lines[start - 1 : end])
