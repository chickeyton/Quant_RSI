"""Protected-only controls for both frontier transitions; no workers or authority.

The companion fixed-context tests exercise typed development-feedback changes.
These synthetic packets add the full decision/grant invariance check without
claiming authenticated market histories or executed descendant research.
"""

import builtins
import io
from pathlib import Path

import pytest
from test_forecast_frontier_policy import frontier_packet
from test_forecast_recursive_changes import plan_value

from quant_rsi import changes, forecast_frontier_policy, package_experiment
from quant_rsi.artifacts import ArtifactStore, canonical


@pytest.mark.parametrize(
    "cap,scores",
    [
        (3, (-4.0, -3.0, -2.0, -3.5)),
        (4, (-5.0, -4.0, -3.0, -2.0, -3.5)),
    ],
)
def test_protected_only_feedback_preserves_full_decision_and_grants(
    cap, scores, tmp_path, monkeypatch
):
    plan = plan_value(cap=cap)
    args = frontier_packet(plan, scores)
    _, request, evidence = args
    fixed = canonical(dict(request=request, evidence=evidence, plan=plan.model_dump(mode="json")))
    store = ArtifactStore(tmp_path / "objects")
    manifest = dict(
        campaign_id="00000000000000000000000000000002",
        request_ref=store.put_json(request),
        evidence_ref=store.put_json(evidence),
    )
    stored_refs = {manifest["request_ref"], manifest["evidence_ref"]}
    original_objects = {ref: store.get(ref) for ref in stored_refs}
    protected = tmp_path / "protected.json"
    protected_store = ArtifactStore(tmp_path / "protected_objects")
    implementation = Path(forecast_frontier_policy.__file__).parent
    allowed_paths = {
        (implementation / name).resolve()
        for name in (
            "forecast_frontier_policy.py",
            "researcher.py",
            "research.py",
            "research_journal.py",
            "task_definition.py",
        )
    } | {(store.root / ref).resolve() for ref in stored_refs}
    original_open, original_get = io.open, ArtifactStore.get
    reads, results = [], []

    def declared_open(path, mode="r", *positional, **keywords):
        assert not isinstance(path, int), "unexpected file descriptor read"
        resolved = Path(path).resolve()
        assert resolved in allowed_paths, "decision read undeclared/protected evidence"
        assert mode in {"r", "rb"}, "pure decision attempted a file write"
        reads.append(resolved)
        return original_open(path, mode, *positional, **keywords)

    def declared_get(current_store, ref):
        assert current_store is store and ref in stored_refs, "undeclared artifact read"
        return original_get(current_store, ref)

    for protected_value in (
        '{"score":100,"selected":"favorable"}',
        '{"score":-100,"selected":"opposite","outcome":"no_patch"}',
    ):
        protected.write_text(protected_value)
        protected_ref = protected_store.put(protected_value.encode())
        with monkeypatch.context() as scoped:
            scoped.setattr(io, "open", declared_open)
            scoped.setattr(builtins, "open", declared_open)
            scoped.setattr(ArtifactStore, "get", declared_get)
            with pytest.raises(AssertionError, match="undeclared/protected"):
                protected.read_text()
            with pytest.raises(AssertionError, match="undeclared artifact"):
                protected_store.get(protected_ref)
            response = changes.propose_change(*args)
            validation = changes.validate_response(response, *args)
            decision_ref, decision, objects = package_experiment.decision_content(
                store, manifest, plan, response
            )
        assert validation == ("patch", None)
        assert decision["outcome"] == "patch"
        assert response["patch"]["value"] == cap + 1
        assert set(decision["grants"]) == {"parent", "candidate"}
        for arm, grant_ref in decision["grants"].items():
            grant = objects[grant_ref]
            assert grant["arm"] == arm
            assert grant["budget"] == plan.development.budget.model_dump(mode="json")
            arm_plan = objects[grant["research_plan_ref"]]
            assert arm_plan["policy"]["window_max"] == cap + (arm == "candidate")
        results.append(canonical([response, validation, decision_ref, decision, objects]))
        assert (
            canonical(dict(request=request, evidence=evidence, plan=plan.model_dump(mode="json")))
            == fixed
        )
        assert {path.name for path in store.root.iterdir()} == stored_refs
        assert {ref: store.get(ref) for ref in stored_refs} == original_objects

    assert results[0] == results[1]
    assert reads and protected.resolve() not in reads
