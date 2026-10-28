"""Bounded discovery: actual forecast interfaces, deterministic engineering data."""

from copy import deepcopy
from datetime import timedelta

import pytest
from test_forecast_comparison import dummy_worker as worker_fixture
from test_forecast_features import prepare_inputs as inputs_fixture

from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.factors import discovery, lifecycle
from quant_rsi.factors.discovery_graph import (
    FactorExpression,
    combine,
    evaluate,
    grammar,
    leaf_expression,
    propose,
)
from quant_rsi.forecast.forecast_experiment import (
    experiment_root,
    prepare_experiment,
)
from quant_rsi.market import market_contracts
from quant_rsi.market.market_contracts import timestamp, utc_text


@pytest.fixture
def prepare_inputs(tmp_path, monkeypatch):
    return inputs_fixture.__wrapped__(tmp_path, monkeypatch)


@pytest.fixture
def dummy_worker(monkeypatch):
    return worker_fixture.__wrapped__(monkeypatch)


def plan(**changes):
    value = dict(
        campaign_id="discovery-test",
        instrument="KO",
        exposure="engineering_fixture",
        max_rounds=3,
        fold_origins=[[80], [90]],
        leaves=[
            dict(
                name="momentum",
                role="target",
                definition=dict(
                    factor_id="equity.price_momentum_5",
                    description="Registered price momentum",
                    kind="numeric",
                    input_kind="positive_price_series",
                    recipe=dict(op="momentum", window=5),
                ),
            ),
            dict(
                name="mean_return",
                role="target",
                definition=dict(
                    factor_id="equity.price_mean_return_5",
                    description="Registered mean return",
                    kind="numeric",
                    input_kind="positive_price_series",
                    recipe=dict(op="mean_return", window=5),
                ),
            ),
        ],
    )
    return discovery.DiscoveryPlan.model_validate_json(canonical(value | changes))


def test_typed_dag_rejects_cycles_unused_nodes_and_unbounded_expressions():
    with pytest.raises(ValueError, match="topologically"):
        FactorExpression.model_validate_json(
            canonical(dict(nodes=[dict(name="x", op="add", inputs=["x", "x"])]))
        )
    with pytest.raises(ValueError, match="unused"):
        FactorExpression.model_validate_json(
            canonical(
                dict(
                    nodes=[
                        dict(name="x", op="leaf", leaf="momentum"),
                        dict(name="y", op="leaf", leaf="mean_return"),
                    ]
                )
            )
        )
    left, right = leaf_expression("momentum"), leaf_expression("mean_return")
    assert combine(left, right, "multiply").ref == combine(right, left, "multiply").ref
    assert combine(left, right, "subtract").ref != combine(right, left, "subtract").ref
    graph = combine(left, right, "add")
    with pytest.raises(ValueError):
        combine(combine(graph, graph, "add"), graph, "add")


@pytest.mark.parametrize("values", [(1e200, 1e200), (float("inf"), 1.0)])
def test_numeric_overflow_excludes_the_same_rows_for_all_candidates(monkeypatch, values):
    from quant_rsi.factors import discovery_graph

    monkeypatch.setattr(
        discovery_graph,
        "source_leaves",
        lambda *args: (dict(zip(("momentum", "mean_return"), values, strict=True)), [], set()),
    )
    results = [evaluate(graph, plan().leaves, None, None, None) for graph in grammar(plan().leaves)]
    assert all(result["value"] is None for result in results)
    assert all(result["reasons"] == ["registered_grammar_nonfinite"] for result in results)


def test_feedback_changes_operator_and_failed_step_keeps_frontier():
    frontier = grammar(plan().leaves)
    first = propose(frontier, [])
    good = [dict(expression_ref=first.ref, improvement_pp=0.1)]
    failed = [dict(expression_ref=first.ref, improvement_pp=None)]
    assert propose(frontier, good).nodes[-1].op == "multiply"
    assert propose(frontier, failed).nodes[-1].op == "subtract"
    assert len(frontier) == 5
    assert first.nodes[-1].op == "leaf"


def test_numeric_sources_are_timed_and_multi_leaf_expression_executes(prepare_inputs):
    inputs, _ = prepare_inputs(future_calendar=5, target_instrument="KO")
    graph = combine(leaf_expression("momentum"), leaf_expression("mean_return"), "subtract")
    result = evaluate(
        graph, plan().leaves, inputs, inputs.target[0].dates[80], inputs.target[0].decision_at(80)
    )
    assert result["value"] is not None and not result["reasons"] and result["source_refs"]
    result = evaluate(
        graph, plan().leaves, inputs, inputs.target[0].dates[80], inputs.target[0].decision_at(70)
    )
    assert result["value"] is None and result["reasons"]


def test_campaign_worker_frontier_budget_replay_and_rebinding(prepare_inputs, dummy_worker):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    initial = discovery.initialize(envelope, plan())
    assert initial["allocated_fits"] == 15 and initial["fits_attempted"] == 0
    partial = discovery.resume("discovery-test", max_fits=2)
    assert partial["fits_attempted"] == 2
    report = discovery.resume("discovery-test", max_fits=20)
    assert report["status"] == "completed"
    assert report["fits_attempted"] == dummy_worker["fits"] == 15
    assert report["reserved_worker_seconds"] == 2700
    assert [h["novelty"] for h in report["history"]] == [
        "library_selection",
        "composed_definition",
        "library_selection",
    ]
    assert report["selection"] == "retain_baseline"
    assert discovery.inspect("discovery-test", replay=True) == report
    assert discovery.resume("discovery-test", max_fits=20) == report
    assert discovery.initialize(envelope, plan()) == report
    assert dummy_worker["fits"] == 15
    changed = deepcopy(envelope)
    changed["evidence_hex"][digest(b"different")] = b"different".hex()
    with pytest.raises(ValueError, match="already binds"):
        discovery.initialize(changed, plan())
    assert report["promotion_decision"] == "not_requested"


def test_child_cannot_rename_or_reset_allocation(prepare_inputs, dummy_worker):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=1))
    discovery.resume("discovery-test", max_fits=1)
    journal, store, _, _, _ = discovery._load("discovery-test")
    try:
        allocation = discovery.event_payload(journal, "discovery_allocation_0")
        binding = store.json(allocation["binding_ref"])
    finally:
        journal.close()
    with pytest.raises(ValueError, match="immutable allocation"):
        prepare_experiment(
            envelope, "renamed-child", fold_origins=plan().fold_origins, discovery=binding
        )
    changed = binding | {"round": 4}
    with pytest.raises(ValueError, match="outside"):
        prepare_experiment(
            envelope, "renamed-child", fold_origins=plan().fold_origins, discovery=changed
        )
    assert dummy_worker["fits"] == 1


def test_failed_candidate_does_not_end_exploration(prepare_inputs, dummy_worker):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=2))
    dummy_worker["fail"].add(1)
    report = discovery.resume("discovery-test", max_fits=20)
    assert len(report["history"]) == 2
    assert report["history"][0]["improvement_pp"] is None
    assert report["history"][1]["novelty"] == "composed_definition"
    assert report["reserved_worker_seconds"] == 1800


def test_missing_authority_fails_closed(prepare_inputs):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan())
    (discovery.campaign_root("discovery-test") / "journal.sqlite").unlink()
    with pytest.raises(ValueError, match="initialized journal"):
        discovery.initialize(envelope, plan())


def test_instance_lifecycle_time_decay_retraction_and_snapshot(prepare_inputs, monkeypatch):
    inputs, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    now = inputs.target[0].decision_at(60)
    clock = [now]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: utc_text(clock[0]))
    template = lifecycle.EventTemplate(
        template_id="issuer.supply_event",
        description="Fixture hypothesis",
        ttl_hours=48,
        decay="linear",
    )
    raw = canonical(dict(title="Invented engineering event"))
    evidence = {digest(raw): raw}
    observed = lifecycle.create("event-one", "KO", template, evidence)
    assert lifecycle.value_at(observed, now - timedelta(seconds=1))["value"] is None
    assert lifecycle.value_at(observed, now)["value"] is None
    for state in ("validated", "experimental", "active_for_registered_research"):
        observed = lifecycle.transition("event-one", state, state, "fixture admission", evidence)
    ref = digest(canonical(observed))
    attached = lifecycle.attach_instance(envelope, observed)
    assert ref in attached["evidence_hex"]
    assert lifecycle.value_at(observed, now)["value"] == 1
    assert lifecycle.value_at(observed, now + timedelta(hours=24))["value"] == 0.5
    assert lifecycle.value_at(observed, now + timedelta(hours=48))["value"] == 0
    clock[0] += timedelta(hours=1)
    withdrawn = lifecycle.transition(
        "event-one", "withdrawn", "inactive", "source corrected", evidence
    )
    assert lifecycle.value_at(withdrawn, now)["value"] == 1
    assert lifecycle.value_at(withdrawn, clock[0])["value"] == 0
    lifecycle.verify_snapshot(observed)
    with pytest.raises(ValueError, match="complete current"):
        lifecycle.verify_snapshot(observed, latest=True)
    assert (
        lifecycle.transition("event-one", "withdrawn", "inactive", "source corrected", evidence)
        == withdrawn
    )
    with pytest.raises(ValueError, match="already binds"):
        lifecycle.transition("event-one", "withdrawn", "archived", "changed", evidence)
    archived = lifecycle.transition("event-one", "archive", "archived", "retain history", evidence)
    assert len(archived["transitions"]) == 6


def test_real_lightgbm_discovery_pair(prepare_inputs):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=1))
    report = discovery.resume("discovery-test", max_fits=5)
    assert report["status"] == "completed"
    assert report["fits_attempted"] == 5
    assert report["history"][0]["improvement_pp"] is not None
    assert discovery.inspect("discovery-test", replay=True) == report


def test_catalog_binding_executes_shared_definitions(prepare_inputs):
    from pathlib import Path

    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    envelope, leaves = discovery.bind_catalog(
        envelope,
        Path(__file__).parents[1] / "instruments",
        "US",
        "KO",
        {"price_momentum": "target", "price_mean_return": "target"},
    )
    configured = plan(leaves=[leaf.model_dump(mode="json") for leaf in leaves])
    report = discovery.initialize(envelope, configured)
    assert report["allocated_fits"] == 15
    assert all(leaf.binding_ref in envelope["evidence_hex"] for leaf in leaves)
    with pytest.raises(ValueError, match="no numeric executor"):
        discovery.bind_catalog(
            envelope,
            Path(__file__).parents[1] / "instruments",
            "US",
            "KO",
            {"announced_events": "target"},
        )


def test_common_rows_are_identical_across_discovery_rounds(prepare_inputs, dummy_worker):
    from quant_rsi.forecast.forecast_experiment import open_experiment

    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO", missing_market=70)
    leaves = [leaf.model_dump(mode="json") for leaf in plan().leaves]
    leaves[1]["role"] = "market"
    discovery.initialize(envelope, plan(leaves=leaves, max_rounds=2, fold_origins=[[85], [91]]))
    report = discovery.resume("discovery-test", max_fits=20)
    masks = []
    for round_ in report["history"]:
        journal, store, manifest, saved, inputs = open_experiment(
            experiment_root(round_["experiment_id"])
        )
        try:
            protocol = store.json(saved.protocol_ref)
            masks.append(
                [
                    store.json(fold[purpose + "_mask_ref"])["eligible_indices"]
                    for fold in protocol["folds"]
                    for purpose in ("training", "evaluation")
                ]
            )
        finally:
            journal.close()
    assert len(masks) == 2 and masks[0] == masks[1]
    assert 70 not in masks[0][0]


def test_insufficient_evidence_is_terminal_without_fits(prepare_inputs, monkeypatch):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    envelope["contexts"].pop("industry")
    leaves = [leaf.model_dump(mode="json") for leaf in plan().leaves]
    leaves[1]["role"] = "industry"
    discovery.initialize(envelope, plan(leaves=leaves))
    report = discovery.resume("discovery-test", max_fits=10)
    assert report["terminal"]["reason"] == "insufficient_evidence"
    assert report["fits_attempted"] == report["reserved_worker_seconds"] == 0
    monkeypatch.setattr(
        discovery, "prepare_experiment", lambda *a, **k: pytest.fail("terminal refreeze")
    )
    assert discovery.resume("discovery-test", max_fits=10) == report
    assert discovery.inspect("discovery-test", replay=True) == report


def test_interrupted_proposal_checkpoint_is_reused(prepare_inputs, dummy_worker, monkeypatch):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=1))
    prepare = discovery.prepare_experiment

    def crash(*args, **kwargs):
        raise RuntimeError("after allocation")

    monkeypatch.setattr(discovery, "prepare_experiment", crash)
    with pytest.raises(RuntimeError, match="after allocation"):
        discovery.resume("discovery-test", max_fits=5)
    monkeypatch.setattr(discovery, "prepare_experiment", prepare)
    monkeypatch.setattr(discovery, "propose", lambda *a, **k: pytest.fail("proposal regenerated"))
    report = discovery.resume("discovery-test", max_fits=5)
    assert report["proposals_created"] == 1 and dummy_worker["fits"] == 5


def test_lost_allocation_does_not_hide_completed_charges(prepare_inputs, dummy_worker):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    discovery.initialize(envelope, plan(max_rounds=1))
    discovery.resume("discovery-test", max_fits=5)
    journal, _, _, _, _ = discovery._load("discovery-test")
    with journal.connection:
        journal.connection.execute("DELETE FROM events WHERE kind='discovery_allocation_0'")
    journal.close()
    with pytest.raises(ValueError, match="lost its allocation"):
        discovery.inspect("discovery-test")


def test_event_leaf_freezes_history_and_new_campaign_rejects_omitted_transition(
    prepare_inputs, monkeypatch
):
    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    now = timestamp("2025-01-01T00:00:00Z")
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: utc_text(now))
    template = lifecycle.EventTemplate(
        template_id="issuer.event", description="fixture", ttl_hours=24
    )
    evidence = {digest(b"fixture"): b"fixture"}
    value = lifecycle.create("leaf-event", "KO", template, evidence)
    envelope = lifecycle.attach_instance(envelope, value)
    leaves = [
        plan().leaves[0].model_dump(mode="json"),
        dict(name="event", role="event", instance_ref=digest(canonical(value))),
    ]
    discovery.initialize(envelope, plan(leaves=leaves))
    lifecycle.transition("leaf-event", "validated", "validated", "checked", evidence)
    # Existing frozen study remains reproducible. New study cannot select an old prefix.
    assert discovery.inspect("discovery-test")["fits_attempted"] == 0
    with pytest.raises(ValueError, match="complete current"):
        discovery.initialize(envelope, plan(leaves=leaves, campaign_id="different"))


def test_stored_forged_lifecycle_snapshot_is_not_authoritative(prepare_inputs):
    prepare_inputs(future_calendar=5, target_instrument="KO")
    template = lifecycle.EventTemplate(
        template_id="issuer.event", description="fixture", ttl_hours=24
    )
    evidence = {digest(b"fixture"): b"fixture"}
    value = lifecycle.create("forgery-test", "KO", template, evidence)
    forged = deepcopy(value)
    forged["transitions"][0]["state"] = "active_for_registered_research"
    # A supplied document can contain any JSON, including this forged snapshot.
    payload = canonical(forged)
    lifecycle.transition(
        "forgery-test", "validated", "validated", "document retained", {digest(payload): payload}
    )
    with pytest.raises(ValueError, match="authoritative history prefix"):
        lifecycle.verify_snapshot(forged)


def test_cli_init_status_resume_replay(prepare_inputs, dummy_worker, tmp_path, capsys):
    import json

    from quant_rsi.cli import main

    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    (tmp_path / "plan.json").write_bytes(canonical(plan(max_rounds=1).model_dump(mode="json")))
    (tmp_path / "inputs.json").write_bytes(canonical(envelope))
    assert (
        main(
            [
                "factor-discovery",
                "init",
                str(tmp_path / "plan.json"),
                "--inputs",
                str(tmp_path / "inputs.json"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["factor-discovery", "resume", "discovery-test", "--max-fits", "5"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "completed"
    for command in ("status", "replay"):
        assert main(["factor-discovery", command, "discovery-test"]) == 0
        assert json.loads(capsys.readouterr().out) == report


def test_news_authorization_cannot_mint_an_extra_legacy_pair(prepare_inputs, tmp_path):
    from test_news_factor import run

    from quant_rsi.factors.news import attach_news_result

    _, envelope = prepare_inputs(future_calendar=5, target_instrument="KO")
    result = run(tmp_path)
    envelope = attach_news_result(envelope, result)
    leaves = [plan().leaves[0].model_dump(mode="json"), dict(name="news", role="news")]
    configured = plan(leaves=leaves)
    discovery.initialize(envelope, configured)
    with pytest.raises(ValueError, match="single comparison allowance"):
        prepare_experiment(
            envelope,
            "discovery-discovery-test",
            news_factor=True,
            fold_origins=configured.fold_origins,
        )
    outcome = discovery.resume("discovery-test", max_fits=5)
    assert outcome["terminal"]["reason"] == "insufficient_evidence"
    assert outcome["fits_attempted"] == 0


def test_news_observer_creates_retires_and_does_not_resurrect(
    prepare_inputs, tmp_path, monkeypatch
):
    from test_news_factor import END, corpus, request, run

    from quant_rsi.factors.event_observer import observe

    prepare_inputs(future_calendar=5, target_instrument="KO")
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: END)
    positive = run(tmp_path)
    template = lifecycle.EventTemplate(
        template_id="issuer.earnings_release_title",
        description="Exact title indicator",
        ttl_hours=168,
    )
    report = observe(positive, template)
    assert report["outcomes"][0]["status"] == "active_for_registered_research"
    instance_id = report["outcomes"][0]["instance_id"]
    old = lifecycle.snapshot(instance_id)
    assert observe(positive, template) == report
    value, evidence = corpus(title="Corrected title: results will be released later", revision=1)
    later = "2025-07-08T00:00:01Z"
    negative = run(
        tmp_path, value, evidence, req=request(request_id="revision"), completed_at=later
    )
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: later)
    retired = observe(negative, template)
    assert retired["outcomes"][0]["status"] == "inactive"
    assert lifecycle.value_at(lifecycle.snapshot(instance_id), END)["value"] == 1
    assert lifecycle.value_at(lifecycle.snapshot(instance_id), later)["value"] == 0
    assert observe(positive, template)["outcomes"][0]["status"] == "older_revision_ignored"
    lifecycle.verify_snapshot(old)


def test_expiry_archives_without_erasing_earlier_active_values(prepare_inputs, monkeypatch):
    prepare_inputs(future_calendar=5, target_instrument="KO")
    now = timestamp("2025-01-01T00:00:00Z")
    clock = [now]
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: utc_text(clock[0]))
    template = lifecycle.EventTemplate(
        template_id="issuer.event", description="fixture", ttl_hours=1
    )
    evidence = {digest(b"receipt"): b"receipt"}
    lifecycle.create("expire-test", "KO", template, evidence)
    for state in ("validated", "experimental", "active_for_registered_research"):
        lifecycle.transition("expire-test", state, state, "fixture admission", evidence)
    clock[0] += timedelta(hours=2)
    archived = lifecycle.expire("expire-test")
    assert archived["transitions"][-1]["state"] == "archived"
    assert lifecycle.value_at(archived, now)["value"] == 1
    assert lifecycle.value_at(archived, clock[0])["value"] == 0
    assert lifecycle.expire("expire-test") == archived


def test_unresolved_revision_suspends_without_becoming_false(prepare_inputs, tmp_path, monkeypatch):
    from test_news_factor import END, corpus, request, run

    from quant_rsi.factors.event_observer import observe

    prepare_inputs(future_calendar=5, target_instrument="KO")
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: END)
    positive = run(tmp_path)
    template = lifecycle.EventTemplate(
        template_id="issuer.earnings_release_title",
        description="Exact title indicator",
        ttl_hours=1,
    )
    instance_id = observe(positive, template)["outcomes"][0]["instance_id"]
    value, evidence = corpus(title="x" * 5000, revision=1)
    later = "2025-07-08T00:00:01Z"
    uncertain = run(
        tmp_path, value, evidence, req=request(request_id="unresolved"), completed_at=later
    )
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: later)
    result = observe(uncertain, template)
    assert result["outcomes"][0]["status"] == "suspended"
    frozen = lifecycle.snapshot(instance_id)
    assert lifecycle.value_at(frozen, END)["value"] == 1
    assert lifecycle.value_at(frozen, later)["value"] is None
    assert observe(positive, template)["outcomes"][0]["status"] == "older_revision_ignored"
    monkeypatch.setattr(market_contracts, "trusted_now", lambda: "2025-07-08T02:00:00Z")
    archived = lifecycle.expire(instance_id)
    assert archived["transitions"][-1]["state"] == "archived"
    assert lifecycle.value_at(archived, "2025-07-08T02:00:00Z")["value"] is None
