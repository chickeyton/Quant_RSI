"""Original registered forecast source, outer allowance, and irreversible handoff."""

import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from quant_rsi.core import controller
from quant_rsi.core.artifacts import ArtifactStore, atomic_write, canonical, decode, digest
from quant_rsi.research.researcher import ResearchPlan
from quant_rsi.research.trajectory_state import (
    TrajectoryPlan,
    anchor_path,
    locked_trajectory,
    require_owner,
    trajectory_root,
)


def registration_path():
    return trajectory_root().parent / "forecast_recursive_registration.json"


def trajectory_units(budget):
    return dict(
        worker_attempts=budget["max_attempts"],
        registered_worker_seconds=budget["max_worker_seconds"],
    )


def require_disjoint_source(source, root):
    source, root = Path(source).resolve(), Path(root).resolve()
    if source == root or root in source.parents or source in root.parents:
        raise ValueError("registered source and trajectory authority must be disjoint")


def identity_definition(plan, campaign_id, binding, grant):
    """Pure frozen identity; no mutable campaign or registration path is consulted."""
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import (
        _normalize,
        exposure_definition,
        policy_binding,
    )

    policy_binding(grant)
    frozen_plan = {
        key: value
        for key, value in grant.items()
        if key not in {"authority_ref", "workspace", "research_plan_ref", "exposure_ref"}
    } | {"kind": "ForecastRecursiveCampaign"}
    if _normalize(frozen_plan) != frozen_plan:
        raise ValueError("recursive frozen grant differs from its explicit normalized plan")
    if grant["exposure_ref"] != digest(canonical(exposure_definition(grant))):
        raise ValueError("recursive frozen grant exposure differs")
    if (
        plan.schema_version != 3
        or plan.package_lineage is not None
        or grant.get("kind") != "ForecastRecursiveCampaignGrant"
        or grant.get("schema_version") not in (3, 4)
        or grant["research_plan"] != plan.model_dump(mode="json")
        or grant["research_plan_ref"] != digest(canonical(plan.model_dump(mode="json")))
        or grant["source_campaign_id"] != campaign_id
        or binding["grant_ref"] != digest(canonical(grant))
        or binding["authority_ref"] != grant["authority_ref"]
        or binding["workspace"] != grant["workspace"]
        or set(binding)
        != {
            "authority_ref",
            "grant_ref",
            "workspace",
            "campaign_id",
            "source_directory",
            "source_campaign_id",
            "research_plan_ref",
            "exposure_ref",
        }
        or any(grant.get(key) != value for key, value in binding.items() if key != "grant_ref")
    ):
        raise ValueError("registered forecast source differs from its original shared grant")
    require_disjoint_source(
        grant["source_directory"], Path(grant["workspace"]) / ".quant_rsi" / "trajectory"
    )
    return dict(
        kind="RegisteredForecastGen0Identity",
        schema_version=3,
        binding=binding,
        grant=grant,
        research_plan_ref=grant["research_plan_ref"],
        runtime_ref=digest(canonical(controller.runtime_identity())),
        campaign_id=campaign_id,
        workspace=grant["workspace"],
        source_directory=grant["source_directory"],
    )


def require_root_identity(plan, root, campaign_id, owner_fd):
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import require_recursive_owner

    scope = require_recursive_owner()
    require_owner(trajectory_root(), owner_fd)
    expected = identity_definition(plan, campaign_id, scope.binding, scope.grant)
    if (
        root.resolve() != Path(expected["source_directory"]).resolve()
        or Path(expected["workspace"]).resolve() / ".quant_rsi" / "trajectory" != trajectory_root()
    ):
        raise ValueError("registered forecast root belongs to another source/workspace")
    record = decode(registration_path().read_bytes())
    if (
        record["kind"] != "ForecastRootRegistration"
        or record["state"] != "ROOT_ACTIVE"
        or record["handoff"] is not None
        or anchor_path(trajectory_root()).exists()
        or (trajectory_root().parent / "trajectory_initialization.json").exists()
        or record["identity"] != expected
    ):
        raise ValueError("registered forecast root is handed off or has a different identity")
    return expected


@contextmanager
def root_execution(plan, root, campaign_id, owner_fd, *, create):
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import recursive_scope

    with recursive_scope(plan=plan, source_directory=root, campaign_id=campaign_id) as scope:
        campaign_id = scope.grant["source_campaign_id"]
        if owner_fd is None:
            with locked_trajectory(bootstrap=True) as (_, owned):
                with _root_execution(plan, root, campaign_id, owned, scope, create=create):
                    yield campaign_id, owned
        else:
            with _root_execution(plan, root, campaign_id, owner_fd, scope, create=create):
                yield campaign_id, owner_fd


@contextmanager
def _root_execution(plan, root, campaign_id, owner_fd, scope, *, create):
    require_owner(trajectory_root(), owner_fd)
    require_disjoint_source(root, trajectory_root())
    identity = identity_definition(plan, campaign_id, scope.binding, scope.grant)
    if root.resolve() != Path(identity["source_directory"]).resolve():
        raise ValueError("registered root belongs to another source directory")
    path = registration_path()
    if not path.exists():
        if not create:
            raise ValueError("registered forecast root identity is missing")
        if any(
            p.exists()
            for p in (
                root / "manifest.json",
                root / "research_initialization.json",
                anchor_path(trajectory_root()),
                trajectory_root().parent / "trajectory_initialization.json",
            )
        ):
            raise ValueError("existing work cannot acquire a replacement root registration")
        atomic_write(
            path,
            canonical(
                dict(
                    kind="ForecastRootRegistration",
                    schema_version=3,
                    state="ROOT_ACTIVE",
                    identity=identity,
                    handoff=None,
                )
            ),
        )
    require_root_identity(plan, root, campaign_id, owner_fd)
    yield


def registered_creation(function):
    @wraps(function)
    def wrapped(plan, root, **kwargs):
        if plan.schema_version == 3 and kwargs.get("launch_context") is None:
            with root_execution(
                plan,
                Path(root).resolve(),
                kwargs.get("campaign_id"),
                kwargs.get("trajectory_owner_fd"),
                create=True,
            ) as (identity, fd):
                return function(
                    plan, root, **(kwargs | dict(campaign_id=identity, trajectory_owner_fd=fd))
                )
        return function(plan, root, **kwargs)

    return wrapped


def persisted_root_plan(root):
    root = Path(root)
    marker = root / "research_initialization.json"
    if marker.exists():
        saved = decode(marker.read_bytes())
        if saved.get("launch_context") is not None:
            return None
        plan = ResearchPlan.model_validate_json(canonical(saved["plan"]))
        return (plan, saved["campaign_id"]) if plan.schema_version == 3 else None
    path = root / "manifest.json"
    if path.exists():
        manifest = decode(path.read_bytes())
        if "launch_context_ref" in manifest:
            return None
        store = ArtifactStore(root / "artifacts")
        plan = ResearchPlan.model_validate_json(store.get(manifest["research_plan_ref"]))
        return (plan, manifest["campaign_id"]) if plan.schema_version == 3 else None
    return None


def registered_resume(function):
    @wraps(function)
    def wrapped(root, **kwargs):
        saved = persisted_root_plan(root)
        if saved is not None:
            plan, campaign_id = saved
            with root_execution(
                plan,
                Path(root).resolve(),
                campaign_id,
                kwargs.get("trajectory_owner_fd"),
                create=False,
            ) as (_, fd):
                return function(root, **(kwargs | dict(trajectory_owner_fd=fd)))
        return function(root, **kwargs)

    return wrapped


def bind_root_market_evidence(identity, store, data_ref):
    """Carry the original full market closure into the actual initialized source."""
    if identity["grant"].get("exposure_mode") != "historical_market_exploration":
        return
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import require_recursive_owner
    from quant_rsi.forecast.recursive.forecast_recursive_preparation import (
        store_market_projection,
        verify_market_registration,
    )

    scope = require_recursive_owner()
    if identity["grant"] != scope.grant:
        raise ValueError("source market closure belongs to another original grant")
    projection = verify_market_registration(scope.grant, scope._authority.store, data_ref=data_ref)
    store_market_projection(store, projection)


def verify_root_manifest(plan, store, manifest):
    if plan.schema_version == 3 and "launch_context_ref" not in manifest:
        identity = store.json(manifest["root_registration_identity_ref"])
        if identity != identity_definition(
            plan, manifest["campaign_id"], identity["binding"], identity["grant"]
        ):
            raise ValueError("forecast root manifest differs from its frozen grant/identity")
        from quant_rsi.forecast.recursive.forecast_recursive_preparation import (
            verify_market_registration,
        )

        if identity["grant"].get("exposure_mode") == "historical_market_exploration":
            verify_market_registration(identity["grant"], store, data_ref=manifest["data_ref"])


def effective_budget(plan, store, manifest):
    """A smaller root allowance never changes the complete paired development template."""
    value = plan.development.budget.model_dump(mode="json")
    if plan.schema_version == 3 and "launch_context_ref" not in manifest:
        verify_root_manifest(plan, store, manifest)
        grant = store.json(manifest["root_registration_identity_ref"])["grant"]
        allowance = grant["source_allowance"]
        for key in ("max_attempts", "max_worker_seconds"):
            if not 0 <= allowance[key] <= value[key]:
                raise ValueError("source allowance exceeds the immutable development template")
            value[key] = allowance[key]
    return value


def match_trajectory_input(plan, grant):
    if (
        plan.schema_version != 3
        or plan.source_directory.resolve() != Path(grant["source_directory"]).resolve()
        or plan.budget.model_dump(mode="json") != trajectory_units(grant["budget"])
        or plan.max_proposal_stages != grant["max_proposal_stages"]
        or plan.max_generations != grant["max_generations"]
    ):
        raise ValueError("trajectory input differs from its shared registered grant")


def begin_handoff(plan, owner_fd):
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import require_recursive_owner
    from quant_rsi.research import research
    from quant_rsi.research.trajectory import substantive
    from quant_rsi.research.trajectory_evidence import capture_research

    scope = require_recursive_owner()
    require_owner(trajectory_root(), owner_fd)
    match_trajectory_input(plan, scope.grant)
    require_disjoint_source(plan.source_directory, trajectory_root())
    marker = trajectory_root().parent / "trajectory_initialization.json"
    if not marker.exists() and trajectory_root().exists() and any(trajectory_root().iterdir()):
        raise ValueError("trajectory data exists without its initialization authority")
    path = registration_path()
    record = decode(path.read_bytes())
    source_plan = ResearchPlan.model_validate_json(canonical(scope.grant["research_plan"]))
    if record["identity"] != identity_definition(
        source_plan, scope.grant["source_campaign_id"], scope.binding, scope.grant
    ):
        raise ValueError("handoff differs from original root registration")
    inputs = substantive(plan.model_dump(mode="json"))
    if record["state"] == "HANDOFF":
        if record["handoff"]["input"] != inputs:
            raise ValueError("registered handoff input conflict")
        return record
    source = plan.source_directory.resolve()
    with controller.controller_lock(source):
        journal, store, manifest, source_plan, _ = research.open_research(source)
        try:
            require_root_identity(source_plan, source, manifest["campaign_id"], owner_fd)
            if store.json(manifest["root_registration_identity_ref"]) != record["identity"]:
                raise ValueError("handoff source lacks its original root registration")
        finally:
            journal.close()
    receipt_ref, receipt = capture_research(
        source, None, publish=None, initial=True, expected_checkpoint=plan.source_checkpoint_ref
    )
    record = record | dict(
        state="HANDOFF",
        handoff=dict(
            authority_id=uuid.uuid4().hex,
            input=inputs,
            checkpoint_ref=receipt["checkpoint_ref"],
            source_manifest_ref=receipt["manifest_ref"],
            source_receipt_ref=receipt_ref,
            costs=receipt["costs"],
        ),
    )
    atomic_write(path, canonical(record))
    return record


def verify_frozen_handoff(program, initial, store):
    from quant_rsi.research.trajectory import substantive

    record = store.json(program["root_registration_ref"])
    source_plan = ResearchPlan.model_validate_json(store.get(initial["research_plan_ref"]))
    manifest = store.json(initial["manifest_ref"])
    identity = record["identity"]
    grant = identity["grant"]
    from quant_rsi.forecast.recursive.forecast_recursive_campaign import policy_binding

    policy = policy_binding(grant)
    from quant_rsi.forecast.recursive.forecast_recursive_preparation import (
        verify_market_registration,
    )

    verify_market_registration(grant, store, data_ref=initial["data_ref"])
    if any(
        (key in program) != (key in policy) or program.get(key) != policy.get(key)
        for key in ("proposer_policy", "proposer_definition")
    ):
        raise ValueError("trajectory proposer selector differs from its immutable grant")
    match_trajectory_input(TrajectoryPlan.model_validate_json(canonical(program["input"])), grant)
    if (
        record["kind"] != "ForecastRootRegistration"
        or record["schema_version"] != 3
        or record["state"] != "HANDOFF"
        or program["authority_root"] != str(Path(grant["workspace"]) / ".quant_rsi" / "trajectory")
        or identity
        != identity_definition(source_plan, initial["campaign_id"], identity["binding"], grant)
        or program["recursive_campaign_binding"] != identity["binding"]
        or program["initial_capacity"] != trajectory_units(grant["source_allowance"])
        or store.json(manifest["root_registration_identity_ref"]) != identity
        or record["handoff"]
        != dict(
            authority_id=program["authority_id"],
            input=substantive(program["input"]),
            checkpoint_ref=initial["checkpoint_ref"],
            source_manifest_ref=initial["manifest_ref"],
            source_receipt_ref=program["initial_head"]["research_receipt_ref"],
            costs=initial["costs"],
        )
    ):
        raise ValueError("trajectory frozen root/shared grant/handoff mismatch")


def registered_trajectory_creation(function):
    @wraps(function)
    def wrapped(plan):
        if plan.schema_version != 3:
            return function(plan)
        from quant_rsi.forecast.recursive.forecast_recursive_campaign import recursive_scope

        with recursive_scope(source_directory=plan.source_directory):
            return function(plan)

    return wrapped


def registered_trajectory_resume(function):
    @wraps(function)
    def wrapped(*, steps=1):
        root = trajectory_root()
        if not anchor_path(root).is_file():
            return function(steps=steps)
        anchor = decode(anchor_path(root).read_bytes())
        program = ArtifactStore(root / "artifacts").json(anchor["program_ref"])
        if program.get("schema_version") != 3:
            return function(steps=steps)
        from quant_rsi.forecast.recursive.forecast_recursive_campaign import recursive_scope

        with recursive_scope(grant_ref=program["recursive_campaign_binding"]["grant_ref"]):
            return function(steps=steps)

    return wrapped
