"""Explicit, fully reserved recursive forecast grants in the shared workspace authority.

This module grants capacity and verifies its original owner descriptor. Actual source
and descendant charges remain in the research/trajectory journals, never a new ledger.
"""

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from quant_rsi import forecast_campaign as campaign
from quant_rsi.artifacts import canonical, digest
from quant_rsi.contracts_base import StrictModel
from quant_rsi.market_contracts import Hash
from quant_rsi.researcher import ResearchPlan

EXPOSURE = {
    "prior_exposure": "engineering_fixture",
    "permitted_evidence": "engineering_fixture_development",
    "protected_evidence_access": "none",
    "predictive_validity_established": False,
}
_ACTIVE = ContextVar("forecast_recursive_campaign_owner", default=None)


class _Allowance(StrictModel):
    max_attempts: int = Field(strict=True, ge=1)
    max_worker_seconds: int = Field(strict=True, ge=1)


def _source_path(value):
    path = Path(value)
    if not path.is_absolute() or str(path.resolve()) != str(value):
        raise ValueError("recursive source_directory must be an exact canonical absolute path")
    return str(path)


class _Plan(StrictModel):
    kind: Literal["ForecastRecursiveCampaign"]
    schema_version: Literal[3]
    campaign_id: str
    source_directory: str
    source_campaign_id: str
    research_plan: ResearchPlan
    source_allowance: _Allowance
    budget: _Allowance
    max_proposal_stages: Literal[2]
    max_generations: Literal[2]

    @model_validator(mode="after")
    def coherent(self):
        campaign._id(self.campaign_id)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", self.source_campaign_id):
            raise ValueError("invalid recursive source campaign ID")
        _source_path(self.source_directory)
        plan = self.research_plan
        if plan.schema_version != 3 or plan.package_lineage is not None:
            raise ValueError("recursive grant requires an initial schema3 ResearchPlan")
        arm = plan.development.budget
        source = self.source_allowance
        if (
            source.max_attempts > arm.max_attempts
            or source.max_worker_seconds > arm.max_worker_seconds
            or source.max_worker_seconds < arm.worker_timeout_seconds
        ):
            raise ValueError("source allowance must fit the full development template")
        if self.budget.model_dump() != {
            "max_attempts": source.max_attempts + 2 * self.max_proposal_stages * arm.max_attempts,
            "max_worker_seconds": (
                source.max_worker_seconds + 2 * self.max_proposal_stages * arm.max_worker_seconds
            ),
        }:
            raise ValueError("recursive budget must reserve source and both complete reset pairs")
        return self


class _FrontierPlan(_Plan):
    schema_version: Literal[4]
    proposer_policy: Literal["terminal_frontier_first_divergence_v1"]
    proposer_definition: dict
    exposure_mode: Literal["engineering_fixture", "historical_market_exploration"]
    market_evidence_ref: Hash | None = None

    @model_validator(mode="after")
    def explicit_policy_and_exposure(self):
        from quant_rsi.forecast_frontier_policy import _require_descriptor

        _require_descriptor(self.model_dump(mode="json"))
        if (self.exposure_mode == "historical_market_exploration") != (
            self.market_evidence_ref is not None
        ):
            raise ValueError("market evidence is required only for historical market exposure")
        return self


def exposure_definition(plan):
    if plan["schema_version"] == 3 or plan["exposure_mode"] == "engineering_fixture":
        return dict(EXPOSURE)
    return {
        "prior_exposure": "previously_exposed_historical_exploration",
        "permitted_evidence": "historical_market_development",
        "protected_evidence_access": "none",
        "predictive_validity_established": False,
    }


def policy_binding(grant):
    """Only the outer grant may introduce the immutable new policy selector."""
    from quant_rsi.forecast_frontier_policy import _require_descriptor

    fields = ("proposer_policy", "proposer_definition")
    if grant.get("schema_version") == 3:
        if any(key in grant for key in (*fields, "exposure_mode", "market_evidence_ref")):
            raise ValueError("legacy recursive grant cannot carry frontier or market selectors")
        return {}
    if grant.get("schema_version") != 4:
        raise ValueError("unsupported recursive grant version")
    _require_descriptor(grant)
    return {key: grant[key] for key in fields}


@dataclass(frozen=True)
class RecursiveScope:
    binding: dict
    grant: dict
    owner_fd: int
    _authority: object
    _grant_ref: str


def _normalize(plan):
    if isinstance(plan, _Plan):
        plan = plan.model_dump(mode="json")
    model = _FrontierPlan if plan.get("schema_version") == 4 else _Plan
    return model.model_validate_json(canonical(plan)).model_dump(mode="json")


def _grant(authority, plan):
    return {
        **plan,
        "kind": "ForecastRecursiveCampaignGrant",
        "authority_ref": authority.authority_ref,
        "workspace": authority.identity["workspace"],
        "research_plan_ref": digest(canonical(plan["research_plan"])),
        "exposure_ref": digest(canonical(exposure_definition(plan))),
    }


def validate_registration(authority, payload):
    """Pure validation used by the shared registry; never create missing CAS members."""
    if set(payload) != {"campaign_id", "grant_ref"}:
        raise ValueError("invalid recursive campaign registration event")
    grant = authority.store.json(payload["grant_ref"])
    plan = {
        key: value
        for key, value in grant.items()
        if key not in {"authority_ref", "workspace", "research_plan_ref", "exposure_ref"}
    }
    if plan.get("kind") != "ForecastRecursiveCampaignGrant":
        raise ValueError("invalid recursive grant kind")
    plan["kind"] = "ForecastRecursiveCampaign"
    normalized = _normalize(plan)
    if (
        grant != _grant(authority, normalized)
        or payload["campaign_id"] != normalized["campaign_id"]
        or authority.store.json(grant["research_plan_ref"]) != normalized["research_plan"]
        or authority.store.json(grant["exposure_ref"]) != exposure_definition(normalized)
    ):
        raise ValueError("recursive grant, plan or exposure binding differs")
    from quant_rsi.forecast_recursive_preparation import verify_market_registration

    verify_market_registration(grant, authority.store)
    return normalized, grant


def _registered(authority):
    campaigns, _, _ = campaign._registry(authority)
    return {
        key: (ref, authority.store.json(ref))
        for key, (ref, plan) in campaigns.items()
        if plan["kind"] == "ForecastRecursiveCampaign"
    }


def _binding(authority, ref, grant):
    return {
        "authority_ref": authority.authority_ref,
        "grant_ref": ref,
        "workspace": authority.identity["workspace"],
        **{
            key: grant[key]
            for key in (
                "campaign_id",
                "source_directory",
                "source_campaign_id",
                "research_plan_ref",
                "exposure_ref",
            )
        },
    }


def _scope(authority, ref, grant):
    return RecursiveScope(
        _binding(authority, ref, grant), grant, authority.owner_fd, authority, ref
    )


def _report(scope):
    return {
        "kind": "ForecastRecursiveCampaignReport",
        "schema_version": scope.grant["schema_version"],
        "binding": scope.binding,
        "grant": scope.grant,
        "reserved": scope.grant["budget"],
        "exposure": exposure_definition(scope.grant),
        "actual_costs": "derive_from_original_research_and_trajectory_journals",
    }


def init_recursive_campaign(plan, *, market_preparation=None):
    """Reserve one immutable source and two full reset pairs; do not start research."""
    normalized = _normalize(plan)
    projection = None
    if normalized.get("exposure_mode") == "historical_market_exploration":
        from quant_rsi.evidence import authority_root
        from quant_rsi.forecast_recursive_preparation import (
            market_research_plan,
            read_prepared_projection,
            verify_actual_export,
        )

        if market_preparation is None:
            raise ValueError("market registration requires its original completed preparation")
        prepared, projection = read_prepared_projection(market_preparation)
        if (
            prepared["meta"]["workspace"] != str(authority_root().parent)
            or projection["market_evidence_ref"] != normalized["market_evidence_ref"]
            or market_research_plan(projection).model_dump(mode="json")
            != normalized["research_plan"]
        ):
            raise ValueError("market registration differs from its original preparation/profile")
        verify_actual_export(projection)
    elif market_preparation is not None:
        raise ValueError("engineering registration cannot adopt market preparation")
    with campaign._authority(initialize=True) as authority:
        campaigns, registrations, acks = campaign._registry(authority)
        campaign._children(authority, registrations, acks)
        name = normalized["campaign_id"]
        if name in campaigns:
            ref, existing = campaigns[name]
            if existing != normalized:
                raise ValueError("campaign ID already binds another immutable plan")
            return _report(_scope(authority, ref, authority.store.json(ref)))
        sources = _registered(authority)
        if normalized["source_campaign_id"] in registrations or any(
            grant["source_campaign_id"] == normalized["source_campaign_id"]
            or grant["source_directory"] == normalized["source_directory"]
            for _, grant in sources.values()
        ):
            raise ValueError("recursive source ID or directory is already reserved")
        source = Path(normalized["source_directory"])
        if source.exists() or source.is_symlink():
            raise ValueError("recursive source must be registered before it exists")
        base = authority.root.parent
        if source == base or source in base.parents or base in source.parents:
            raise ValueError("recursive source overlaps the shared authority directory")
        grant = _grant(authority, normalized)
        authority.store.put_json(normalized["research_plan"])
        authority.store.put_json(exposure_definition(normalized))
        if projection is not None:
            from quant_rsi.forecast_recursive_preparation import store_market_projection

            store_market_projection(authority.store, projection)
        ref = authority.store.put_json(grant)
        with authority.journal.connection:
            authority.journal.event(
                None,
                "recursive_campaign_registered",
                {"campaign_id": name, "grant_ref": ref},
            )
        return _report(_scope(authority, ref, grant))


def require_recursive_owner():
    """Revalidate active controller scope; workers inherit the FD only for exclusion."""
    scope = _ACTIVE.get()
    if scope is None:
        raise ValueError("recursive work requires its active original authority scope")
    campaign._require_owner(scope._authority)
    if scope.owner_fd != scope._authority.owner_fd:
        raise ValueError("recursive scope owner descriptor differs")
    record = _registered(scope._authority).get(scope.binding["campaign_id"])
    if (
        record is None
        or record[0] != scope._grant_ref
        or record[1] != scope.grant
        or _binding(scope._authority, *record) != scope.binding
    ):
        raise ValueError("active recursive grant or binding changed")
    return scope


def _matches(scope, *, plan, source_directory, campaign_id, grant_ref):
    if (
        (
            source_directory is not None
            and _source_path(source_directory) != scope.grant["source_directory"]
        )
        or (campaign_id is not None and campaign_id != scope.grant["source_campaign_id"])
        or (grant_ref is not None and grant_ref != scope.binding["grant_ref"])
    ):
        raise ValueError("recursive identity selectors disagree")
    if plan is not None:
        if isinstance(plan, ResearchPlan):
            plan = plan.model_dump(mode="json")
        normalized = ResearchPlan.model_validate_json(canonical(plan)).model_dump(mode="json")
        if normalized != scope.grant["research_plan"]:
            raise ValueError("recursive ResearchPlan differs from the frozen full template")


@contextmanager
def recursive_scope(*, plan=None, source_directory=None, campaign_id=None, grant_ref=None):
    """Select by the C2 source campaign ID, never by the outer grant campaign ID."""
    active = _ACTIVE.get()
    if active is not None:
        scope = require_recursive_owner()
        _matches(
            scope,
            plan=plan,
            source_directory=source_directory,
            campaign_id=campaign_id,
            grant_ref=grant_ref,
        )
        yield scope
        return
    if source_directory is None and campaign_id is None and grant_ref is None:
        raise ValueError("recursive scope requires an explicit registered identity")
    with campaign._authority() as authority:
        if authority is None:
            raise ValueError("recursive campaign authority is not initialized")
        matches = []
        for ref, grant in _registered(authority).values():
            if (
                (
                    source_directory is None
                    or _source_path(source_directory) == grant["source_directory"]
                )
                and (campaign_id is None or campaign_id == grant["source_campaign_id"])
                and (grant_ref is None or grant_ref == ref)
            ):
                matches.append((ref, grant))
        if len(matches) != 1:
            raise ValueError("unknown or conflicting recursive campaign identity")
        scope = _scope(authority, *matches[0])
        _matches(
            scope,
            plan=plan,
            source_directory=source_directory,
            campaign_id=campaign_id,
            grant_ref=grant_ref,
        )
        token = _ACTIVE.set(scope)
        try:
            require_recursive_owner()
            yield scope
        finally:
            _ACTIVE.reset(token)
