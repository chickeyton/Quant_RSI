"""Managed bootstrap must reject missing parent authority before publication."""

from pathlib import Path

import pytest

from quant_rsi.core import controller
from quant_rsi.core.artifacts import canonical, digest
from quant_rsi.research import research
from quant_rsi.research.researcher import ResearchPlan


def plan():
    return ResearchPlan.model_validate_json(
        (Path(__file__).parents[1] / "examples/research_development.json").read_bytes()
    )


def files(root):
    return {
        str(p.relative_to(root)): digest(p.read_bytes()) for p in root.rglob("*") if p.is_file()
    }


def context():
    return dict(
        package_campaign_id="unowned", arm="parent", grant_ref="a" * 64, state_policy="reset"
    )


@pytest.mark.parametrize("entry", ["create", "resume", "finish"])
def test_managed_bootstrap_requires_parent_before_writes(tmp_path, entry):
    root = tmp_path / "child"
    artifact = {"kind": "must_not_publish"}
    arguments = dict(
        campaign_id="child",
        launch_context=context(),
        package_artifacts={digest(canonical(artifact)): artifact},
    )
    if entry != "create":
        root.mkdir()
        saved = dict(
            kind="research_initialization_v1",
            plan=plan().model_dump(mode="json"),
            runtime=controller.runtime_identity(),
            bound_view=None,
            **arguments,
        )
        (root / "research_initialization.json").write_bytes(canonical(saved))
    if entry == "finish":
        with controller.controller_lock(root):
            before = files(root)
            with pytest.raises(ValueError, match="managed research arm requires package resume"):
                research.finish_initialization(root)
            assert files(root) == before
    else:
        before = files(root)
        with pytest.raises(ValueError, match="managed research arm requires package resume"):
            if entry == "create":
                research.create_research(plan(), root, initialize_only=True, **arguments)
            else:
                research.resume_research(root)
        assert files(root) == before
        if entry == "create":
            assert not root.exists()


def test_arbitrary_fd_cannot_authorize_managed_creation(tmp_path):
    outer = tmp_path / "package"
    outer.mkdir()
    (outer / ".controller.lock").touch()
    root = outer / "arms" / "parent"
    with (tmp_path / "unrelated.lock").open("w+") as owner:
        with pytest.raises(ValueError, match="owning controller lock"):
            research.create_research(
                plan(),
                root,
                campaign_id="child",
                launch_context=context(),
                owner_fd=owner.fileno(),
                initialize_only=True,
            )
    assert not root.exists()
