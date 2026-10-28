"""Idempotent registry CAS; production authorization is fixed by its authority."""

import copy

from quant_rsi.evaluation.evidence import ref


def transition(state, request_id, binding, decision, *, fixture=False):
    """Pure transaction mutation, shared by production retain and isolated fixtures."""
    existing = state["decisions"].get(request_id)
    if existing is not None:
        if existing["binding_ref"] != ref(binding):
            raise ValueError("decision request ID was reused with another binding")
        return existing["decision"]
    registry = state["registry"]
    if (registry["package_ref"], registry["revision"]) != (
        binding["expected_package_ref"],
        binding["expected_revision"],
    ):
        raise ValueError("promotion registry package/revision CAS conflict")
    if decision not in {"retain", "promote"} or (decision == "promote" and not fixture):
        raise ValueError("production engineering authority cannot promote")
    result = {
        "request_id": request_id,
        "binding_ref": ref(binding),
        "decision": decision,
        "old_package_ref": registry["package_ref"],
        "old_revision": registry["revision"],
        "package_ref": binding["candidate_package_ref"]
        if decision == "promote"
        else registry["package_ref"],
        "revision": registry["revision"] + 1,
        "report_ref": binding["report_ref"],
        "plan_ref": binding["plan_ref"],
    }
    state["registry"] = {
        "package_ref": result["package_ref"],
        "revision": result["revision"],
        "last_report_ref": binding["report_ref"],
        "last_decision_ref": ref(result),
    }
    state["decisions"][request_id] = {"binding_ref": ref(binding), "decision": result}
    return result


class TestRegistry:
    """In-memory fixtures only. No workspace, anchor, SQL connection, or CLI access."""

    __test__ = False

    def __init__(self, package_ref):
        self.state = {"registry": {"package_ref": package_ref, "revision": 0}, "decisions": {}}

    def decide(self, request_id, binding, outcome):
        if outcome not in {"positive", "negative", "inconclusive", "invalid"}:
            raise ValueError("unknown fixture outcome")
        staged = copy.deepcopy(self.state)
        result = transition(
            staged,
            request_id,
            binding,
            "promote" if outcome == "positive" else "retain",
            fixture=True,
        )
        self.state = staged
        return result
