"""Fixed trusted computations in a resource-limited process, not a Python-code sandbox."""

import argparse
import resource
import signal
from pathlib import Path

from quant_rsi.artifacts import ArtifactStore, atomic_write, canonical, decode, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = decode(args.job.read_bytes())
    if not isinstance(job, dict) or set(job) != {"binding", "job_digest"}:
        raise ValueError("invalid worker job envelope")
    binding = job["binding"]
    if not isinstance(binding, dict) or set(binding) != {
        "ordinal",
        "proposal_index",
        "proposal",
        "manifest",
        "timeout_seconds",
    }:
        raise ValueError("invalid worker computation binding")
    computation_digest = digest(canonical(binding))
    if job["job_digest"] != computation_digest:
        raise ValueError("worker input digest mismatch")
    seconds = binding["timeout_seconds"]
    if type(seconds) is not int or not 1 <= seconds <= 300:
        raise ValueError("invalid worker deadline")
    # A default signal terminates even inside native code after the parent has died.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds + 1, seconds + 1))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    from quant_rsi.data import MarketData
    from quant_rsi.model import fit_predict
    from quant_rsi.schemas import DevelopmentPlan, FactorSpec

    store = ArtifactStore(args.job.resolve().parents[2] / "artifacts")
    manifest = binding["manifest"]
    plan = DevelopmentPlan.model_validate_json(store.get(manifest["plan_ref"]))
    ordinal = binding["proposal_index"]
    if (
        type(ordinal) is not int
        or type(binding["ordinal"]) is not int
        or binding["ordinal"] < 0
        or not 0 <= ordinal < len(plan.proposals)
        or binding["proposal"] != plan.proposals[ordinal].model_dump(mode="json")
        or seconds != plan.budget.worker_timeout_seconds
    ):
        raise ValueError("worker computation differs from the sealed plan")
    factor = FactorSpec.model_validate_json(canonical(binding["proposal"]["factor"]))
    data = MarketData.from_json(store.json(manifest["data_ref"]))
    result = fit_predict(data, plan, factor)
    result["job_digest"] = computation_digest
    atomic_write(args.job.parent / "result.json", canonical(result))


if __name__ == "__main__":
    main()
