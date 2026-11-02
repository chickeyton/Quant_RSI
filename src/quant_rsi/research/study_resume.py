"""Run-local, input-bound successful operation receipts; no pickle or job database."""

import fcntl
import json
import os
import platform
import shutil
from argparse import Namespace
from contextlib import contextmanager
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from quant_rsi.core.artifacts import atomic_write, canonical, digest
from quant_rsi.research.provider_process import STUDY_ROOT, provider_lock


def binding(configuration):
    source = Path(__file__).resolve().parents[1]
    files = list(source.rglob("*.py"))
    profiles = Path(configuration["factor_root"]).resolve()
    files += list(profiles.rglob("*.json"))
    files += list((source.parents[1] / "instruments").rglob("*.json"))
    if configuration.get("catalog"):
        files.append(Path(configuration["catalog"]).resolve())
    dependencies = ["numpy", "lightgbm", "pydantic"]
    if configuration.get("financials"):
        dependencies += ["pypdf", "fonttools", "openpyxl", "et-xmlfile"]
    if configuration.get("compare_qlib"):
        dependencies += ["pyqlib", "mlflow", "pandas"]
    return {
        "configuration": configuration,
        "files": {str(p): digest(p.read_bytes()) for p in sorted(set(files))},
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in dependencies},
    }


@contextmanager
def run_lock(root):
    with (root / ".study.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This study is already running") from None
        token = None
        try:
            probe = provider_lock(root)
            os.close(probe)
            token = STUDY_ROOT.set(root)
            yield
        finally:
            if token is not None:
                STUDY_ROOT.reset(token)
            fcntl.flock(stream, fcntl.LOCK_UN)


def resume_arguments(root, *, ui=False):
    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_bytes())
    # Derived fields are not command arguments.
    for name in (
        "model_configuration",
        "research_rounds",
        "fit_budget",
        "comparison_fit_budget",
        "publication_fit_budget",
        "generated_recipes",
    ):
        config.pop(name, None)
    for name in (
        "output",
        "catalog",
        "factor_root",
        "prices_dir",
        "related_prices_dir",
        "history_dir",
        "news_archive",
        "news_snapshot",
        "financials_dir",
        "external_drivers_dir",
        "external_drivers_cache_dir",
    ):
        if config.get(name) is not None:
            config[name] = Path(config[name])
    if config["output"].resolve() != root:
        raise ValueError("Resume must use the original study directory")
    config["ui"] = ui
    return Namespace(**config)


class StudyCache:
    def __init__(self, root, configuration, *, resume=False):
        self.root = root / "checkpoints"
        self.progress = None
        self.previous = []
        # Presentation switches do not change the experiment.
        configuration = {k: v for k, v in configuration.items() if k not in {"ui", "quiet", "json"}}
        identity = binding(configuration)
        manifest = self.root / "identity.json"
        if resume:
            if not manifest.exists():
                raise ValueError("This study predates resumable checkpoints; start a new study")
            if json.loads(manifest.read_bytes()) != identity:
                raise ValueError(
                    "Study code, dependencies or configuration changed; start a new study"
                )
            runtime = json.loads((root / "runtime.json").read_bytes())
            if runtime["status"] == "completed":
                raise ValueError("Completed studies are immutable; start a new study")
            attempts = root / "attempts"
            attempts.mkdir(exist_ok=True)
            archive = attempts / str(len(list(attempts.iterdir())) + 1)
            archive.mkdir()
            for name in ("runtime.json", "trace.jsonl", "failure.json"):
                if (root / name).exists():
                    shutil.copy2(root / name, archive / name)
            self.previous = runtime.get("previous_attempts", []) + [
                {
                    k: runtime[k]
                    for k in ("status", "counts", "token_usage", "total_elapsed_seconds")
                }
            ]
            for name in ("trace.jsonl", "failure.json"):
                (root / name).unlink(missing_ok=True)
        else:
            self.root.mkdir()
            atomic_write(manifest, canonical(identity))

    def call(self, name, inputs, operation):
        key = digest(canonical(inputs))
        path = self.root / (name + ".json")
        if path.exists():
            receipt = json.loads(path.read_bytes())
            if receipt["input_sha256"] != key:
                raise ValueError(f"Checkpoint inputs changed for {name}; start a new study")
            if receipt["output_sha256"] != digest(canonical(receipt["output"])):
                raise ValueError(f"Checkpoint output changed for {name}")
            if self.progress and self.progress.active:
                self.progress.stages[-1].setdefault("reused_operations", []).append(name)
            return deepcopy(receipt["output"])
        output = operation()
        atomic_write(
            path,
            canonical(
                {"input_sha256": key, "output_sha256": digest(canonical(output)), "output": output}
            ),
        )
        return output
