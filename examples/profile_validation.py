"""Profile offline pytest runs without splitting a module's fixtures across processes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THREAD_ENV = dict(
    PYTHONHASHSEED="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)


def grouped_shards(selectors, count):
    groups = {}
    for selector in selectors:
        module = str(Path(selector.split("::", 1)[0]).resolve())
        groups.setdefault(module, []).append(selector)
    shards = [[] for _ in range(min(count, len(groups)))]
    for index, group in enumerate(groups.values()):
        shards[index % len(shards)].extend(group)
    return shards


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def source_state():
    paths = [*ROOT.glob("src/**/*.py"), *ROOT.glob("tests/**/*.py"), Path(__file__)]
    return {
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain"),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=ROOT)
        ).hexdigest(),
        "files": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))
        },
    }


def run_shard(output, index, selectors, env):
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--durations=20",
        "--basetemp",
        str(output / f"tmp-{index}"),
        "--junitxml",
        str(output / f"shard-{index}.xml"),
        *selectors,
    ]
    (output / f"shard-{index}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.monotonic()
    with (output / f"shard-{index}.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return dict(shard=index, exit_code=result.returncode, seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shards", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--cache-note", default="OS/package caches uncontrolled; fresh pytest basetemp"
    )
    parser.add_argument(
        "selectors", nargs="*", help="Test files/node IDs; defaults to all test modules"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    selectors = args.selectors or sorted(
        {
            str(path.relative_to(ROOT))
            for pattern in ("test_*.py", "*_test.py")
            for path in ROOT.glob(f"tests/**/{pattern}")
        }
    )
    shards = grouped_shards(selectors, args.shards)
    if not shards:
        parser.error("no test modules found")
    env = dict(os.environ) | THREAD_ENV
    env.pop("PYTHONPATH", None)  # Workers also clear it; use an installed checkout.
    loaded_package = subprocess.check_output(
        [sys.executable, "-c", "import quant_rsi; print(quant_rsi.__file__)"],
        cwd=ROOT,
        env=env,
        text=True,
    ).strip()
    if Path(loaded_package).resolve() != ROOT / "src/quant_rsi/__init__.py":
        parser.error(f"Python imports another checkout: {loaded_package}; install this checkout")
    before = source_state()
    metadata = dict(
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source=before,
        loaded_package=loaded_package,
        python=sys.executable,
        python_version=sys.version,
        platform=platform.platform(),
        cpu_count=os.cpu_count(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        load_average=os.getloadavg(),
        resource_limits={
            name: path.read_text().strip() if path.exists() else None
            for name, path in {
                "cpu_max": Path("/sys/fs/cgroup/cpu.max"),
                "memory_max": Path("/sys/fs/cgroup/memory.max"),
            }.items()
        },
        dependencies={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        environment=THREAD_ENV
        | {
            key: env.get(key)
            for key in ("PYTEST_ADDOPTS", "QUANT_RSI_UPSTREAM_CACHE", "QUANT_RSI_WORKSPACE")
        },
        cache_note=args.cache_note,
        shard_selectors=shards,
    )
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(shards)) as pool:
        futures = [pool.submit(run_shard, output, i, tests, env) for i, tests in enumerate(shards)]
        results = [future.result() for future in futures]
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    changed = source_state() != before
    exit_code = next((r["exit_code"] for r in results if r["exit_code"]), 0)
    if exit_code < 0:
        exit_code = 128 - exit_code
    summary = dict(
        elapsed_seconds=time.monotonic() - started,
        child_cpu_seconds=dict(
            user=usage_after.ru_utime - usage_before.ru_utime,
            system=usage_after.ru_stime - usage_before.ru_stime,
        ),
        shards=results,
        source_changed=changed,
        exit_code=exit_code or (1 if changed else 0),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(f"Logs, per-test JUnit timings, commands and environment: {output}")
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
