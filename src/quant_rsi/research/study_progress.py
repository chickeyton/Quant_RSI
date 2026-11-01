"""Small, run-local progress records; no execution or recovery policy."""

import json
import sys
import time
from copy import deepcopy

from quant_rsi.core.artifacts import atomic_write


class StudyProgress:
    def __init__(self, root, llm_mode, quiet=False, previous_attempts=()):
        self.root, self.llm_mode, self.quiet = root, llm_mode, quiet
        self.previous_attempts = list(previous_attempts)
        self.started = time.monotonic()
        self.ended = None
        self.active = None
        self.stages = []
        self.usage_records = []
        self._save()

    def start(self, stage):
        if self.active is not None:
            raise ValueError("Finish the current progress stage before starting another")
        self.active = (stage, time.monotonic())
        self.stages.append({"stage": stage, "status": "running", "duration_seconds": None})
        self._print(f"running {stage}")
        self._save()

    def finish(self, stage, output=None):
        if self.active is None:
            return
        if self.active[0] != stage:
            raise ValueError("Progress completion does not match the running stage")
        self._close("completed")
        if output is not None:
            self.stages[-1]["output"] = output
        self._save()

    def usage(self, event):
        self.usage_records.append(
            {"mode": event["mode"]}
            | {
                key: value if type(value := event.get(key)) is int and value >= 0 else None
                for key in ("input_tokens", "cached_input_tokens", "output_tokens")
            }
        )
        self._save()

    def snapshot(self, status="running"):
        stages = deepcopy(self.stages)
        now = self.ended if self.ended is not None else time.monotonic()
        if self.active is not None:
            stages[-1]["duration_seconds"] = round(now - self.active[1], 6)
        counts = {}
        groups = {
            "llm_requests": lambda name: (
                self.llm_mode != "simulated"
                and (
                    name in {"initial_hypothesis", "revised_hypothesis", "extract_guidance"}
                    or (name.startswith("round_") and name.endswith("_hypothesis"))
                    or name.startswith(("news_search_", "news_review_"))
                )
            ),
            "search_requests": lambda name: name.startswith("news_search_"),
            "model_stages": lambda name: (
                name in {"first_prediction", "second_prediction"}
                or (name.startswith("round_") and name.endswith("_prediction"))
            ),
        }
        comparisons = {"qlib_comparison", "financial_comparison", "external_comparison"}
        if any(item["stage"] in comparisons for item in stages):
            groups["comparison_model_stages"] = lambda name: name in comparisons
        for group, matches in groups.items():
            selected = [
                item
                for item in stages
                if matches(item["stage"])
                and not item.get("reused_operations")
                and (item["stage"] != "extract_guidance" or item.get("llm_request"))
            ]
            counts[group + "_started"] = len(selected)
            counts[group + "_completed"] = sum(item["status"] == "completed" for item in selected)
        totals = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            values = [item[key] for item in self.usage_records if item[key] is not None]
            totals[key] = sum(values) if values else None
        reported = sum(
            item["input_tokens"] is not None and item["output_tokens"] is not None
            for item in self.usage_records
        )
        totals.update(
            reported_responses=reported,
            status="complete"
            if reported
            and reported == counts["llm_requests_started"]
            and all(item["cached_input_tokens"] is not None for item in self.usage_records)
            else "partial"
            if any(value is not None for value in totals.values())
            else "unavailable",
            notice="Provider-reported tokens only; cached input is included in input tokens. "
            "Missing usage is unknown, not zero. No currency cost is inferred.",
        )
        return {
            "previous_attempts": self.previous_attempts,
            "reused_operations": sum(len(s.get("reused_operations", [])) for s in stages),
            "status": status,
            "llm_mode": self.llm_mode,
            "total_elapsed_seconds": round(now - self.started, 6),
            "current_stage": self.active[0] if self.active else None,
            "stages": stages,
            "counts": counts,
            "token_usage": totals,
            "usage": deepcopy(self.usage_records),
        }

    def complete(self, snapshot=None):
        """Publish completion after artifacts save, reusing their timing snapshot if supplied."""
        if self.active is not None:
            raise ValueError("Cannot complete progress while a stage is running")
        self.ended = time.monotonic()
        result = self._save("completed", snapshot)
        self._print(f"completed in {result['total_elapsed_seconds']:.2f}s")
        return result

    def fail(self):
        if self.active is not None:
            self._close("failed")
        else:
            self._print("failed")
        self.ended = time.monotonic()
        return self._save("failed")

    def _close(self, status):
        stage, started = self.active
        elapsed = time.monotonic() - started
        self.stages[-1].update(status=status, duration_seconds=round(elapsed, 6))
        self.active = None
        self._print(f"{status} {stage} ({elapsed:.2f}s)")

    def _print(self, message):
        if not self.quiet:
            print("[quant-rsi] " + message, file=sys.stderr, flush=True)

    def _save(self, status="running", snapshot=None):
        result = self.snapshot(status) if snapshot is None else snapshot
        atomic_write(self.root / "runtime.json", (json.dumps(result, indent=2) + "\n").encode())
        return result
