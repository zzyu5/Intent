from __future__ import annotations

import csv
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import statistics
import shutil
import threading


def _budget(rows: list[dict], terminal_agent: dict | None = None) -> dict:
    agents = [row["agent"] for row in rows]
    if terminal_agent is not None:
        agents.append(terminal_agent)
    usages = [usage for agent in agents for usage in agent["usage"]]
    return {"agent_seconds": sum(agent["seconds"] for agent in agents),
            "benchmark_seconds": sum(row.get("preparation_and_benchmark_seconds", 0) for row in rows),
            "input_tokens": sum(usage.get("input_tokens", 0) for usage in usages),
            "cached_input_tokens": sum(usage.get("cached_input_tokens", 0) for usage in usages),
            "output_tokens": sum(usage.get("output_tokens", 0) for usage in usages),
            "calls_without_reported_usage": sum(not agent["usage"] for agent in agents)}


class Records:
    def __init__(self, directory: Path, suite: dict):
        self.directory = directory
        self.suite = suite
        self.lock = threading.RLock()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "observations.jsonl"
        issues_path = directory / "reference-issues.json"
        self.reference_issues = json.loads(issues_path.read_text()) if issues_path.exists() else []
        with self._locked() as source:
            self._reload(source)

    @contextmanager
    def _locked(self):
        with self.lock, self.path.open("a+") as source:
            fcntl.flock(source, fcntl.LOCK_EX)
            yield source

    @contextmanager
    def _publication(self, name):
        destination = self.directory / name
        pending = destination.with_name(f".{name}.pending")
        with pending.open("w", newline="") as output:
            yield output
        pending.replace(destination)

    def _reload(self, source):
        source.seek(0)
        self.rows = [json.loads(line) for line in source]
        unresolved = {item["task"]: item for item in self.reference_issues
                      if item["status"] in {"awaiting_oracle_decision", "superseded_by_corrected_reference"}}
        for row in self.rows:
            if row["task"] in unresolved and row["status"] in {"pass", "numerical_failure"}:
                row.update(original_status=row["status"], status="reference_contract_failure",
                           failure_stage="reference_contract", error=unresolved[row["task"]]["issue"])
            recheck = Path(__file__).resolve().parents[3] / Path(row["program"]).parent / "reference-recheck.json"
            source_failure = row.get("failure_stage", "").startswith("source_")
            warmup_failure = (row.get("failure_stage") == "candidate_precompile"
                              and any(message in row.get("error", "") for message in (
                                  "compile_kernel() got multiple values for keyword argument 'warmup'",
                                  "Triton launch did not return a compiled kernel artifact")))
            if warmup_failure:
                recheck = recheck.with_name("evaluation-recheck.json")
                row.update(original_status=row["status"], status="evaluation_error")
            if (source_failure or warmup_failure) and recheck.exists():
                measured = json.loads(recheck.read_text())
                measured["evaluation_recheck_seconds"] = measured.pop("preparation_and_benchmark_seconds", 0)
                row.update(original_status=row["status"], evaluation_repaired=True, **measured)
                if measured["status"] == "pass":
                    row.pop("failure_stage", None)
                    row.pop("error", None)
                    row.pop("traceback", None)

    def add(self, row: dict) -> None:
        with self._locked() as source:
            source.write(json.dumps(row) + "\n")
            source.flush()
            self._publish(source)
        print(json.dumps({key: row[key] for key in ("task", "arm", "repeat", "stage", "candidate", "status", "candidate_ms", "reference_ms", "ratio")}), flush=True)

    def reuse_triton(self, directory: Path, environment: dict, catalog: list[dict]) -> None:
        source_environment = json.loads((directory / "environment.json").read_text())
        for name, value in environment.items():
            if name in {"compiler_commit", "reuse_triton_from"}:
                continue
            previous = source_environment[name]
            if name == "suite":
                value = {key: item for key, item in value.items() if key != "reference_corrections"}
                previous = {key: item for key, item in previous.items() if key != "reference_corrections"}
            if previous != value:
                raise ValueError(f"cannot reuse direct Triton with changed {name}")
        old_catalog = {row["task"]: row for row in json.loads((directory / "tasks.json").read_text())}
        reusable = {row["task"] for row in catalog if row["disposition"] == "selected"
                    and old_catalog[row["task"]] == row}
        source = Records(directory, source_environment["suite"])
        origin = str(directory.relative_to(Path(__file__).resolve().parents[3]))
        rows = [{**row, "reused_from": origin} for row in source.rows
                if row["arm"] == "triton" and row["task"] in reusable]
        with self._locked() as output:
            self._reload(output)
            if self.rows:
                raise ValueError("reference reuse initializes an empty experiment batch only")
            for task, repeat in {(row["task"], row["repeat"]) for row in rows}:
                relative = Path("programs") / task / "triton" / f"repeat-{repeat}"
                destination = self.directory / relative
                destination.mkdir(parents=True, exist_ok=True)
                for stage in ("generation", "optimization"):
                    stop = directory / relative / f"{stage}-stop.json"
                    if stop.exists():
                        shutil.copyfile(stop, destination / stop.name)
            for row in rows:
                output.write(json.dumps(row) + "\n")
            output.flush()
            self._publish(output)

    def stop(self, path: Path, result: dict) -> None:
        with self._locked() as source:
            pending = path.with_name(f".{path.name}.pending")
            pending.write_text(json.dumps(result, indent=2) + "\n")
            pending.replace(path)
            self._publish(source)

    def publish(self) -> None:
        with self._locked() as source:
            self._publish(source)

    def _publish(self, source) -> None:
        self._reload(source)
        fields = ("task", "case", "arm", "repeat", "stage", "candidate", "status", "candidate_ms", "reference_ms", "ratio", "failure_stage", "error", "reference_timing_note", "reference_correction", "original_status", "evaluation_repaired", "reused_from")
        with self._publication("candidates.csv") as output:
            writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        summaries, checkpoints = [], []
        for task in self.suite["tasks"]:
            for repeat in range(self.suite["repetitions"]):
                for arm in ("triton", "intent"):
                    relevant = [row for row in self.rows if row["task"] == task["id"]
                                and row["repeat"] == repeat and row["arm"] == arm]
                    generation_events = [row for row in relevant if row["stage"] == "generation"]
                    generated = [row for row in generation_events if row["status"] != "agent_environment_failure"]
                    correct = [row for row in generated if row["status"] == "pass"]
                    reference_conflict = any(row["status"] == "reference_contract_failure" for row in generated)
                    optimization_rows = [row for row in relevant if row["stage"] == "optimization"]
                    stop_path = self.directory / "programs" / task["id"] / arm / f"repeat-{repeat}" / "optimization-stop.json"
                    stop = json.loads(stop_path.read_text()) if stop_path.exists() else {}
                    stop_reason = stop.get("reason")
                    optimization_started = bool(optimization_rows or stop_reason)
                    optimized = correct + [row for row in optimization_rows if row["status"] == "pass"]
                    best = min(optimized, key=lambda row: row["candidate_ms"]) if optimized else None
                    seed = min(correct, key=lambda row: row["candidate"]) if correct else None
                    anchors = []
                    for paired_arm in ("triton", "intent"):
                        paired_seeds = [row for row in self.rows if row["task"] == task["id"] and row["repeat"] == repeat
                                        and row["arm"] == paired_arm and row["stage"] == "generation" and row["status"] == "pass"]
                        if paired_seeds:
                            anchor = min(paired_seeds, key=lambda row: row["candidate"])["reference_ms"]
                            if anchor is not None:
                                anchors.append(anchor)
                    target = statistics.median(anchors) if len(anchors) == 2 else None
                    first_target = {"generation": None, "optimization": None}
                    for stage, stage_rows in (("generation", generation_events), ("optimization", optimization_rows)):
                        if not stage_rows and not (stage == "optimization" and stop_reason):
                            continue
                        snapshots = [(row["candidate"] - int(row["status"] == "agent_environment_failure"), stage_rows[:index + 1], None,
                                      "interruption" if row["status"] == "agent_environment_failure" else "submission")
                                     for index, row in enumerate(stage_rows)]
                        if stage == "optimization" and seed:
                            snapshots.insert(0, (0, [], None, "seed"))
                            if stop.get("agent"):
                                snapshots.append((len(stage_rows), stage_rows, stop["agent"], "stop"))
                        for submission, prefix, terminal, event in snapshots:
                            eligible = ([seed] if stage == "optimization" and seed else []) + [row for row in prefix if row["status"] == "pass"]
                            winner = min(eligible, key=lambda row: row["candidate_ms"]) if eligible else None
                            reached = bool(winner and target is not None and winner["candidate_ms"] <= target)
                            if reached and first_target[stage] is None:
                                first_target[stage] = submission
                            stage_budget = _budget(prefix, terminal)
                            workflow_budget = _budget((generation_events if stage == "optimization" else []) + prefix, terminal)
                            checkpoints.append({"task": task["id"], "arm": arm, "repeat": repeat, "stage": stage,
                                                "submission": submission, "event": event,
                                                "best_correct_ms": winner["candidate_ms"] if winner else None,
                                                "best_program": winner["program"] if winner else None,
                                                "paired_reference_target_ms": target, "reached_reference": reached,
                                                **{f"stage_{key}": value for key, value in stage_budget.items()},
                                                **{f"workflow_{key}": value for key, value in workflow_budget.items()}})
                    summaries.append({"task": task["id"], "arm": arm, "repeat": repeat,
                                      "first_correct": bool(generated and generated[0]["status"] == "pass"),
                                      "budget_correct": bool(correct),
                                      "generation_evaluation_repaired": any(row.get("evaluation_repaired", False) for row in generated),
                                      "generation_status": "pass" if correct else "reference_contract_failure" if reference_conflict else generated[-1]["status"] if generated else "interrupted" if generation_events else "not_run",
                                      "seed_ms": seed["candidate_ms"] if seed else None,
                                      "static_intent_ms": seed["candidate_ms"] if seed and arm == "intent" else None,
                                      "optimization_status": stop_reason or ("in_progress" if optimization_rows else "not_run"),
                                      "optimized_ms": best["candidate_ms"] if best and optimization_started else None,
                                      "optimized_best_stage": best["stage"] if best and optimization_started else None,
                                      "seed_ratio": seed["ratio"] if seed else None,
                                      "optimized_ratio": best["ratio"] if best and optimization_started else None,
                                      "paired_reference_target_ms": target,
                                      "generation_first_reference_checkpoint": first_target["generation"],
                                      "optimization_first_reference_checkpoint": first_target["optimization"],
                                      **{f"generation_{key}": value for key, value in _budget(generation_events).items()},
                                      **{f"optimization_{key}": value for key, value in _budget(optimization_rows, stop.get("agent")).items()}})
        with self._publication("summary.csv") as output:
            writer = csv.DictWriter(output, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        if checkpoints:
            with self._publication("budget.csv") as output:
                writer = csv.DictWriter(output, fieldnames=list(checkpoints[0]))
                writer.writeheader()
                writer.writerows(checkpoints)
        metrics = {"ratio": "candidate_ms / PyTorch_reference_ms; smaller is better", "arms": {}}
        for arm in ("triton", "intent"):
            rows = [row for row in summaries if row["arm"] == arm]
            metrics["arms"][arm] = {
                "denominator": len(rows), "attempted": sum(row["generation_status"] != "not_run" for row in rows),
                "first_correct": sum(row["first_correct"] for row in rows),
                "budget_correct": sum(row["budget_correct"] for row in rows),
                "correct_and_no_slower_than_reference": sum(row["seed_ratio"] is not None and row["seed_ratio"] <= 1 for row in rows)}
            counts = metrics["arms"][arm]
            counts["not_run"] = counts["denominator"] - counts["attempted"]
            for name in ("first_correct", "budget_correct", "correct_and_no_slower_than_reference"):
                counts[f"{name}_rate"] = counts[name] / counts["denominator"]
        metrics["common_success_pairs"] = sum(
            all(next(row for row in summaries if row["task"] == task["id"] and row["repeat"] == repeat and row["arm"] == arm)["budget_correct"]
                for arm in ("triton", "intent")) for task in self.suite["tasks"] for repeat in range(self.suite["repetitions"]))
        paired = []
        indexed = {(row["task"], row["repeat"], row["arm"]): row for row in summaries}
        for task in self.suite["tasks"]:
            for repeat in range(self.suite["repetitions"]):
                direct, intent = (indexed[task["id"], repeat, arm] for arm in ("triton", "intent"))
                seeds = all(row["seed_ms"] is not None for row in (direct, intent))
                optimized = all(row["optimized_ms"] is not None for row in (direct, intent))
                paired.append({"task": task["id"], "repeat": repeat,
                               "direct_generation_status": direct["generation_status"], "intent_generation_status": intent["generation_status"],
                               "direct_seed_ms": direct["seed_ms"], "intent_seed_ms": intent["seed_ms"],
                               "intent_over_direct_seed": intent["seed_ms"] / direct["seed_ms"] if seeds else None,
                               "direct_optimization_status": direct["optimization_status"], "intent_optimization_status": intent["optimization_status"],
                               "direct_optimized_ms": direct["optimized_ms"], "intent_optimized_ms": intent["optimized_ms"],
                               "intent_over_direct_optimized": intent["optimized_ms"] / direct["optimized_ms"] if optimized else None,
                               "direct_optimization_speedup": direct["seed_ms"] / direct["optimized_ms"] if direct["optimized_ms"] is not None else None,
                               "intent_optimization_speedup": intent["seed_ms"] / intent["optimized_ms"] if intent["optimized_ms"] is not None else None})
        with self._publication("paired.csv") as output:
            writer = csv.DictWriter(output, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        metrics["paired_performance"] = {}
        for stage in ("seed", "optimized"):
            ratios = [row[f"intent_over_direct_{stage}"] for row in paired if row[f"intent_over_direct_{stage}"] is not None]
            metrics["paired_performance"][stage] = {
                "measured_pairs": len(ratios), "intent_faster_pairs": sum(ratio < 1 for ratio in ratios),
                "geomean_intent_over_direct": math.exp(statistics.mean(math.log(ratio) for ratio in ratios)) if ratios else None,
                "population": "pairs where both arms have a correct measured program; not a full-suite success rate"}
        with self._publication("metrics.json") as output:
            output.write(json.dumps(metrics, indent=2) + "\n")
        coverage = "; ".join(
            f"{arm}: {counts['attempted']}/{counts['denominator']} generation trials attempted, "
            f"{counts['first_correct']} first-correct, {counts['budget_correct']} budget-correct, {counts['not_run']} not run"
            for arm, counts in metrics["arms"].items()
        )
        lines = ["# TritonBench-T Agent Results", "", "Codex / gpt-5.6-luna / max. Fixed denominator: 50 tasks x 3 independent repetitions per arm.",
                 f"Current coverage: {coverage}. Not-run repetitions are pending, not observed failures. Fixed-denominator rates are incomplete until coverage is complete.",
                 "Times are median CUDA Graph operator milliseconds among correct repetitions; `-` means no correct measured program.",
                 "The PyTorch task implementation is a performance anchor, not an optimized Triton upper bound. Failures remain in the denominator.", "",
                 "Uncapturable references remain unchanged numerical oracles; candidate CUDA Graph times remain usable without a reference ratio. Reference-only evaluation repairs preserve raw observations and original agent programs. Canceled calls without a submission retain their cost but do not consume a submission slot.", "",
                 "Budgets include the work actually incurred, including retries after erroneous evaluator feedback. Repaired trials are flagged in summary.csv; their observed costs are not an estimate of an error-free workflow. The candidate time limit includes preparation and GPU queue time, not only device execution.", "",
                 "Optimization columns show the best correct program after optimization starts, including the unchanged seed when it remains best; they are not a claim of improvement. Stage and stopping status are in summary.csv.", "",
                 "paired.csv compares Intent/direct absolute operator times without requiring a PyTorch timing anchor. Paired performance is conditional on both arms succeeding; all failures remain in the separate fixed-denominator correctness results.", "",
                 "budget.csv records best-so-far absolute time and cumulative stage/full-workflow costs. Its common performance target is the median PyTorch seed-reference time of the two paired arms; no paired target is reported when either seed is missing. Cached input tokens are a subset of input tokens. Missing provider usage is disclosed, not estimated.", "",
                 "| Task | Direct correct / 3 | Intent correct / 3 | Direct seed ms | Intent static ms | Direct best after optimization ms | Intent-start best after optimization ms |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for task in self.suite["tasks"]:
            arms = {arm: [row for row in summaries if row["task"] == task["id"] and row["arm"] == arm] for arm in ("triton", "intent")}
            def median(arm, key):
                values = [row[key] for row in arms[arm] if row[key] is not None]
                return f"{statistics.median(values):.6f}" if values else "-"
            counts = [sum(row["budget_correct"] for row in arms[arm]) for arm in ("triton", "intent")]
            lines.append(f"| {task['id']} | {counts[0]} | {counts[1]} | {median('triton', 'seed_ms')} | {median('intent', 'seed_ms')} | {median('triton', 'optimized_ms')} | {median('intent', 'optimized_ms')} |")
        recheck_index = self.directory / "compiler-rechecks/index.json"
        reused = sorted({row["reused_from"] for row in self.rows if "reused_from" in row})
        if reused:
            lines += ["", "Unchanged direct Triton programs, measurements and their actual budgets were reused from: " + ", ".join(reused) + ". Tasks with changed reference contracts were not imported."]
        corrections = self.suite.get("reference_corrections", {})
        if corrections:
            lines += ["", "Project-local reference corrections: " + ", ".join(f"{task}: {correction}" for task, correction in corrections.items()) + ". Upstream files are unchanged; sub_gelu restores the missing 1 + in the exact GELU formula."]
        if self.reference_issues:
            lines += ["", "## Reference Issues", ""]
            for item in self.reference_issues:
                lines.append(f"- {item['task']} ({item['status']}): {item['issue']} {item['action']}")
        if recheck_index.exists():
            lines += ["", "## Compiler Rechecks", "",
                      "Human compiler development, reusing the original agent DSL. These measurements do not change agent first-submission correctness, budgets or optimization seeds above.", "",
                      "| Task | Before | Current ms | Reference ms | Current / reference | Status |",
                      "|---|---|---:|---:|---:|---|"]
            for item in json.loads(recheck_index.read_text()):
                measured = json.loads((self.directory / item["measurement"]).read_text())
                def number(key):
                    value = measured[key]
                    return f"{value:.6f}" if value is not None else "-"
                lines.append(f"| {item['task']} | {item['before']} | {number('candidate_ms')} | {number('reference_ms')} | {number('ratio')} | {measured['status']} |")
            lines += ["", "Compiler commits, original programs and ref comparisons are recorded in compiler-rechecks/index.json."]
        with self._publication("results.md") as output:
            output.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    import argparse
    from .tasks import read_suite

    parser = argparse.ArgumentParser(description="Publish existing agent-study benchmark observations")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[3] / "report/agent-tritonbench")
    arguments = parser.parse_args()
    environment = arguments.output / "environment.json"
    suite = json.loads(environment.read_text())["suite"] if environment.exists() else read_suite()
    Records(arguments.output, suite).publish()
