from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import torch
import triton

from .agent import execute, materialize_language
from .records import Records
from .tasks import catalog, description, invocation, read_suite


def revision(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def compiler_revision(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "log", "-1", "--format=%H", "--",
                                    "CMakeLists.txt", "include", "lib", "python/intent", "tools"], text=True).strip()


def run_benchmark(arguments, task, program, language, result_path, gpu_lock, remaining_seconds) -> dict:
    started = time.monotonic()
    limit = min(arguments.suite["candidate_seconds"], remaining_seconds)
    if limit <= 0:
        return {"status": "phase_time_budget", "candidate_ms": None, "reference_ms": None, "ratio": None,
                "error": "stage budget ended before benchmark submission", "preparation_and_benchmark_seconds": 0.0}
    command = [sys.executable, "-B", "-m", "repro.agent_study.benchmark", "--reference", str(arguments.reference),
               "--compiler", str(arguments.compiler), "--task", task, "--program", str(program),
               "--language", language, "--result", str(result_path), "--gpu-lock", str(gpu_lock)]
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"status": "benchmark_timeout", "candidate_ms": None, "reference_ms": None, "ratio": None,
                    "error": f"candidate preparation/queue/execution exceeded the remaining {limit:.3f}-second budget",
                    "preparation_and_benchmark_seconds": time.monotonic() - started}
        if not result_path.exists():
            log.seek(0)
            return {"status": "benchmark_environment_failure", "candidate_ms": None, "reference_ms": None, "ratio": None,
                    "error": log.read()[-6000:], "preparation_and_benchmark_seconds": time.monotonic() - started}
    result = json.loads(result_path.read_text())
    result["preparation_and_benchmark_seconds"] = time.monotonic() - started
    return result


def trial(arguments, row, repeat, arm, records, runtime) -> None:
    if arguments.stop.is_set():
        return
    suite = arguments.suite
    task = next(task for task in suite["tasks"] if task["id"] == row["task"])
    if any(item["task"] == row["task"] and item["status"] == "awaiting_oracle_decision" for item in records.reference_issues):
        return
    identity = f"{row['task']}/{arm}/repeat-{repeat}"
    output_root = arguments.output / "programs" / identity
    if output_root.exists() and not arguments.resume and not (arm == "triton" and arguments.reuse_triton_from):
        raise ValueError(f"trial already has records: {identity}; do not silently replace an independent repetition")
    output_root.mkdir(parents=True, exist_ok=True)
    prior_rows = [item for item in records.rows if item["task"] == row["task"] and item["arm"] == arm and item["repeat"] == repeat]
    seed, seed_result = None, None
    for stage in ("generation", "optimization"):
        previous = [item for item in prior_rows if item["stage"] == stage]
        stop_path = output_root / f"{stage}-stop.json"
        if stage == "generation" and any(item["status"] == "pass" for item in previous):
            seed_result = next(item for item in previous if item["status"] == "pass")
            seed = (arguments.project / seed_result["program"]).parent
            if arm == "intent":
                seed = seed / "triton_seed"
            continue
        if arguments.stage is not None and stage != arguments.stage:
            continue
        if stage == "optimization" and seed is None:
            break
        if stop_path.exists():
            continue
        language = arm if stage == "generation" else "triton"
        directory = runtime / identity / stage
        directory.mkdir(parents=True)
        materials = materialize_language(arguments.project, arguments.triton_ref, directory / "materials", language)
        if seed is not None:
            for path in seed.glob("*.py"):
                shutil.copyfile(path, directory / path.name)
        task_text = description(arguments.reference, row)
        task_text += "\n\nFixed invocation (tensor values are not disclosed):\n" + json.dumps(row["invocation"], indent=2)
        task_text += "\n\nTolerance: " + json.dumps(suite["tolerances"][task["tolerance"]])
        task_text += "\n\nProfile note: " + row["reason"]
        (directory / "TASK.md").write_text(task_text + "\n")
        prefix = f"Implement {row['entry']} using {language}. Stage: {stage}. Read TASK.md and materials/. "
        prefix += "Write candidate.py with build(context) returning the task wrapper. "
        prefix += "The following is a generic vector-add API example, not a solution to your task:\n```python\n"
        prefix += (directory / "materials/vector_add.py").read_text() + "```\n"
        if stage == "optimization":
            prefix += "The existing files are your actual seed; optimize its full operator time. "
            prefix += "Measured seed: " + json.dumps({key: seed_result[key] for key in ("candidate_ms", "reference_ms", "ratio")})
        submissions = [item for item in previous if item["status"] != "agent_environment_failure"]
        feedback_keys = {"status", "candidate_ms", "reference_ms", "ratio", "failure_stage", "error", "reference_timing_note", "tuning"}
        history = [{"candidate": item["candidate"], "feedback": {key: value for key, value in item.items() if key in feedback_keys},
                    "programs": {path.name: path.read_text() for path in (arguments.project / item["program"]).parent.glob("*.py")}}
                   for item in submissions]
        if submissions:
            previous_directory = (arguments.project / submissions[-1]["program"]).parent
            for path in previous_directory.glob("*.py"):
                shutil.copyfile(path, directory / path.name)
        started = time.monotonic()
        spent = sum(item["agent"]["seconds"] + item.get("preparation_and_benchmark_seconds", 0) for item in previous)
        for candidate in range(len(submissions) + 1, suite[f"{stage}_candidates"] + 1):
            if arguments.stop.is_set():
                return
            remaining = suite["phase_seconds"] - spent - (time.monotonic() - started)
            if remaining <= 0:
                records.stop(stop_path, {"reason": "phase_time_budget", "submissions": candidate - 1,
                                         "limit_seconds": suite["phase_seconds"]})
                break
            prompt = prefix + f"\nSubmission {candidate} of at most 5.\n"
            if history:
                prompt += "Previous submissions and benchmark feedback from this repetition only:\n" + json.dumps(history)
            agent_result = execute(directory, suite, prompt, remaining_seconds=remaining,
                                   executable=arguments.codex, stop=arguments.stop)
            destination = output_root / f"{stage}-{candidate}"
            resumed = 0
            while destination.exists():
                resumed += 1
                destination = output_root / f"{stage}-{candidate}-resume-{resumed}"
            destination.mkdir()
            if agent_result["action"] == "stop" and stage == "optimization":
                (destination / "agent.json").write_text(json.dumps(agent_result, indent=2) + "\n")
                records.stop(stop_path, {"reason": "agent_stop", "submissions": candidate - 1, "agent": agent_result})
                break
            for path in directory.glob("*.py"):
                shutil.copyfile(path, destination / path.name)
            program = destination / "candidate.py"
            if agent_result["action"] == "unavailable":
                measured = {"status": agent_result["status"], "candidate_ms": None, "reference_ms": None,
                            "ratio": None, "error": agent_result["error"]}
            elif not program.exists():
                measured = {"status": "agent_program_error", "candidate_ms": None, "reference_ms": None,
                            "ratio": None, "error": "No candidate.py was submitted"}
            else:
                remaining = suite["phase_seconds"] - spent - (time.monotonic() - started)
                measured = run_benchmark(arguments, row["task"], program, language, destination / "measurement.json", arguments.gpu_lock or runtime / "gpu.lock", remaining)
            agent_record = {**agent_result, "language_materials": materials}
            (destination / "agent.json").write_text(json.dumps(agent_record, indent=2) + "\n")
            observation = {"task": row["task"], "case": f"upstream-profile-{task['input_index']}",
                           "arm": arm, "repeat": repeat, "stage": stage, "candidate": candidate,
                           "compiler_commit": arguments.compiler_commit,
                           "compiler_executable": str(arguments.compiler),
                           "program": str(program.relative_to(arguments.project)),
                           "agent": agent_record, **measured}
            records.add(observation)
            if measured["status"] in {"agent_environment_failure", "benchmark_environment_failure", "reference_failure"}:
                arguments.stop.set()
                raise RuntimeError(f"{identity}: {measured['status']}: {measured['error']}")
            if measured["status"] in {"agent_timeout", "phase_time_budget"}:
                records.stop(stop_path, {"reason": "phase_time_budget", "submissions": candidate})
                break
            history.append({"candidate": candidate, "programs": {path.name: path.read_text() for path in destination.glob("*.py")},
                            "feedback": {key: value for key, value in measured.items() if key != "traceback"}})
            if stage == "generation" and measured["status"] == "pass":
                seed = destination / "triton_seed" if arm == "intent" else destination
                seed_result = measured
                records.stop(stop_path, {"reason": "first_correct", "submissions": candidate})
                break
        else:
            records.stop(stop_path, {"reason": "candidate_budget", "submissions": suite[f"{stage}_candidates"]})
        with records.lock:
            records.publish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired TritonBench-T agent performance study")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--triton-ref", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--compiler-revision", help="Caller-declared build revision of an immutable compiler snapshot; verifies only that the live Python frontend matches")
    parser.add_argument("--codex", type=Path, required=True, help="Native Codex executable, not a shell/Node launcher")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=("triton", "intent"), default=("triton", "intent"))
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), action="append")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-lock", type=Path, help="Shared timing lock when scheduling independent study processes on one GPU")
    parser.add_argument("--stage", choices=("generation", "optimization"), help="Schedule one phase across the suite before the other")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, help="Independent report directory for a newly frozen experiment configuration")
    parser.add_argument("--reuse-triton-from", type=Path, help="Initialize a new compiler batch with unchanged direct Triton trials and original budgets")
    arguments = parser.parse_args()
    arguments.project = Path(__file__).resolve().parents[3]
    arguments.output = (arguments.output or arguments.project / "report/agent-tritonbench").resolve()
    if arguments.reuse_triton_from:
        arguments.reuse_triton_from = arguments.reuse_triton_from.resolve(strict=True)
    arguments.reference = arguments.reference.resolve()
    arguments.triton_ref = arguments.triton_ref.resolve()
    arguments.compiler = arguments.compiler.resolve(strict=True)
    arguments.codex = arguments.codex.resolve(strict=True)
    arguments.stop = threading.Event()
    arguments.compiler_commit = arguments.compiler_revision or compiler_revision(arguments.project)
    compiler_paths = ["python/intent"] if arguments.compiler_revision else ["CMakeLists.txt", "include", "lib", "python/intent", "tools"]
    if subprocess.check_output(["git", "-C", str(arguments.project), "status", "--porcelain", "--",
                                *compiler_paths], text=True).strip() and arguments.stage != "optimization":
        parser.error("commit compiler changes before agent trials; use the existing benchmark entry for compiler development")
    if arguments.compiler_revision:
        arguments.compiler_commit = subprocess.check_output(
            ["git", "-C", str(arguments.project), "rev-parse", "--verify", f"{arguments.compiler_revision}^{{commit}}"], text=True).strip()
        if arguments.stage != "optimization" and subprocess.check_output(["git", "-C", str(arguments.project), "diff", arguments.compiler_commit, "--", "python/intent"], text=True).strip():
            parser.error("the live Python frontend differs from the specified compiler snapshot")
    arguments.suite = read_suite()
    if not 1 <= arguments.workers <= 8:
        parser.error("use between one and eight concurrent preparation/agent workers")
    torch.set_num_threads(1)
    rows = catalog(arguments.reference, arguments.suite)
    by_id = {task["id"]: task for task in arguments.suite["tasks"]}
    for row in rows:
        if row["task"] in by_id:
            row.update(by_id[row["task"]])
            row["invocation"] = invocation(arguments.reference, row, by_id[row["task"]], arguments.suite, device="cpu").metadata()
    arguments.output.mkdir(parents=True, exist_ok=True)
    catalog_path = arguments.output / "tasks.json"
    if catalog_path.exists() and json.loads(catalog_path.read_text()) != rows:
        raise ValueError("task contracts changed after study results were started")
    catalog_path.write_text(json.dumps(rows, indent=2) + "\n")
    selected = arguments.tasks if arguments.tasks else list(by_id)
    if set(selected) - set(by_id):
        parser.error("--tasks must be drawn from the frozen 50-task suite")
    environment = {"compiler_commit": arguments.compiler_commit, "reference_commit": revision(arguments.reference),
                   "triton_ref_commit": revision(arguments.triton_ref), "torch": torch.__version__, "triton": triton.__version__,
                   "codex": subprocess.check_output([str(arguments.codex), "--version"], text=True).strip(),
                   "gpu": torch.cuda.get_device_name(0), "suite": arguments.suite,
                   "session_policy": "fresh ephemeral Codex call per submission; prior programs and feedback replayed only within a stage/repetition",
                   "instruction_policy": "native Codex instructions plus study developer instructions; common vector-add teaching example in each language",
                   "instructions": Path(__file__).with_name("instructions.md").read_text(),
                   "isolation": "agent editing: workspace-only filesystem, network off, user config/rules/skills/plugins/hooks/memory/multi-agent disabled; benchmark candidate API validation is not a hostile-Python sandbox",
                   "tuning_policy": "up to 16 evenly spaced configurations including endpoints after legality pruning; median CUDA Graph, 5 warmups / 30 samples"}
    if arguments.reuse_triton_from:
        environment["reuse_triton_from"] = str(arguments.reuse_triton_from.relative_to(arguments.project))
    environment_path = arguments.output / "environment.json"
    if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
        raise ValueError("recorded experiment environment changed; affected comparisons require explicit reconciliation")
    environment_path.write_text(json.dumps(environment, indent=2) + "\n")
    records = Records(arguments.output, arguments.suite)
    if arguments.reuse_triton_from and not records.rows:
        records.reuse_triton(arguments.reuse_triton_from, environment, rows)
    records.publish()
    runtime = Path(tempfile.mkdtemp(prefix="intent-agent-study-")).resolve()
    print(f"Selected {len(selected)} of 50 tasks; project results: {arguments.output}", flush=True)
    futures = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        for row in rows:
            if row["task"] not in selected:
                continue
            for repeat in arguments.repeat if arguments.repeat else range(arguments.suite["repetitions"]):
                for arm in arguments.arms:
                    futures.append(executor.submit(trial, arguments, row, repeat, arm, records, runtime))
        for future in as_completed(futures):
            try:
                future.result()
            except BaseException:
                arguments.stop.set()
                for pending in futures:
                    pending.cancel()
                raise


if __name__ == "__main__":
    main()
