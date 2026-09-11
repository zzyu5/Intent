from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

import torch
import triton

from .agent import execute, materialize_language
from .tasks import catalog, description, invocation, read_suite


def revision(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def run_benchmark(arguments, task, program, language, result_path) -> dict:
    command = [sys.executable, "-B", "-m", "repro.agent_study.benchmark", "--reference", str(arguments.reference),
               "--compiler", str(arguments.compiler), "--task", task, "--program", str(program),
               "--language", language, "--result", str(result_path), "--gpu-lock", str(arguments.gpu_lock)]
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        try:
            process.wait(timeout=arguments.suite["benchmark_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"status": "benchmark_timeout", "error": "preparation/queue/execution exceeded the benchmark limit"}
        if not result_path.exists():
            log.seek(0)
            return {"status": "benchmark_environment_failure", "error": log.read()[-6000:]}
    return json.loads(result_path.read_text())


def trial(arguments, row, language) -> dict:
    directory = Path(tempfile.mkdtemp(prefix=f"{row['task']}-{language}-", dir=arguments.state_root / "candidates")).resolve()
    materials = materialize_language(arguments.project, arguments.triton_ref, directory / "materials", language)
    task_text = description(arguments.reference, row)
    task_text += "\n\nFixed invocation (tensor values are not disclosed):\n" + json.dumps(row["invocation"], indent=2)
    task_text += "\n\nTolerance: " + json.dumps(arguments.suite["tolerances"][row["tolerance"]])
    task_text += "\n\nProfile note: " + row["reason"]
    (directory / "TASK.md").write_text(task_text)
    agent = execute(directory, arguments.suite, f"Implement TASK.md using {language}. Submit one complete candidate.py.\n",
                    executable=arguments.codex, state_root=arguments.state_root, language=language, stop=arguments.stop)
    destination = arguments.output / row["task"] / language
    destination.mkdir(parents=True)
    agent["language_materials"] = materials
    (destination / "agent.json").write_text(json.dumps(agent, indent=2) + "\n")
    result_path = destination / "measurement.json"
    program = destination / "candidate.py"
    if agent["action"] != "submit":
        measured = {"status": agent["status"], "error": agent["error"]}
    elif not (directory / "candidate.py").exists():
        measured = {"status": "agent_program_error", "error": "No candidate.py was submitted"}
    else:
        shutil.copyfile(directory / "candidate.py", program)
        measured = run_benchmark(arguments, row["task"], program, language, result_path)
    result_path.write_text(json.dumps(measured, indent=2) + "\n")
    result = {"task": row["task"], "profile": row["input_index"], "language": language,
              "program": str(program.relative_to(arguments.output)) if program.exists() else "", **measured}
    print(json.dumps({key: result.get(key) for key in ("task", "language", "status", "candidate_ms", "reference_ms", "ratio")}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="One program submission per task/arm on the fixed TritonBench-T subset")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--triton-ref", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True, help="Native Codex executable")
    parser.add_argument("--state-root", type=Path, required=True, help="Dedicated external config.toml, provider.key and Codex state")
    parser.add_argument("--output", type=Path, required=True, help="New result directory; previous batches are never resumed")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=("triton", "intent"), default=("triton", "intent"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-lock", type=Path)
    arguments = parser.parse_args()
    arguments.project = Path(__file__).resolve().parents[3]
    arguments.suite, arguments.stop = read_suite(), threading.Event()
    for name in ("reference", "triton_ref", "compiler", "codex", "state_root"):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    if arguments.state_root.is_relative_to(arguments.project):
        parser.error("dedicated state/candidates must live outside the project")
    if not 1 <= arguments.workers <= 8:
        parser.error("use one to eight concurrent preparation workers")
    by_id = {task["id"]: task for task in arguments.suite["tasks"]}
    selected = arguments.tasks or list(by_id)
    if set(selected) - set(by_id):
        parser.error("--tasks must be drawn from the fixed 50-task suite")
    torch.set_num_threads(1)
    rows = [row for row in catalog(arguments.reference, arguments.suite) if row["task"] in selected]
    for row in rows:
        row.update(by_id[row["task"]])
        row["invocation"] = invocation(arguments.reference, row, by_id[row["task"]], arguments.suite, device="cpu").metadata()
    arguments.output = arguments.output.resolve()
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.state_root / "candidates").mkdir(exist_ok=True, mode=0o700)
    (arguments.state_root / "codex").mkdir(exist_ok=True, mode=0o700)
    arguments.gpu_lock = arguments.gpu_lock or arguments.state_root / "gpu.lock"
    environment = {"project_revision": revision(arguments.project), "reference_revision": revision(arguments.reference),
                   "triton_ref_revision": revision(arguments.triton_ref), "compiler": str(arguments.compiler),
                   "torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0),
                   "model": arguments.suite["model"], "reasoning_effort": arguments.suite["reasoning_effort"],
                   "tasks": rows, "submission_policy": "one complete program; documentation tools; no benchmark feedback",
                   "timing": "complete CUDA Graph operator; compilation/tuning/allocations outside timing",
                   "isolation": "dedicated Codex state/provider; workspace-only shell, no network; read-only public manual MCP; no reference/history/agents",
                   "instructions": Path(__file__).with_name("instructions.md").read_text()}
    (arguments.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    fields = ("task", "profile", "language", "status", "candidate_ms", "reference_ms", "ratio", "failure_stage", "error", "program")
    with (arguments.output / "results.csv").open("w", newline="") as output, ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        futures = [executor.submit(trial, arguments, row, language) for row in rows for language in arguments.arms]
        for future in as_completed(futures):
            try:
                writer.writerow(future.result())
                output.flush()
            except BaseException:
                arguments.stop.set()
                for pending in futures:
                    pending.cancel()
                raise


if __name__ == "__main__":
    main()
