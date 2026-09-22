from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import inspect
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

import torch
import triton

from experiments import PROJECT_ROOT
from .agent import execute, materialize_language
from .tasks import SUITE_PATH, catalog, description, invocation, read_suite, reference, return_contract


def revision(directory: Path) -> str:
    return subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()


def run_benchmark(arguments, task, program, language, result_path, *, artifacts=None, target="triton") -> dict:
    phase_read, phase_write = os.pipe()
    command = [sys.executable, "-B", "-m", "experiments.agent_tritonbench.benchmark", "--reference", str(arguments.reference),
               "--compiler", str(arguments.compiler), "--task", task, "--program", str(program),
               "--language", language, "--result", str(result_path), "--gpu-lock", str(arguments.gpu_lock),
               "--suite", str(arguments.suite_path), "--phase-fd", str(phase_write), "--target", target]
    if artifacts is not None:
        command.extend(("--artifacts", str(artifacts)))
    with os.fdopen(phase_read, "rb", buffering=0) as phases, tempfile.TemporaryFile(mode="w+") as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True,
                                       pass_fds=(phase_write,))
        finally:
            os.close(phase_write)
        started = time.monotonic()
        queued_seconds, queued_at = 0.0, None
        stage, pending = "worker_startup", b""
        try:
            while process.poll() is None:
                if select.select([phases], [], [], 0.1)[0]:
                    pending += phases.read(65536)
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        event = json.loads(line)
                        stage = event["stage"]
                        if stage == "gpu_queue":
                            queued_at = event["time"]
                        elif queued_at is not None:
                            queued_seconds += event["time"] - queued_at
                            queued_at = None
                now = time.monotonic()
                active_seconds = now - started - queued_seconds
                if queued_at is not None:
                    active_seconds -= now - queued_at
                if active_seconds > arguments.suite["benchmark_seconds"]:
                    log.seek(0)
                    diagnostic = log.read()[-6000:]
                    return {"status": "benchmark_timeout", "failure_stage": stage,
                            "error": "preparation/execution exceeded the benchmark limit, excluding GPU queue time"
                                     + ("\n" + diagnostic if diagnostic else "")}
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if not result_path.exists():
            log.seek(0)
            return {"status": "benchmark_environment_failure", "error": log.read()[-6000:]}
    return json.loads(result_path.read_text())


def finish_trial(arguments, result, measured) -> dict:
    destination = arguments.output / result["task"] / result["language"]
    (destination / "measurement.json").write_text(json.dumps(measured, indent=2) + "\n")
    result.update(measured)
    print(json.dumps({key: result.get(key) for key in ("task", "language", "status", "candidate_ms", "reference_ms", "ratio")}), flush=True)
    return result


def generate_trial(arguments, row, language) -> dict:
    directory = Path(tempfile.mkdtemp(prefix=f"{row['task']}-{language}-", dir=arguments.state_root / "candidates")).resolve()
    materials = materialize_language(arguments.project, arguments.triton_ref, directory / "materials", language)
    task_text = description(arguments.reference, row)
    task_text += "\n\nCallable parameter signature (names, order and defaults):\n" + row["reference_signature"]
    task_text += "\n\nFixed profile parameters (reference defaults included; tensor values are not disclosed):\n" + json.dumps(row["invocation"], indent=2)
    task_text += "\nParameters at their defaults may be omitted by the caller. Preserve optional parameter defaults in the returned callable."
    task_text += "\n\nExpected return structure, shape and dtype:\n" + json.dumps(row["return_contract"], indent=2)
    task_text += "\n\nTolerance: " + json.dumps(arguments.suite["tolerances"][row["tolerance"]])
    task_text += "\n\nTiming: " + row["timing"]
    task_text += "\n\nProfile note: " + row["reason"]
    (directory / "TASK.md").write_text(task_text)
    prompt = f"Implement TASK.md correctly and efficiently using {language}. Submit one complete candidate.py.\n"
    if language == "intent":
        prompt += (
            "Choose a Triton-style algorithm, including independent logical work and explicit kernel stages, "
            "then express it with Intent domains. Read the public language and kernel/host contracts before choosing "
            "the organization; the compiler chooses physical mapping within the declared kernels. "
            "Use @intent.kernel and context.compile(); do not import or call Triton.\n"
        )
    agent = execute(directory, arguments.suite, prompt,
                    executable=arguments.codex, state_root=arguments.state_root, language=language, stop=arguments.stop)
    destination = arguments.output / row["task"] / language
    destination.mkdir(parents=True)
    agent["language_materials"] = materials
    (destination / "agent.json").write_text(json.dumps(agent, indent=2) + "\n")
    program = destination / "candidate.py"
    result = {"task": row["task"], "profile": row["input_index"], "language": language, "program": ""}
    if agent["action"] != "submit":
        measured = {"status": agent["status"], "error": agent["error"]}
    elif not (directory / "candidate.py").exists():
        measured = {"status": "agent_program_error", "error": "No candidate.py was submitted"}
    else:
        shutil.copyfile(directory / "candidate.py", program)
        return {**result, "program": str(program.relative_to(arguments.output)), "status": "submitted"}
    return finish_trial(arguments, result, measured)


def evaluate_trial(arguments, result) -> dict:
    program = arguments.output / result["program"]
    measured = run_benchmark(arguments, result["task"], program, result["language"], program.parent / "measurement.json")
    return finish_trial(arguments, result, measured)


def main() -> None:
    parser = argparse.ArgumentParser(description="One program submission per task/arm on the fixed TritonBench-T subset")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--triton-ref", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--codex", type=Path, required=True, help="Native Codex executable")
    parser.add_argument("--state-root", type=Path, required=True, help="Dedicated external config.toml, provider.key and Codex state")
    parser.add_argument("--output", type=Path, required=True, help="New result directory; previous batches are never resumed")
    parser.add_argument("--suite", dest="suite_path", type=Path, default=SUITE_PATH,
                        help="Fixed task and generation configuration")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--arms", nargs="+", choices=("triton", "intent"), default=("triton", "intent"))
    parser.add_argument("--workers", type=int, default=8, help="Concurrent isolated code-generation workers")
    parser.add_argument("--benchmark-workers", type=int, default=4,
                        help="Concurrent compiler/JIT processes; GPU execution uses the shared lock")
    parser.add_argument("--gpu-lock", type=Path)
    arguments = parser.parse_args()
    arguments.project = PROJECT_ROOT
    arguments.suite_path = arguments.suite_path.resolve(strict=True)
    arguments.suite, arguments.stop = read_suite(arguments.suite_path), threading.Event()
    for name in ("reference", "triton_ref", "compiler", "codex", "state_root"):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    if arguments.state_root.is_relative_to(arguments.project):
        parser.error("dedicated state/candidates must live outside the project")
    if not 1 <= arguments.workers <= 8:
        parser.error("use one to eight concurrent generation workers")
    if not 1 <= arguments.benchmark_workers <= 8:
        parser.error("use one to eight concurrent benchmark preparation workers")
    by_id = {task["id"]: task for task in arguments.suite["tasks"]}
    selected = arguments.tasks or list(by_id)
    if set(selected) - set(by_id):
        parser.error("--tasks must be drawn from the selected fixed suite")
    torch.set_num_threads(1)
    rows = [row for row in catalog(arguments.reference, arguments.suite) if row["task"] in selected]
    for row in rows:
        row.update(by_id[row["task"]])
        row["timing"] = by_id[row["task"]].get("timing", arguments.suite["timing"])
        material_inputs = invocation(arguments.reference, row, by_id[row["task"]], arguments.suite, device="cpu")
        row["invocation"] = material_inputs.metadata()
        source = reference(arguments.reference, row)
        row["reference_signature"] = row["entry"] + str(inspect.signature(source))
        with torch.no_grad():
            row["return_contract"] = return_contract(material_inputs.call(source))
    arguments.output = arguments.output.resolve()
    arguments.output.mkdir(parents=True, exist_ok=False)
    (arguments.state_root / "candidates").mkdir(exist_ok=True, mode=0o700)
    (arguments.state_root / "codex").mkdir(exist_ok=True, mode=0o700)
    arguments.gpu_lock = arguments.gpu_lock or arguments.state_root / "gpu.lock"
    environment = {"project_revision": revision(arguments.project), "reference_revision": revision(arguments.reference),
                   "triton_ref_revision": revision(arguments.triton_ref), "compiler": str(arguments.compiler),
                   "torch": torch.__version__, "triton": triton.__version__, "gpu": torch.cuda.get_device_name(0),
                   "model": arguments.suite["model"], "reasoning_effort": arguments.suite["reasoning_effort"],
                   "suite": str(arguments.suite_path),
                   "tasks": rows, "submission_policy": "one complete program; documentation tools; no benchmark feedback",
                   "timing": "complete operator; per-task paired CUDA Graph or CUDA event timing; compilation/tuning excluded",
                   "isolation": "dedicated Codex state/provider; workspace-only shell, no network; read-only public manual MCP; no reference/history/agents",
                   "instructions": Path(__file__).with_name("instructions.md").read_text()}
    (arguments.output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    fields = ("task", "profile", "language", "status", "candidate_ms", "reference_ms", "ratio", "timing", "failure_stage", "error", "program")
    with ((arguments.output / "results.csv").open("w", newline="") as output,
          ThreadPoolExecutor(max_workers=arguments.workers) as generation,
          ThreadPoolExecutor(max_workers=arguments.benchmark_workers) as evaluation):
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        futures = {generation.submit(generate_trial, arguments, row, language): True
                   for row in rows for language in arguments.arms}
        try:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    result = future.result()
                    generated = futures.pop(future)
                    if generated and result["status"] == "submitted":
                        futures[evaluation.submit(evaluate_trial, arguments, result)] = False
                    else:
                        writer.writerow(result)
                        output.flush()
        except BaseException:
            arguments.stop.set()
            for pending in futures:
                pending.cancel()
            raise


if __name__ == "__main__":
    main()
