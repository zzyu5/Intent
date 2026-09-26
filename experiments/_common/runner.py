from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import fcntl
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import torch

import intent
from experiments import PROJECT_ROOT

from .measurement import evaluate
from .measurement import observe_stages
from .measurement import report_stage
from .measurement import cpu_preparation, gpu_execution, gpu_queue_seconds
from .measurement import NumericalComparisonError
from .measurement import PipelineStageError
from .model import Context
from .model import ComparisonUnavailable
from .model import ResultRow
from .providers import load_cases
from .registry import BY_PROVIDER, PROVIDER_GROUPS


FIELDS = (
    "kernel",
    "case",
    "generated_p50_ms",
    "source_p50_ms",
    "ratio",
    "status",
    "note",
)

WORKER_TIMEOUT_SECONDS = 300
CPU_WAIT_ENVIRONMENT = {"OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0", "GOMP_SPINCOUNT": "0"}


def _target(provider: str, entry):
    if provider == "mojo":
        return intent.MojoTarget(workers=8)
    if provider == "weft":
        from intent.runtime.weft import TargetProfile
        path = Path(os.environ.get("INTENT_WEFT_PROFILE", PROJECT_ROOT / "experiments/cpu/providers" / (entry.deployment or "weft/rvv.json")))
        profile = TargetProfile.from_deployment(json.loads(path.read_text()))
        return intent.WeftTarget(vector_bits=profile.vlen_bits, workers=len(profile.cpus),
                                 matrix_extension=profile.matrix_extension)
    if provider == "bangc":
        return intent.BangCTarget(device=int(os.environ.get("INTENT_BANGC_DEVICE", "0")))
    if provider == "triton":
        return intent.TritonTarget(device=0)
    if provider == "cutile":
        return intent.CuTileTarget(device=0)
    if provider == "tilelang":
        return intent.TileLangTarget(device=0)
    raise ValueError(f"unknown provider: {provider}")


def _write(path: Path, rows: list[ResultRow], *, target: str | None = None,
           changed: ResultRow | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open(newline="") as stream:
            fields = csv.DictReader(stream).fieldnames
        if fields and "intent_triton_status" in fields:
            _write_target_rows(path, [changed] if changed is not None else rows,
                               target=target)
            return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "kernel": row.kernel,
                    "case": row.case,
                    "generated_p50_ms": (
                        "" if row.generated_p50_ms is None else f"{row.generated_p50_ms:.6f}"
                    ),
                    "source_p50_ms": (
                        "" if row.source_p50_ms is None else f"{row.source_p50_ms:.6f}"
                    ),
                    "ratio": "" if row.ratio is None else f"{row.ratio:.6f}",
                    "status": row.status,
                    "note": row.note,
                }
            )
    temporary.replace(path)


def _write_target_rows(path: Path, rows: list[ResultRow], *, target: str | None) -> None:
    if target not in {"triton", "cutile"}:
        raise ValueError("a cross-backend result table requires its Intent target")
    from intent.compiler.cache import cache_root
    lock_path = cache_root() / "benchmark-results.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            records = {(r["kernel"], r["case"]): r for r in reader}
        prefix = "intent_" + target + "_"
        for row in rows:
            key = row.kernel, row.case
            record = records.setdefault(key, {"kernel": row.kernel, "case": row.case})
            record[prefix + "p50_ms"] = "" if row.generated_p50_ms is None else f"{row.generated_p50_ms:.6f}"
            record[prefix + "ratio"] = "" if row.ratio is None else f"{row.ratio:.6f}"
            record[prefix + "status"] = row.status
            record[prefix + "note"] = row.note.strip()
            if row.source_p50_ms is not None:
                record["source_p50_ms"] = f"{row.source_p50_ms:.6f}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records.values())
        temporary.replace(path)


def _write_stage(path: Path, stage: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage, "time": time.monotonic(),
                                     "queued_seconds": gpu_queue_seconds()}))
    temporary.replace(path)


def _run_entry(
    provider: str, compiler: str, entry, compiler_timeout: int,
    tuning_config: Path | None, before_benchmark, *, target: str,
    source_time_ms: float | None = None,
    measure_source: bool = True,
) -> ResultRow:
    report_stage("device_setup")
    if provider == "mojo":
        from experiments.cpu.providers.mojo.common import configure_cpu_budget
        configure_cpu_budget(8)
    elif provider not in {"weft", "bangc"}:
        torch.cuda.set_device(0)
    torch.manual_seed(0)
    project_root = PROJECT_ROOT
    context = Context(
        compiler=compiler,
        project_root=project_root,
        target=_target(target, entry),
        provider=provider,
        compiler_timeout_seconds=compiler_timeout,
        tuning_config=tuning_config,
    )
    report_stage("adapter_loading")
    try:
        cases = load_cases(provider)
    except Exception as error:
        status = "adapter_loading_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status)

    factory = cases.get(entry.kernel)
    if factory is None:
        status = "adapter_missing"
        print(f"{provider}:{entry.kernel}: {status}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status)
    report_stage("adapter_preparation")
    try:
        comparison = factory(context)
    except ComparisonUnavailable as error:
        print(f"{provider}:{entry.kernel}: {error.status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, error.status)
    except NotImplementedError as error:
        print(f"{provider}:{entry.kernel}: unsupported: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, "unsupported", str(error))
    except PipelineStageError as error:
        status = f"{error.stage}_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(
            entry.kernel, entry.case, None, None, None, status,
            "; ".join(str(error).splitlines()[:2]),
        )
    except Exception as error:
        status = "adapter_preparation_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status,
                         "; ".join(str(error).splitlines()[:2]))

    if provider == "mojo":
        settings = ", ".join(f"{name}={os.environ.get(name, 'unset')}" for name in CPU_WAIT_ENVIRONMENT)
        comparison = replace(comparison, note=comparison.note + " CPU idle wait: " + settings + ".")
    if target != provider:
        comparison = replace(comparison, note=(f"Intent target={target}; source corpus={provider}. " + comparison.note).rstrip())
    try:
        generated_p50, source_p50 = evaluate(
            comparison, before_benchmark=before_benchmark,
            source_time_ms=source_time_ms,
            measure_source=measure_source,
        )
    except NumericalComparisonError as error:
        print(f"{provider}:{entry.kernel}: numerical_failed: {error}")
        return ResultRow(
            entry.kernel, entry.case, None, None, None, "numerical_failed", str(error),
        )
    except PipelineStageError as error:
        status = f"{error.stage}_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(
            entry.kernel, entry.case, None, None, None, status,
            "; ".join(str(error).splitlines()[:2]),
        )

    if comparison.status == "run_only":
        print(f"{provider}:{entry.kernel}: run_only generated={generated_p50:.6f} ms; no reference")
        return ResultRow(entry.kernel, entry.case, generated_p50, None, None, "run_only", comparison.note)
    if comparison.status != "pass":
        print(
            f"{provider}:{entry.kernel}: {comparison.status}: "
            "numerical comparison completed; timing contract is not comparable"
        )
        return ResultRow(entry.kernel, entry.case, None, None, None, comparison.status)
    ratio = generated_p50 / source_p50 if generated_p50 is not None and source_p50 is not None else None
    def formatted(value):
        return "unavailable" if value is None else f"{value:.6f}"
    print(
        f"{provider}:{entry.kernel}: {comparison.status} "
        f"generated={formatted(generated_p50)} ms source={formatted(source_p50)} ms ratio={formatted(ratio)}"
    )
    return ResultRow(
        entry.kernel,
        entry.case,
        generated_p50,
        source_p50,
        ratio,
        comparison.status,
        comparison.note,
    )


def _read_rows(path: Path, *, target: str | None = None) -> list[ResultRow]:
    with path.open(newline="") as stream:
        records = tuple(csv.DictReader(stream))
    if records and "intent_triton_status" in records[0]:
        if target not in {"triton", "cutile"}:
            raise ValueError("a cross-backend result table requires its Intent target")
        prefix = "intent_" + target + "_"
        records = tuple({"kernel": r["kernel"], "case": r["case"],
                         "generated_p50_ms": r[prefix + "p50_ms"],
                         "source_p50_ms": r["source_p50_ms"],
                         "ratio": r[prefix + "ratio"], "status": r[prefix + "status"],
                         "note": r[prefix + "note"]} for r in records)
    return [_row_from_record(record) for record in records]


def _read_worker_row(path: Path) -> ResultRow:
    records = _read_rows(path)
    if len(records) != 1:
        raise RuntimeError(f"worker produced {len(records)} result rows")
    return records[0]


def _row_from_record(record: dict[str, str]) -> ResultRow:

    def optional_float(name: str) -> float | None:
        value = record[name]
        return None if value == "" else float(value)

    return ResultRow(
        record["kernel"],
        record["case"],
        optional_float("generated_p50_ms"),
        optional_float("source_p50_ms"),
        optional_float("ratio"),
        record["status"],
        record["note"] if "note" in record else "",
    )


@dataclass
class _Worker:
    index: int
    process: subprocess.Popen
    output: Path
    phase: Path
    started_at: float
    benchmarking: bool = False
    queue_baseline: float = 0.0


def _stop_worker(worker: _Worker) -> None:
    if worker.process.poll() is not None:
        return
    os.killpg(worker.process.pid, signal.SIGTERM)
    try:
        worker.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(worker.process.pid, signal.SIGKILL)
        worker.process.wait()


def _wait_for_benchmark() -> None:
    with cpu_preparation():
        report_stage("ready_for_benchmark")
        if sys.stdin.readline() != "benchmark\n":
            raise RuntimeError("benchmark coordinator disconnected")


def _run_batch(arguments, indexes, publish) -> None:
    provider = arguments.provider
    workers: list[_Worker] = []
    pending = iter(indexes)
    with tempfile.TemporaryDirectory(prefix="intentdsl-experiment-") as directory:
        try:
            while True:
                while len(workers) < arguments.jobs:
                    index = next(pending, None)
                    if index is None:
                        break
                    entry = BY_PROVIDER[provider][index]
                    output = Path(directory) / f"{index}.csv"
                    phase = output.with_suffix(".phase.json")
                    _write_stage(phase, "worker_startup")
                    command = [
                        sys.executable, "-u", "-m", f"experiments.{PROVIDER_GROUPS[provider]}", provider,
                        "--compiler", arguments.compiler, "--output", str(output),
                        "--worker-kernel", entry.kernel, "--worker-case", entry.case,
                        "--wait-for-benchmark",
                        "--cutile-compiler-timeout", str(arguments.cutile_compiler_timeout),
                        "--target", arguments.target,
                    ]
                    if arguments.tuning_config is not None:
                        command.extend(("--tuning-config", str(arguments.tuning_config)))
                    if arguments.source_results is not None:
                        command.extend(("--source-results", str(arguments.source_results)))
                    if arguments.gpu_lock is not None:
                        command.extend(("--gpu-lock", str(arguments.gpu_lock)))
                    process = subprocess.Popen(
                        command, stdin=subprocess.PIPE, text=True, start_new_session=True,
                        # Torch preparation and Mojo execution use separate pools
                        # on the same CPU budget; idle workers must yield the cores.
                        env={**os.environ, **CPU_WAIT_ENVIRONMENT} if provider == "mojo" else None,
                    )
                    workers.append(_Worker(index, process, output, phase, time.monotonic()))
                    print(f"{provider}:{entry.kernel}: preparing", flush=True)
                if not workers:
                    break
                ready: list[_Worker] = []
                for worker in tuple(workers):
                    entry = BY_PROVIDER[provider][worker.index]
                    phase = json.loads(worker.phase.read_text())
                    stage = phase["stage"]
                    returncode = worker.process.poll()
                    if returncode is not None:
                        if returncode == 0 and worker.output.exists():
                            row = _read_worker_row(worker.output)
                            if (row.kernel, row.case) != (entry.kernel, entry.case):
                                raise RuntimeError(f"worker returned {row.kernel}:{row.case} for {entry.kernel}:{entry.case}")
                        else:
                            row = ResultRow(
                                entry.kernel, entry.case, None, None, None,
                                f"{stage}_process_failed", f"worker exited with code {returncode}",
                            )
                        publish(row)
                        worker.process.stdin.close()
                        workers.remove(worker)
                        continue
                    if stage == "ready_for_benchmark" and not worker.benchmarking:
                        ready.append(worker)
                        continue
                    now = time.monotonic()
                    queued = phase["queued_seconds"] - worker.queue_baseline
                    if stage == "gpu_queue":
                        queued += now - phase["time"]
                    if now - worker.started_at - queued > arguments.worker_timeout:
                        _stop_worker(worker)
                        publish(ResultRow(
                            entry.kernel, entry.case, None, None, None,
                            f"{stage}_timeout",
                            f"exceeded {arguments.worker_timeout} seconds",
                        ))
                        worker.process.stdin.close()
                        workers.remove(worker)
                can_measure = (arguments.gpu_lock is not None or len(ready) == len(workers))
                if ready and can_measure and not any(worker.benchmarking for worker in workers):
                    worker = ready[0]
                    worker.benchmarking = True
                    worker.started_at = time.monotonic()
                    worker.queue_baseline = json.loads(worker.phase.read_text())["queued_seconds"]
                    entry = BY_PROVIDER[provider][worker.index]
                    print(f"{provider}:{entry.kernel}: measuring", flush=True)
                    worker.process.stdin.write("benchmark\n")
                    worker.process.stdin.flush()
                if workers:
                    time.sleep(0.1)
        finally:
            for worker in workers:
                _stop_worker(worker)
                worker.process.stdin.close()


def main(*, providers: tuple[str, ...] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(BY_PROVIDER if providers is None else providers))
    parser.add_argument("--target", choices=sorted(BY_PROVIDER),
                        help="generated Intent backend; defaults to the source corpus provider")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", action="append")
    parser.add_argument("--jobs", type=int, default=4,
                        help="maximum concurrent preparation workers; measurement is isolated")
    parser.add_argument("--tuning-config", type=Path,
                        help="compile-time JSON profile override; cuTile defaults to its production profile")
    parser.add_argument("--source-results", type=Path,
                        help="reuse source_p50_ms from an existing result CSV; still compare outputs")
    parser.add_argument("--gpu-lock", type=Path,
                        help="shared GPU execution lock; native and Intent compilation run outside it")
    parser.add_argument("--worker-timeout", type=int, default=WORKER_TIMEOUT_SECONDS,
                        help="limit for preparation or measurement; scheduling wait is excluded")
    parser.add_argument("--cutile-compiler-timeout", type=int, default=15,
                        help="default external cuTile compiler limit per candidate, in seconds")
    parser.add_argument("--worker-kernel", help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", help=argparse.SUPPRESS)
    parser.add_argument("--wait-for-benchmark", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker_timeout <= 0:
        parser.error("--worker-timeout must be positive")
    if arguments.cutile_compiler_timeout <= 0:
        parser.error("--cutile-compiler-timeout must be positive")
    if arguments.jobs <= 0:
        parser.error("--jobs must be positive")
    if arguments.tuning_config is None and (arguments.target or arguments.provider) == "cutile":
        arguments.tuning_config = PROJECT_ROOT / "experiments/gpu/providers/cutile/tuning.json"
    if arguments.tuning_config is not None:
        arguments.tuning_config = arguments.tuning_config.resolve(strict=True)
    source_rows = None
    if arguments.source_results is not None:
        arguments.source_results = arguments.source_results.resolve(strict=True)
        with arguments.source_results.open(newline="") as stream:
            source_rows = tuple(csv.DictReader(stream))

    def saved_source_time(entry):
        if source_rows is None:
            return None
        matches = [row["source_p50_ms"] for row in source_rows
                   if (row["kernel"], row["case"]) == (entry.kernel, entry.case)]
        if len(matches) != 1:
            parser.error(f"source results need one row for {entry.kernel}:{entry.case}")
        value = float(matches[0]) if matches[0] else None
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"saved source time must be positive and finite for {entry.kernel}:{entry.case}")
        return value

    provider = arguments.provider
    arguments.target = arguments.target or provider
    if PROVIDER_GROUPS[provider] == "gpu":
        if arguments.gpu_lock is None:
            from intent.compiler.cache import cache_root
            arguments.gpu_lock = cache_root() / "gpu.lock"
        arguments.gpu_lock = arguments.gpu_lock.resolve()
        arguments.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    elif arguments.gpu_lock is not None:
        parser.error("--gpu-lock requires a GPU provider")
    if arguments.target != provider and {arguments.target, provider} != {"triton", "cutile"}:
        parser.error("cross-backend comparisons currently support Triton and cuTile corpora")
    selected = set(arguments.kernel or ())
    entries = tuple(
        entry
        for entry in BY_PROVIDER[provider]
        if not selected or entry.kernel in selected
    )
    if selected - {entry.kernel for entry in entries}:
        unknown = ", ".join(sorted(selected - {entry.kernel for entry in entries}))
        parser.error(f"unknown {provider} kernel(s): {unknown}")

    if (arguments.worker_kernel is None) != (arguments.worker_case is None):
        parser.error("worker kernel and case must be provided together")
    if arguments.worker_kernel is not None:
        matching = [entry for entry in BY_PROVIDER[provider]
                    if (entry.kernel, entry.case) == (arguments.worker_kernel, arguments.worker_case)]
        if len(matching) != 1:
            parser.error("worker kernel/case no longer identifies one registry entry")
        entry = matching[0]
        phase_path = arguments.output.with_suffix(".phase.json")
        with observe_stages(lambda stage: _write_stage(phase_path, stage)):
            compile_budget = nullcontext()
            if "cutile" in {provider, arguments.target}:
                report_stage("provider_compiler_setup")
                import cuda.tile as ct
                compile_budget = ct.compiler_timeout(arguments.cutile_compiler_timeout)
            execution = (gpu_execution(arguments.gpu_lock,
                                       providers={provider, arguments.target})
                         if arguments.gpu_lock is not None else nullcontext())
            with compile_budget, execution:
                _write(arguments.output, [_run_entry(
                    provider, arguments.compiler, entry, arguments.cutile_compiler_timeout,
                    arguments.tuning_config,
                    _wait_for_benchmark if arguments.wait_for_benchmark else None,
                    target=arguments.target,
                    source_time_ms=saved_source_time(entry),
                    measure_source=source_rows is None,
                )])
        return

    rows = {
        row.kernel: row
        for row in (_read_rows(arguments.output, target=arguments.target)
                    if arguments.output.exists() else ())
    }
    known = {entry.kernel for entry in BY_PROVIDER[provider]}
    if rows.keys() - known:
        parser.error("output CSV contains kernels from a different registry")

    def publish(row: ResultRow) -> None:
        rows[row.kernel] = row
        _write(
            arguments.output,
            [rows[entry.kernel] for entry in BY_PROVIDER[provider] if entry.kernel in rows],
            target=arguments.target,
            changed=row,
        )
        print(f"{provider}:{row.kernel}: saved {row.status}", flush=True)

    selected_indexes = [
        index
        for index, entry in enumerate(BY_PROVIDER[provider])
        if not selected or entry.kernel in selected
    ]
    for entry in entries:
        saved_source_time(entry)
    if arguments.gpu_lock is not None:
        _run_batch(arguments, selected_indexes, publish)
    else:
        for offset in range(0, len(selected_indexes), arguments.jobs):
            _run_batch(arguments, selected_indexes[offset:offset + arguments.jobs], publish)


if __name__ == "__main__":
    main()
