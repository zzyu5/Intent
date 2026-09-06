from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import torch

import intent

from .measurement import evaluate
from .measurement import observe_stages
from .measurement import report_stage
from .measurement import NumericalComparisonError
from .measurement import PipelineStageError
from .model import Context
from .model import ComparisonUnavailable
from .model import ResultRow
from .providers import load_cases
from .registry import BY_PROVIDER


FIELDS = (
    "kernel",
    "case",
    "generated_p50_ms",
    "source_p50_ms",
    "ratio",
    "status",
)

WORKER_TIMEOUT_SECONDS = 300


def _target(provider: str):
    if provider == "triton":
        return intent.TritonTarget(device=0)
    if provider == "cutile":
        return intent.CuTileTarget(device=0)
    if provider == "tilelang":
        return intent.TileLangTarget(device=0)
    raise ValueError(f"unknown provider: {provider}")


def _write(path: Path, rows: list[ResultRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
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
                }
            )
    temporary.replace(path)


def _write_stage(path: Path, stage: str) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": stage}))
    temporary.replace(path)


def _run_entry(provider: str, compiler: str, entry) -> ResultRow:
    report_stage("device_setup")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    project_root = Path(__file__).resolve().parents[3]
    context = Context(
        compiler=compiler,
        project_root=project_root,
        target=_target(provider),
        provider=provider,
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
        return ResultRow(entry.kernel, entry.case, None, None, None, "unsupported")
    except PipelineStageError as error:
        status = f"{error.stage}_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status)
    except Exception as error:
        status = "adapter_preparation_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status)

    try:
        generated_p50, source_p50 = evaluate(comparison)
    except NumericalComparisonError as error:
        print(f"{provider}:{entry.kernel}: numerical_failed: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, "numerical_failed")
    except PipelineStageError as error:
        status = f"{error.stage}_failed"
        print(f"{provider}:{entry.kernel}: {status}: {error}")
        return ResultRow(entry.kernel, entry.case, None, None, None, status)

    if comparison.status != "pass":
        print(
            f"{provider}:{entry.kernel}: {comparison.status}: "
            "numerical comparison completed; timing contract is not comparable"
        )
        return ResultRow(entry.kernel, entry.case, None, None, None, comparison.status)
    ratio = generated_p50 / source_p50
    print(
        f"{provider}:{entry.kernel}: {comparison.status} "
        f"generated={generated_p50:.6f} ms source={source_p50:.6f} ms ratio={ratio:.6f}"
    )
    return ResultRow(
        entry.kernel,
        entry.case,
        generated_p50,
        source_p50,
        ratio,
        comparison.status,
    )


def _read_worker_row(path: Path) -> ResultRow:
    with path.open(newline="") as stream:
        records = tuple(csv.DictReader(stream))
    if len(records) != 1:
        raise RuntimeError(f"worker produced {len(records)} result rows")
    record = records[0]

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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=sorted(BY_PROVIDER))
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel", action="append")
    parser.add_argument("--worker-entry", type=int, help=argparse.SUPPRESS)
    arguments = parser.parse_args()

    provider = arguments.provider
    selected = set(arguments.kernel or ())
    entries = tuple(
        entry
        for entry in BY_PROVIDER[provider]
        if not selected or entry.kernel in selected
    )
    if selected - {entry.kernel for entry in entries}:
        unknown = ", ".join(sorted(selected - {entry.kernel for entry in entries}))
        parser.error(f"unknown {provider} V2 kernel(s): {unknown}")

    if arguments.worker_entry is not None:
        entry = BY_PROVIDER[provider][arguments.worker_entry]
        phase_path = arguments.output.with_suffix(".phase.json")
        with observe_stages(lambda stage: _write_stage(phase_path, stage)):
            _write(arguments.output, [_run_entry(provider, arguments.compiler, entry)])
        return

    rows: list[ResultRow] = []
    selected_indexes = (
        index
        for index, entry in enumerate(BY_PROVIDER[provider])
        if not selected or entry.kernel in selected
    )
    for index in selected_indexes:
        entry = BY_PROVIDER[provider][index]
        with tempfile.TemporaryDirectory(prefix="intentdsl-baseline-v2-") as directory:
            worker_output = Path(directory) / "result.csv"
            phase_path = worker_output.with_suffix(".phase.json")
            _write_stage(phase_path, "worker_startup")
            worker = subprocess.Popen(
                (
                    sys.executable,
                    "-m",
                    "repro.v2.runner",
                    provider,
                    "--compiler",
                    arguments.compiler,
                    "--output",
                    str(worker_output),
                    "--worker-entry",
                    str(index),
                ),
                start_new_session=True,
            )
            try:
                returncode = worker.wait(timeout=WORKER_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                stage = json.loads(phase_path.read_text())["stage"]
                status = f"{stage}_timeout"
                print(
                    f"{provider}:{entry.kernel}: {status}: "
                    f"exceeded {WORKER_TIMEOUT_SECONDS} seconds"
                )
                row = ResultRow(entry.kernel, entry.case, None, None, None, status)
            else:
                if returncode != 0 or not worker_output.exists():
                    stage = json.loads(phase_path.read_text())["stage"]
                    status = f"{stage}_process_failed"
                    print(
                        f"{provider}:{entry.kernel}: {status}: "
                        f"worker exited with code {returncode}"
                    )
                    row = ResultRow(entry.kernel, entry.case, None, None, None, status)
                else:
                    try:
                        row = _read_worker_row(worker_output)
                    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                        status = "worker_result_failed"
                        print(f"{provider}:{entry.kernel}: {status}: {error}")
                        row = ResultRow(
                            entry.kernel, entry.case, None, None, None, status
                        )
        rows.append(row)
        _write(arguments.output, rows)


if __name__ == "__main__":
    main()
