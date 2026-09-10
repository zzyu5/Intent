from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from repro.v2.measurement import evaluate, NumericalComparisonError, PipelineStageError
from repro.v2.model import PreparedComparison, PreparedLaunch

from .program import load_program, ProgramContext, TuningBudget, export_seed
from .tasks import catalog, invocation, read_suite, reference, tolerance


class CandidateTorchPolicy(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        tensors = [value for value in tree_flatten((args, kwargs))[0] if isinstance(value, torch.Tensor)]
        allowed = {"aten.empty", "aten.empty_strided", "aten.empty_like", "aten.view", "aten._unsafe_view",
                   "aten.as_strided", "aten.detach", "aten.alias", "aten.permute", "aten.transpose",
                   "aten.squeeze", "aten.unsqueeze", "aten.slice", "aten.select", "aten.expand"}
        name = str(function).rsplit(".", 1)[0]
        device = (kwargs or {}).get("device")
        uses_cuda = any(value.is_cuda for value in tensors) or device is not None and torch.device(device).type == "cuda"
        if uses_cuda and name not in allowed:
            raise ValueError(f"GPU computation must use the submitted language, not PyTorch {function}")
        return function(*args, **(kwargs or {}))


def _tensors(call) -> tuple:
    return tuple(value for value in tree_flatten((call.args, call.kwargs))[0]
                 if isinstance(value, torch.Tensor))


def _observe(call, function, *, task: str, enforce: bool) -> PreparedLaunch:
    latest = []

    def launch():
        if enforce:
            with CandidateTorchPolicy():
                result = call.call(function)
        else:
            result = call.call(function)
        out = call.bound.get("out")
        if out is not None and result is not out:
            raise ValueError("the task's out buffer must also be the returned tensor")
        if task == "permute_copy" and result.untyped_storage().data_ptr() == call.bound["input"].untyped_storage().data_ptr():
            raise ValueError("permute_copy requires a fresh output allocation, not an alias")
        latest[:] = [result]
        return result

    # Include observable input buffers in the one normal comparison. In
    # particular, a candidate cannot overwrite its input before the reference.
    return PreparedLaunch(launch, lambda: (latest[0], _tensors(call)))


def run(arguments) -> dict:
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suite = read_suite()
    task = next(task for task in suite["tasks"] if task["id"] == arguments.task)
    row = next(row for row in catalog(arguments.reference, suite) if row["task"] == arguments.task)
    result = {"status": "pending", "candidate_ms": None, "reference_ms": None, "ratio": None,
              "timing": "cuda_graph", "tolerance": suite["tolerances"][task["tolerance"]]}
    started = time.monotonic()
    stage = "reference_preparation"
    budget = TuningBudget(suite["max_tuning_configurations"], CandidateTorchPolicy)
    artifact_directory = arguments.artifacts or arguments.program.parent
    artifact_directory.mkdir(parents=True, exist_ok=True)
    context = ProgramContext(arguments.compiler, artifact_directory, language=arguments.language,
                             keep_ir=arguments.artifacts is not None)
    try:
        candidate_call = invocation(arguments.reference, row, task, suite, device="cpu")
        reference_call = invocation(arguments.reference, row, task, suite, device="cpu")
        reference_function = reference(arguments.reference, row)
        stage = "candidate_load"
        with budget:
            with CandidateTorchPolicy():
                module = load_program(arguments.program, language=arguments.language)
                stage = "candidate_build"
                function = module.build(context)
            stage = "comparison"
            with arguments.gpu_lock.open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                candidate_call = candidate_call.to_device("cuda")
                reference_call = reference_call.to_device("cuda")
                candidate = _observe(candidate_call, function, task=arguments.task, enforce=True)
                source = _observe(reference_call, reference_function, task=arguments.task, enforce=False)
                measured, anchor = evaluate(PreparedComparison(candidate, source, tolerance(task, suite), cuda_graph=True))
            result.update(status="pass", candidate_ms=measured, reference_ms=anchor, ratio=measured / anchor)
            if arguments.language == "intent":
                stage = "source_export"
                export_seed(arguments.program, context, artifact_directory / "triton_seed")
    except Exception as error:
        # A failed program is a result in the fixed denominator, never a fallback.
        causes = []
        cause = error
        while cause is not None:
            causes.append(cause)
            cause = cause.__cause__
        if any(isinstance(cause, ModuleNotFoundError) and cause.name.split(".")[0] in {"mlir", "torch", "triton", "intent"} for cause in causes):
            status = "benchmark_environment_failure"
        elif isinstance(error, NumericalComparisonError):
            status = "numerical_failure"
        elif isinstance(error, PipelineStageError):
            status = "execution_failure"
            stage = error.stage
        elif hasattr(error, "stage"):
            status = "compilation_failure"
            stage = error.stage
        else:
            status = "reference_failure" if stage == "reference_preparation" else "agent_program_error"
        result.update(status=status, failure_stage=stage, error=str(error), traceback=traceback.format_exc())
    result["tuning"] = budget.records()
    result["preparation_and_benchmark_seconds"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--language", choices=("intent", "triton"), required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, help="Separate compiler recheck artifacts from an existing submission")
    arguments = parser.parse_args()
    with redirect_stdout(sys.stderr):
        result = run(arguments)
    arguments.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}))


if __name__ == "__main__":
    main()
