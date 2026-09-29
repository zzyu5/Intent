from __future__ import annotations

import argparse
from contextlib import nullcontext, redirect_stdout
import json
import os
from pathlib import Path
import sys
import time
import traceback

import intent
import torch
from triton.compiler.errors import CompilationError, CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from experiments._common.measurement import evaluate, NumericalComparisonError, PipelineStageError, observe_stages, report_stage
from experiments._common.measurement import gpu_execution
from experiments._common.model import PreparedComparison, PreparedLaunch

from .program import load_program, ProgramContext, TuningBudget
from .tasks import SUITE_PATH, catalog, invocation, read_suite, reference, tolerance


class CandidateTorchPolicy(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        allowed = {"aten.empty", "aten.empty_strided", "aten.empty_like", "aten.new_empty", "aten.view", "aten._unsafe_view",
                   "aten.as_strided", "aten.detach", "aten.alias", "aten.permute", "aten.transpose",
                   "aten.squeeze", "aten.unsqueeze", "aten.slice", "aten.select", "aten.expand",
                   "aten.view_as_real", "aten.view_as_complex", "aten.result_type"}
        name = str(function).rsplit(".", 1)[0]
        # These calls may occur in unused host branches; execution is forbidden.
        if name in {"aten.ones", "aten.ones_like", "aten.zeros_like", "aten.std", "aten.std_mean"}:
            raise ValueError(f"Tensor computation must use the submitted language, not PyTorch {function}")
        if name in allowed:
            return function(*args, **(kwargs or {}))
        tensors = [value for value in tree_flatten((args, kwargs))[0] if isinstance(value, torch.Tensor)]
        device = (kwargs or {}).get("device")
        uses_cuda = any(value.is_cuda for value in tensors) or device is not None and torch.device(device).type == "cuda"
        if uses_cuda:
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


def run(arguments, *, suite_path: Path = SUITE_PATH) -> dict:
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suite = read_suite(suite_path)
    task = next(task for task in suite["tasks"] if task["id"] == arguments.task)
    row = next(row for row in catalog(arguments.reference, suite) if row["task"] == arguments.task)
    timing = arguments.timing or task.get("timing", suite["timing"])
    cuda_graph = timing == "cuda_graph"
    reference_only = arguments.language == "reference"
    result = {"status": "pending", "candidate_ms": None, "reference_ms": None, "ratio": None,
              "target": arguments.target, "program": str(arguments.program) if arguments.program is not None else None,
              "reference_time_reused": arguments.reference_ms is not None,
              "reference_timing_note": None,
              "timing": timing, "tolerance": suite["tolerances"][task["tolerance"]]}
    def source_timing_error(error: Exception) -> bool:
        cause = error
        while cause is not None:
            if isinstance(cause, torch.AcceleratorError) and "operation not permitted when stream is capturing" in str(cause):
                result["reference_timing_note"] = (
                    "The unchanged PyTorch reference cannot be captured by CUDA Graph. "
                    "Its eager output remains the numerical oracle; no reference ratio is available."
                )
                result["reference_timing_error"] = str(cause)
                return True
            cause = cause.__cause__ or cause.__context__
        return False
    started = time.monotonic()
    stage = "reference_preparation"
    report_stage(stage)
    budget = TuningBudget(CandidateTorchPolicy)
    context = None if reference_only else ProgramContext(arguments.compiler, arguments.program.parent,
                                                        language=arguments.language, target=arguments.target,
                                                        tuning_config=arguments.tuning_config)
    try:
        reference_call = invocation(arguments.reference, row, task, suite, device="cpu")
        reference_function = reference(arguments.reference, row)
        if reference_only:
            with gpu_execution(arguments.gpu_lock, providers={"triton"}):
                report_stage(stage)
                reference_call = reference_call.to_device("cuda")
                torch.cuda.synchronize()
                source = _observe(reference_call, reference_function, task=arguments.task, enforce=False)
                _, anchor = evaluate(PreparedComparison(None, source, tolerance(task, suite), cuda_graph=cuda_graph),
                                     source_timing_error=source_timing_error, benchmark_time_budget_ms=200)
            result.update(status="reference_pass" if anchor is not None else "reference_timing_unavailable",
                          reference_ms=anchor)
        else:
            candidate_call = invocation(arguments.reference, row, task, suite, device="cpu")
            stage = "candidate_load"
            report_stage(stage)
            provider_budget = nullcontext()
            if arguments.target == "cutile":
                import cuda.tile as ct
                provider_budget = ct.compiler_timeout(arguments.cutile_compiler_timeout)
            with budget, provider_budget, context.native_compilation_cache():
                with CandidateTorchPolicy():
                    module = load_program(arguments.program, language=arguments.language)
                    stage = "candidate_build"
                    report_stage(stage)
                    function = module.build(context)
                    if arguments.language == "intent" and not context.generated:
                        raise ValueError("Intent submission did not compile any Intent kernels")
                stage = "candidate_input_preparation"
                with gpu_execution(arguments.gpu_lock, providers={arguments.target}):
                    report_stage(stage)
                    candidate_call = candidate_call.to_device("cuda")
                    reference_call = reference_call.to_device("cuda")
                    torch.cuda.synchronize()
                stage = "candidate_precompile"
                report_stage(stage)
                compile_started = time.monotonic()
                preparation = (budget.compilation_only() if arguments.target == "triton"
                               else context.compilation_only())
                with preparation, CandidateTorchPolicy():
                    candidate_call.call(function)
                result["precompile_seconds"] = time.monotonic() - compile_started
                result["preparation_policy"] = "compile_only_before_gpu_timing_lock"
                for artifact in context.generated.values():
                    artifact.backend_ir.clear()
                stage = "comparison"
                with gpu_execution(arguments.gpu_lock, providers={arguments.target}):
                    report_stage(stage)
                    candidate = _observe(candidate_call, function, task=arguments.task, enforce=True)
                    source = _observe(reference_call, reference_function, task=arguments.task, enforce=False)
                    measured, anchor = evaluate(PreparedComparison(candidate, source, tolerance(task, suite), cuda_graph=cuda_graph),
                                                source_timing_error=source_timing_error,
                                                benchmark_time_budget_ms=200,
                                                source_time_ms=arguments.reference_ms,
                                                measure_source=not arguments.reference_time_unavailable)
                result.update(status="pass", candidate_ms=measured, reference_ms=anchor,
                              ratio=measured / anchor if anchor is not None else None)
    except Exception as error:
        # A failed program is a result in the fixed denominator, never a fallback.
        causes = []
        cause = error
        while cause is not None:
            causes.append(cause)
            cause = cause.__cause__
        if any(isinstance(cause, ModuleNotFoundError) and cause.name.split(".")[0] in {"mlir", "torch", "triton", "intent", "cuda"} for cause in causes):
            status = "benchmark_environment_failure"
        elif isinstance(error, NumericalComparisonError):
            status = "numerical_failure"
        elif isinstance(error, PipelineStageError) and error.stage.startswith("source_"):
            status, stage = "reference_failure", error.stage
        elif arguments.language == "intent" and (
            any(isinstance(cause, (CompilationError, CompileTimeAssertionFailure, OutOfResources, PTXASError)) for cause in causes)
            or arguments.target == "cutile" and any(isinstance(cause, ct.TileError) for cause in causes)
        ):
            status, stage = "compilation_failure", "provider_compilation"
        elif isinstance(error, PipelineStageError):
            stage = error.stage
            status = "execution_failure"
        elif hasattr(error, "stage"):
            status = "compilation_failure"
            stage = error.stage
        else:
            status = "reference_failure" if stage == "reference_preparation" else "agent_program_error"
        result.update(status=status, failure_stage=stage, error=str(error), traceback=traceback.format_exc())
        if isinstance(error, intent.CompilationStageError) and error.cache_directory is not None:
            result["failed_compiler_artifact"] = str(error.cache_directory)
    if context is not None:
        result["compiler_artifacts"] = {name: str(artifact.cache_directory)
                                        for name, artifact in context.generated.items()}
        result["precompile_failures"] = budget.precompile_failures + context.precompile_failures
        if arguments.target == "cutile":
            result["native_compile_reuses"] = context.native_compile_reuses
        result["tuning"] = context.tuning if arguments.target == "cutile" else budget.records()
    result["preparation_and_benchmark_seconds"] = time.monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--language", choices=("intent", "triton", "reference"), required=True)
    parser.add_argument("--target", choices=("triton", "cutile"), default="triton",
                        help="backend for the unchanged Intent submission")
    parser.add_argument("--cutile-compiler-timeout", type=int, default=15)
    parser.add_argument("--tuning-config", type=Path,
                        help="fixed compiler profile for Intent generation; does not change the submission")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    parser.add_argument("--phase-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--suite", type=Path, default=SUITE_PATH,
                        help="Fixed task and numerical configuration used by generation")
    parser.add_argument("--timing", choices=("cuda_graph", "cuda_event"),
                        help="Override the task's paired candidate/reference timing path")
    reference_timing = parser.add_mutually_exclusive_group()
    reference_timing.add_argument("--reference-ms", type=float,
                                  help="Reuse this workload's existing reference time; still check its output")
    reference_timing.add_argument("--reference-time-unavailable", action="store_true",
                                  help="Preserve unavailable reference timing; still check its output")
    arguments = parser.parse_args()
    if arguments.language != "reference" and arguments.program is None:
        parser.error("candidate evaluation requires --program")
    if arguments.language == "reference" and (arguments.program is not None or arguments.reference_ms is not None
                                               or arguments.reference_time_unavailable):
        parser.error("reference measurement does not take a candidate or a reused reference time")
    if arguments.language != "intent" and arguments.target != "triton":
        parser.error("only Intent submissions can select a different backend")
    if arguments.tuning_config is not None and arguments.language != "intent":
        parser.error("only Intent submissions use a compiler tuning profile")
    if arguments.cutile_compiler_timeout <= 0:
        parser.error("--cutile-compiler-timeout must be positive")
    if arguments.reference_ms is not None and not 0 < arguments.reference_ms < float("inf"):
        parser.error("--reference-ms must be finite and positive")
    with redirect_stdout(sys.stderr):
        if arguments.phase_fd is None:
            result = run(arguments, suite_path=arguments.suite)
        else:
            with os.fdopen(arguments.phase_fd, "w", buffering=1) as phases:
                def publish(stage):
                    phases.write(json.dumps({"stage": stage, "time": time.monotonic()}) + "\n")
                with observe_stages(publish):
                    result = run(arguments, suite_path=arguments.suite)
    arguments.result.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}))


if __name__ == "__main__":
    main()
