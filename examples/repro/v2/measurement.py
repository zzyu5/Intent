from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
import statistics

import torch

import intent
from repro.common.support import benchmark
from repro.common.support import prepare_kernel_call

from .model import Context
from .model import PreparedComparison
from .model import PreparedLaunch
from .model import TensorTree
from .model import Tolerance


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class NumericalComparisonError(RuntimeError):
    pass


MEASUREMENT_REPETITIONS = 200


_stage_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "repro_stage_observer", default=None
)


@contextmanager
def observe_stages(observer: Callable[[str], None]):
    token = _stage_observer.set(observer)
    try:
        yield
    finally:
        _stage_observer.reset(token)


def report_stage(stage: str) -> None:
    observer = _stage_observer.get()
    if observer is not None:
        observer(stage)


def initial_launch(function, *, side: str):
    stage = f"{side}_provider_jit_or_initial_launch"
    report_stage(stage)
    try:
        result = function()
        torch.cuda.synchronize()
    except Exception as error:
        raise PipelineStageError(stage, str(error)) from error
    report_stage("adapter_preparation")
    return result


def compile_single(
    context: Context,
    definition,
    arguments: tuple[object, ...],
    *,
    constexprs: dict[str, object] | None = None,
) -> tuple[object, PreparedLaunch]:
    report_stage("generated_compilation")
    try:
        artifact = intent.compile(
            definition,
            target=context.target,
            compiler=context.compiler,
            constexprs=constexprs,
        )
    except intent.CompilationStageError as error:
        raise PipelineStageError(f"generated_{error.stage}", str(error)) from error
    result = initial_launch(lambda: artifact.run(*arguments), side="generated")
    report_stage("generated_launcher_preparation")
    try:
        launch_outputs = () if result is None else result
        launch = prepare_kernel_call(artifact, arguments, launch_outputs)
    except Exception as error:
        raise PipelineStageError("generated_launcher_preparation", str(error)) from error
    report_stage("adapter_preparation")
    return artifact, PreparedLaunch(launch=launch, outputs=lambda: result)


def functional_launch(function) -> PreparedLaunch:
    state: dict[str, TensorTree] = {}

    def launch():
        state["output"] = function()
        return state["output"]

    initial_launch(launch, side="source")
    return PreparedLaunch(launch=launch, outputs=lambda: state["output"])


def _leaves(value: TensorTree) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
        return
    if not isinstance(value, tuple):
        raise TypeError(
            "V2 result trees contain only tensors and tuples, got "
            f"{type(value).__name__}"
        )
    for element in value:
        yield from _leaves(element)


def _result_structure(value: TensorTree):
    if isinstance(value, torch.Tensor):
        return "tensor"
    if not isinstance(value, tuple):
        raise TypeError(
            "V2 result trees contain only tensors and tuples, got "
            f"{type(value).__name__}"
        )
    return tuple(_result_structure(element) for element in value)


def compare_outputs(
    generated: TensorTree,
    source: TensorTree,
    tolerance: Tolerance | tuple[Tolerance, ...],
) -> tuple[float, ...]:
    generated_structure = _result_structure(generated)
    source_structure = _result_structure(source)
    if generated_structure != source_structure:
        raise NumericalComparisonError(
            "generated/source result structure differs: "
            f"{generated_structure!r} != {source_structure!r}"
        )
    generated_values = tuple(_leaves(generated))
    source_values = tuple(_leaves(source))
    if len(generated_values) != len(source_values):
        raise NumericalComparisonError(
            "generated/source result count differs: "
            f"{len(generated_values)} != {len(source_values)}"
        )
    tolerances = (
        tolerance
        if isinstance(tolerance, tuple)
        else tuple(tolerance for _ in generated_values)
    )
    if len(tolerances) != len(generated_values):
        raise NumericalComparisonError(
            "tolerance count differs from result count: "
            f"{len(tolerances)} != {len(generated_values)}"
        )
    errors: list[float] = []
    for leaf_index, (generated_value, source_value, leaf_tolerance) in enumerate(
        zip(generated_values, source_values, tolerances)
    ):
        if generated_value.shape != source_value.shape:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} shape differs: "
                f"{tuple(generated_value.shape)} != {tuple(source_value.shape)}"
            )
        if generated_value.dtype != source_value.dtype:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} dtype differs: "
                f"{generated_value.dtype} != {source_value.dtype}"
            )
        if generated_value.dtype == torch.bool or not generated_value.is_floating_point():
            if not torch.equal(generated_value, source_value):
                raise NumericalComparisonError(
                    f"generated/source integer result {leaf_index} differs"
                )
            errors.append(0.0)
            continue
        generated_compare = generated_value.float()
        source_compare = source_value.float()
        generated_finite = torch.isfinite(generated_compare)
        source_finite = torch.isfinite(source_compare)
        if not torch.equal(generated_finite, source_finite):
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} finite-value masks differ"
            )
        finite = generated_finite & source_finite
        nonfinite = ~finite
        if nonfinite.any():
            generated_nonfinite = generated_compare[nonfinite]
            source_nonfinite = source_compare[nonfinite]
            if not (
                torch.equal(
                    torch.isnan(generated_nonfinite),
                    torch.isnan(source_nonfinite),
                )
                and torch.equal(
                    torch.isposinf(generated_nonfinite),
                    torch.isposinf(source_nonfinite),
                )
                and torch.equal(
                    torch.isneginf(generated_nonfinite),
                    torch.isneginf(source_nonfinite),
                )
            ):
                raise NumericalComparisonError(
                    f"generated/source result {leaf_index} non-finite values differ"
                )
        if finite.any():
            difference = (
                generated_compare[finite] - source_compare[finite]
            ).abs()
            limit = (
                leaf_tolerance.atol
                + leaf_tolerance.rtol * source_compare[finite].abs()
            )
            maximum = difference.max().item()
            if torch.any(difference > limit):
                raise NumericalComparisonError(
                    f"generated/source floating result {leaf_index} differs: "
                    f"max_abs={maximum}, atol={leaf_tolerance.atol}, "
                    f"rtol={leaf_tolerance.rtol}"
                )
            errors.append(maximum)
        else:
            errors.append(0.0)
    return tuple(errors)


def evaluate(comparison: PreparedComparison) -> tuple[float | None, float | None]:
    report_stage("generated_launch")
    try:
        if comparison.generated.prepare is not None:
            comparison.generated.prepare()
        comparison.generated.launch()
        torch.cuda.synchronize()
    except Exception as error:
        raise PipelineStageError("generated_launch", str(error)) from error
    report_stage("source_launch")
    try:
        if comparison.source.prepare is not None:
            comparison.source.prepare()
        comparison.source.launch()
        torch.cuda.synchronize()
    except Exception as error:
        raise PipelineStageError("source_launch", str(error)) from error
    report_stage("numerical_comparison")
    compare_outputs(
        comparison.generated.outputs(),
        comparison.source.outputs(),
        comparison.tolerance,
    )
    if comparison.status != "pass":
        return None, None
    report_stage("generated_benchmark")
    try:
        generated_first, _ = benchmark(
            comparison.generated.launch,
            warmup=25,
            repetitions=MEASUREMENT_REPETITIONS,
            cuda_graph=comparison.cuda_graph,
            prepare=comparison.generated.prepare,
        )
    except Exception as error:
        raise PipelineStageError("generated_benchmark", str(error)) from error
    report_stage("source_benchmark")
    try:
        source_first, _ = benchmark(
            comparison.source.launch,
            warmup=25,
            repetitions=MEASUREMENT_REPETITIONS,
            cuda_graph=comparison.cuda_graph,
            prepare=comparison.source.prepare,
        )
    except Exception as error:
        raise PipelineStageError("source_benchmark", str(error)) from error
    report_stage("source_reverse_benchmark")
    try:
        source_second, _ = benchmark(
            comparison.source.launch,
            warmup=0,
            repetitions=MEASUREMENT_REPETITIONS,
            cuda_graph=comparison.cuda_graph,
            prepare=comparison.source.prepare,
        )
    except Exception as error:
        raise PipelineStageError("source_reverse_benchmark", str(error)) from error
    report_stage("generated_reverse_benchmark")
    try:
        generated_second, _ = benchmark(
            comparison.generated.launch,
            warmup=0,
            repetitions=MEASUREMENT_REPETITIONS,
            cuda_graph=comparison.cuda_graph,
            prepare=comparison.generated.prepare,
        )
    except Exception as error:
        raise PipelineStageError("generated_reverse_benchmark", str(error)) from error
    return (
        statistics.median((generated_first, generated_second)),
        statistics.median((source_first, source_second)),
    )
