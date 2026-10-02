from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
import fcntl
from functools import wraps
import statistics
import time
import math

import torch

import intent
from intent.targets import CuTileTarget
from experiments._common.support import benchmark
from experiments._common.support import prepare_kernel_call

from .model import Context
from .model import PreparedComparison
from .model import PreparedLaunch
from .model import TensorTree
from .model import Tolerance
from .model import IntegerTolerance, SimilarityTolerance, NumericalTolerance
from intent.runtime.cutile_compilation import CuTileCompilation


class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class NumericalComparisonError(RuntimeError):
    pass


MEASUREMENT_REPETITIONS = 200
COMPARISON_CHUNK_ELEMENTS = 1 << 20
CUTILE_TUNING_LAUNCH_TIMEOUT_SECONDS = 3


_stage_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "repro_stage_observer", default=None
)
_current_stage: ContextVar[str] = ContextVar("benchmark_stage", default="worker_startup")
_gpu_window: ContextVar[object | None] = ContextVar("benchmark_gpu_window", default=None)


@contextmanager
def observe_stages(observer: Callable[[str], None]):
    token = _stage_observer.set(observer)
    try:
        yield
    finally:
        _stage_observer.reset(token)


def report_stage(stage: str) -> None:
    _current_stage.set(stage)
    observer = _stage_observer.get()
    if observer is not None:
        observer(stage)


class _GPUWindow:
    def __init__(self, stream):
        self.stream = stream
        self.locked = False
        self.queued_seconds = 0.0

    def acquire(self):
        stage = _current_stage.get()
        begin = time.monotonic()
        report_stage("gpu_queue")
        fcntl.flock(self.stream, fcntl.LOCK_EX)
        self.queued_seconds += time.monotonic() - begin
        self.locked = True
        report_stage(stage)

    def release(self):
        torch.cuda.synchronize()
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.locked = False


def gpu_queue_seconds() -> float:
    window = _gpu_window.get()
    return window.queued_seconds if window is not None else 0.0


@contextmanager
def cpu_preparation():
    window = _gpu_window.get()
    if window is None or not window.locked:
        yield
        return
    window.release()
    try:
        yield
    finally:
        window.acquire()


@contextmanager
def gpu_execution(path, *, providers):
    """Serialize device work, yielding the device during CPU compilation."""
    with path.open("a") as stream, ExitStack() as patches:
        window = _GPUWindow(stream)
        token = _gpu_window.set(window)

        def compile_outside_window(owner, name):
            original = getattr(owner, name)

            @wraps(original)
            def compile_cpu(*args, **kwargs):
                with cpu_preparation():
                    return original(*args, **kwargs)

            setattr(owner, name, compile_cpu)
            patches.callback(setattr, owner, name, original)

        try:
            compile_outside_window(intent, "compile")
            compile_outside_window(intent, "generate")
            # These entries compile native binaries; their callers launch only
            # after returning, when the execution window has been reacquired.
            if "triton" in providers:
                from triton.runtime.jit import JITFunction
                compile_outside_window(JITFunction, "_do_compile")
            if "cutile" in providers:
                import cuda.tile as ct
                import cuda.tile.tune as tune
                compile_outside_window(ct.kernel, "_compile")
                original_search = tune.exhaustive_search

                @wraps(original_search)
                def bounded_search(*args, **kwargs):
                    kwargs.setdefault("single_run_timeout_sec",
                                      CUTILE_TUNING_LAUNCH_TIMEOUT_SECONDS)
                    return original_search(*args, **kwargs)

                tune.exhaustive_search = bounded_search
                patches.callback(setattr, tune, "exhaustive_search", original_search)
            window.acquire()
            yield
        finally:
            if window.locked:
                window.release()
            _gpu_window.reset(token)


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
            tuning_config=context.tuning_config,
            options=context.compile_options,
        )
    except intent.CompilationStageError as error:
        raise PipelineStageError(f"generated_{error.stage}", str(error)) from error
    if isinstance(context.target, CuTileTarget):
        compilation = CuTileCompilation()
        with compilation.cache():
            report_stage("generated_native_compilation")
            try:
                with cpu_preparation(), compilation.compilation_only((artifact,)):
                    artifact.run(*arguments)
            except Exception as error:
                raise PipelineStageError("generated_native_compilation", str(error)) from error
            result = initial_launch(lambda: artifact.run(*arguments), side="generated")
    else:
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
    tolerance: NumericalTolerance | tuple[NumericalTolerance, ...],
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
        if generated_value.device != source_value.device:
            raise NumericalComparisonError(
                f"generated/source result {leaf_index} device differs: "
                f"{generated_value.device} != {source_value.device}"
            )
        if isinstance(leaf_tolerance, IntegerTolerance):
            if generated_value.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32):
                raise TypeError("integer absolute comparison requires at most 32-bit integer outputs")
            actual, expected = generated_value.reshape(-1), source_value.reshape(-1)
            maximum = 0
            for begin in range(0, actual.numel(), COMPARISON_CHUNK_ELEMENTS):
                end = begin + COMPARISON_CHUNK_ELEMENTS
                difference = (actual[begin:end].long() - expected[begin:end].long()).abs()
                maximum = max(maximum, difference.max().item())
            if maximum > leaf_tolerance.max_abs:
                raise NumericalComparisonError(
                    f"generated/source integer result {leaf_index} differs: max_abs={maximum}, limit={leaf_tolerance.max_abs}"
                )
            errors.append(float(maximum))
            continue
        if isinstance(leaf_tolerance, SimilarityTolerance):
            if not generated_value.is_floating_point():
                raise TypeError("similarity comparison requires floating outputs")
            actual, expected = generated_value.double(), source_value.double()
            if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise NumericalComparisonError(f"similarity comparison requires finite result {leaf_index}")
            denominator = (actual * actual + expected * expected).sum()
            error = 0.0 if denominator.item() == 0.0 else abs(1.0 - (2.0 * (actual * expected).sum() / denominator).item())
            if not math.isfinite(error) or error > leaf_tolerance.max_error:
                raise NumericalComparisonError(
                    f"generated/source result {leaf_index} differs: similarity_error={error}, limit={leaf_tolerance.max_error}"
                )
            errors.append(error)
            continue
        if generated_value.dtype == torch.bool or not generated_value.is_floating_point():
            if not torch.equal(generated_value, source_value):
                raise NumericalComparisonError(
                    f"generated/source integer result {leaf_index} differs"
                )
            errors.append(0.0)
            continue
        generated_flat = generated_value.reshape(-1)
        source_flat = source_value.reshape(-1)
        maximum = 0.0
        for begin in range(0, generated_flat.numel(), COMPARISON_CHUNK_ELEMENTS):
            end = begin + COMPARISON_CHUNK_ELEMENTS
            maximum = max(
                maximum,
                _compare_float_chunk(
                    generated_flat[begin:end], source_flat[begin:end],
                    leaf_tolerance, leaf_index, begin,
                ),
            )
        errors.append(maximum)
    return tuple(errors)


def _compare_float_chunk(
    generated: torch.Tensor,
    source: torch.Tensor,
    tolerance: Tolerance,
    leaf_index: int,
    begin: int,
) -> float:
    comparison_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    generated_compare = generated.to(comparison_dtype)
    source_compare = source.to(comparison_dtype)
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
        difference = (generated_compare[finite] - source_compare[finite]).abs()
        limit = tolerance.atol + tolerance.rtol * source_compare[finite].abs()
        maximum = difference.max().item()
        if torch.any(difference > limit):
            worst = (difference - limit).argmax()
            first_failure = torch.nonzero(difference > limit)[0, 0]
            first_position = torch.nonzero(finite)[first_failure, 0].item()
            raise NumericalComparisonError(
                f"generated/source floating result {leaf_index} differs: "
                f"max_abs={maximum}, atol={tolerance.atol}, "
                f"rtol={tolerance.rtol}; first_failure={begin + first_position}; "
                f"largest tolerance excess has "
                f"generated={generated_compare[finite][worst].item()}, "
                f"source={source_compare[finite][worst].item()}, "
                f"abs_error={difference[worst].item()}, limit={limit[worst].item()}"
            )
        return maximum
    return 0.0


def _synchronize(comparison: PreparedComparison) -> None:
    if comparison.device_type == "cuda":
        torch.cuda.synchronize()
    elif comparison.device_type != "cpu":
        raise NotImplementedError(f"benchmark completion for {comparison.device_type}")


def _benchmark_launch(launch: PreparedLaunch, comparison: PreparedComparison, warmup: int,
                      time_budget_ms: float | None = None) -> float:
    if comparison.device_type == "cpu" and comparison.cpu_host_timing:
        def invoke() -> float:
            if launch.prepare is not None:
                launch.prepare()
            begin = time.perf_counter_ns()
            launch.launch()
            return (time.perf_counter_ns() - begin) * 1e-6

        first = invoke()
        remaining_warmup_ms = warmup - first
        while remaining_warmup_ms > 0:
            first = invoke()
            remaining_warmup_ms -= first
        repetitions = min(50, max(1, int(10.0 / first)))
        return statistics.median(
            sum(invoke() for _ in range(repetitions)) / repetitions for _ in range(7)
        )
    if launch.native_benchmark is not None:
        return launch.native_benchmark()
    if comparison.device_type != "cuda":
        raise NotImplementedError("CPU benchmark requires a native repeat/timing entry")
    return benchmark(launch.launch, warmup=warmup, repetitions=MEASUREMENT_REPETITIONS,
                     cuda_graph=comparison.cuda_graph, prepare=launch.prepare, time_budget_ms=time_budget_ms)[0]


def evaluate(
    comparison: PreparedComparison,
    *,
    before_benchmark: Callable[[], None] | None = None,
    source_timing_error: Callable[[Exception], bool] | None = None,
    benchmark_time_budget_ms: float | None = None,
    source_time_ms: float | None = None,
    measure_source: bool = True,
) -> tuple[float | None, float | None]:
    run_only = comparison.status == "run_only"
    if run_only:
        if comparison.source is not None or comparison.tolerance is not None or comparison.native_comparison is not None:
            raise PipelineStageError("adapter_preparation", "run_only requires a generated launch without reference or tolerance")
    elif comparison.tolerance is None or (comparison.source is None and comparison.native_comparison is None):
        raise PipelineStageError("adapter_preparation", "comparison requires an explicit reference and tolerance")
    if comparison.device_type == "cpu" and before_benchmark is not None:
        before_benchmark()
    if comparison.native_comparison is not None:
        report_stage("native_benchmark")
        measured = comparison.native_comparison()
        report_stage("numerical_comparison")
        compare_outputs(measured.generated, measured.source, comparison.tolerance)
        return measured.generated_ms, measured.source_ms
    if comparison.generated is not None:
        report_stage("generated_launch")
        try:
            if comparison.generated.prepare is not None:
                comparison.generated.prepare()
            comparison.generated.launch()
            _synchronize(comparison)
        except Exception as error:
            raise PipelineStageError("generated_launch", str(error)) from error
    if run_only:
        if comparison.device_type != "cpu" and before_benchmark is not None:
            before_benchmark()
        report_stage("generated_benchmark")
        try:
            first = _benchmark_launch(comparison.generated, comparison, 25)
            second = _benchmark_launch(comparison.generated, comparison, 0)
        except Exception as error:
            raise PipelineStageError("generated_benchmark", str(error)) from error
        report_stage("generated_result_access")
        try:
            _result_structure(comparison.generated.outputs())
        except Exception as error:
            raise PipelineStageError("generated_result_access", str(error)) from error
        return statistics.median((first, second)), None
    report_stage("source_launch")
    try:
        if comparison.source.prepare is not None:
            comparison.source.prepare()
        comparison.source.launch()
        _synchronize(comparison)
    except Exception as error:
        raise PipelineStageError("source_launch", str(error)) from error

    def measure_reference() -> float | None:
        if source_time_ms is not None or not measure_source:
            return source_time_ms
        samples = []
        for stage, warmup in (("source_benchmark", 25), ("source_reverse_benchmark", 0)):
            report_stage(stage)
            try:
                samples.append(_benchmark_launch(comparison.source, comparison, warmup, benchmark_time_budget_ms))
            except Exception as error:
                if source_timing_error is not None and source_timing_error(error):
                    return None
                raise PipelineStageError(stage, str(error)) from error
        return statistics.median(samples)

    if comparison.generated is None:
        if comparison.device_type != "cpu" and before_benchmark is not None:
            before_benchmark()
        return None, measure_reference()

    def validate_outputs() -> None:
        report_stage("numerical_comparison")
        compare_outputs(
            comparison.generated.outputs(),
            comparison.source.outputs(),
            comparison.tolerance,
        )

    if comparison.device_type != "cpu" or comparison.status != "pass":
        validate_outputs()
    if comparison.status != "pass":
        return None, None
    if comparison.device_type != "cpu" and before_benchmark is not None:
        before_benchmark()
    report_stage("generated_benchmark")
    try:
        generated_first = _benchmark_launch(comparison.generated, comparison, 25, benchmark_time_budget_ms)
    except Exception as error:
        raise PipelineStageError("generated_benchmark", str(error)) from error
    source_ms = measure_reference()
    if source_ms is None and measure_source:
        # The candidate window and numerical comparison already finished.
        # Do not reuse a stream after an unsupported reference capture.
        return generated_first, None
    report_stage("generated_reverse_benchmark")
    try:
        generated_second = _benchmark_launch(comparison.generated, comparison, 0, benchmark_time_budget_ms)
    except Exception as error:
        raise PipelineStageError("generated_reverse_benchmark", str(error)) from error
    if comparison.device_type == "cpu":
        # Torch's comparison workers must not spin alongside native CPU timing.
        validate_outputs()
    return (
        statistics.median((generated_first, generated_second)),
        source_ms,
    )
