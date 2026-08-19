from __future__ import annotations

from collections.abc import Iterable

import torch

import intent
from repro.common.support import benchmark
from repro.common.support import prepare_kernel_call

from .model import Context
from .model import PreparedComparison
from .model import PreparedLaunch
from .model import TensorTree
from .model import Tolerance


class GeneratedCompilationError(RuntimeError):
    pass


class NumericalComparisonError(RuntimeError):
    pass


def compile_single(
    context: Context,
    definition,
    arguments: tuple[object, ...],
    *,
    constexprs: dict[str, object] | None = None,
) -> tuple[object, PreparedLaunch]:
    try:
        artifact = intent.compile(
            definition,
            target=context.target,
            compiler=context.compiler,
            constexprs=constexprs,
        )
        result = artifact.run(*arguments)
        launch_outputs = () if result is None else result
        launch = prepare_kernel_call(artifact, arguments, launch_outputs)
    except Exception as error:
        raise GeneratedCompilationError(str(error)) from error
    return artifact, PreparedLaunch(launch=launch, outputs=lambda: result)


def functional_launch(function) -> PreparedLaunch:
    state: dict[str, TensorTree] = {}

    def launch():
        state["output"] = function()
        return state["output"]

    launch()
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
    for generated_value, source_value, leaf_tolerance in zip(
        generated_values, source_values, tolerances
    ):
        if generated_value.shape != source_value.shape:
            raise NumericalComparisonError(
                "generated/source result shape differs: "
                f"{tuple(generated_value.shape)} != {tuple(source_value.shape)}"
            )
        if generated_value.dtype != source_value.dtype:
            raise NumericalComparisonError(
                "generated/source result dtype differs: "
                f"{generated_value.dtype} != {source_value.dtype}"
            )
        if generated_value.dtype == torch.bool or not generated_value.is_floating_point():
            if not torch.equal(generated_value, source_value):
                raise NumericalComparisonError(
                    "generated/source integer result differs"
                )
            errors.append(0.0)
            continue
        generated_compare = generated_value.float()
        source_compare = source_value.float()
        generated_finite = torch.isfinite(generated_compare)
        source_finite = torch.isfinite(source_compare)
        if not torch.equal(generated_finite, source_finite):
            raise NumericalComparisonError(
                "generated/source finite-value masks differ"
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
                    "generated/source non-finite values differ"
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
                    "generated/source floating result differs: "
                    f"max_abs={maximum}, atol={leaf_tolerance.atol}, "
                    f"rtol={leaf_tolerance.rtol}"
                )
            errors.append(maximum)
        else:
            errors.append(0.0)
    return tuple(errors)


def evaluate(comparison: PreparedComparison) -> tuple[float, float]:
    if comparison.generated.prepare is not None:
        comparison.generated.prepare()
    comparison.generated.launch()
    if comparison.source.prepare is not None:
        comparison.source.prepare()
    comparison.source.launch()
    torch.cuda.synchronize()
    compare_outputs(
        comparison.generated.outputs(),
        comparison.source.outputs(),
        comparison.tolerance,
    )
    generated_p50, _ = benchmark(
        comparison.generated.launch,
        warmup=3,
        repetitions=100,
        cuda_graph=comparison.cuda_graph,
        prepare=comparison.generated.prepare,
    )
    source_p50, _ = benchmark(
        comparison.source.launch,
        warmup=3,
        repetitions=100,
        cuda_graph=comparison.cuda_graph,
        prepare=comparison.source.prepare,
    )
    return generated_p50, source_p50
