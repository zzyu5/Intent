from __future__ import annotations

from collections.abc import Callable

import torch

import intent
from intent.targets.base import Target

from .support import benchmark
from .support import prepare_kernel_call
from .support import print_artifact


Runner = Callable[[str, Target, str], None]
TensorOutputs = torch.Tensor | tuple[torch.Tensor, ...]


def outputs(value: TensorOutputs) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def errors(actual: TensorOutputs, expected: TensorOutputs) -> tuple[float, ...]:
    actual_values = outputs(actual)
    expected_values = outputs(expected)
    if len(actual_values) != len(expected_values):
        raise RuntimeError("generated result count does not match the reference")
    result = []
    for generated, wanted in zip(actual_values, expected_values):
        if generated.shape != wanted.shape or generated.dtype != wanted.dtype:
            raise RuntimeError(
                "generated result shape or dtype does not match the reference: "
                f"got shape={tuple(generated.shape)}, dtype={generated.dtype}; "
                f"expected shape={tuple(wanted.shape)}, dtype={wanted.dtype}"
            )
        if generated.dtype == torch.bool or not generated.is_floating_point():
            result.append(0.0 if torch.equal(generated, wanted) else 1.0)
        else:
            generated_finite = torch.isfinite(generated)
            wanted_finite = torch.isfinite(wanted)
            if not torch.equal(generated_finite, wanted_finite):
                result.append(float("inf"))
                continue
            nonfinite_equal = torch.equal(
                generated[~generated_finite], wanted[~wanted_finite]
            )
            if not nonfinite_equal:
                result.append(float("inf"))
                continue
            finite_error = torch.where(
                generated_finite,
                (generated - wanted).abs(),
                torch.zeros((), device=generated.device, dtype=generated.dtype),
            )
            result.append(finite_error.max().item())
    return tuple(result)


def require_close(
    *,
    actual: TensorOutputs,
    expected: TensorOutputs,
    tolerance: float | tuple[float, ...],
    target_name: str,
    kernel_name: str,
) -> tuple[float, ...]:
    measured = errors(actual, expected)
    tolerances = (
        tolerance
        if isinstance(tolerance, tuple)
        else tuple(tolerance for _ in measured)
    )
    if len(tolerances) != len(measured):
        raise RuntimeError("tolerance count does not match the generated results")
    if any(error > limit for error, limit in zip(measured, tolerances)):
        raise RuntimeError(
            f"{target_name} {kernel_name} numerical comparison failed: {measured}"
        )
    return measured


def run_generated(
    *,
    definition,
    arguments: tuple[object, ...],
    reference: Callable[[], TensorOutputs],
    compiler: str,
    target: Target,
    target_name: str,
    kernel_name: str,
    tolerance: float | tuple[float, ...],
    constexprs: dict[str, object] | None = None,
    cuda_graph: bool = True,
) -> None:
    artifact = intent.compile(
        definition,
        target=target,
        compiler=compiler,
        constexprs=constexprs,
    )
    generated = artifact.run(*arguments)
    expected = reference()
    torch.cuda.synchronize()
    measured_errors = require_close(
        actual=generated,
        expected=expected,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=kernel_name,
    )
    generated_call = prepare_kernel_call(artifact, arguments, generated)
    p50, p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=cuda_graph,
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} {kernel_name} numerical comparison: PASS "
        f"(generated/reference={measured_errors})"
    )
    print(
        f"{target_name} {kernel_name} kernel-only performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} {kernel_name} upstream baseline: unavailable")


def report_pipeline(
    *,
    artifacts: tuple[object, ...],
    launch: Callable[[], None],
    errors: tuple[float, ...],
    target_name: str,
    kernel_name: str,
    cuda_graph: bool = False,
    prepare: Callable[[], object] | None = None,
    performance_scope: str = "end-to-end GPU pipeline",
) -> None:
    p50, p95 = benchmark(
        launch,
        warmup=3,
        repetitions=100,
        cuda_graph=cuda_graph,
        prepare=prepare,
    )
    for artifact in artifacts:
        print_artifact(artifact, target_name)
    print(
        f"{target_name} {kernel_name} numerical comparison: PASS "
        f"(generated/reference={errors})"
    )
    print(
        f"{target_name} {kernel_name} {performance_scope} performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} {kernel_name} upstream baseline: unavailable")


def report_variant_pipeline(
    *,
    variant_artifacts: tuple[object, ...],
    original_launch: Callable[[], object],
    variant_launch: Callable[[], object],
    original_prepare: Callable[[], object] | None,
    variant_prepare: Callable[[], object] | None,
    original_errors: tuple[float, ...],
    variant_errors: tuple[float, ...],
    pair_errors: tuple[float, ...],
    target_name: str,
    kernel_name: str,
    performance_scope: str,
    cuda_graph: bool = False,
) -> None:
    original_p50, original_p95 = benchmark(
        original_launch,
        warmup=3,
        repetitions=100,
        cuda_graph=cuda_graph,
        prepare=original_prepare,
    )
    variant_p50, variant_p95 = benchmark(
        variant_launch,
        warmup=3,
        repetitions=100,
        cuda_graph=cuda_graph,
        prepare=variant_prepare,
    )
    for artifact in variant_artifacts:
        print_artifact(artifact, target_name)
    print(
        f"{target_name} {kernel_name} equivalent decomposition comparison: PASS "
        f"(original/reference={original_errors}, "
        f"variant/reference={variant_errors}, variant/original={pair_errors})"
    )
    print(
        f"{target_name} {kernel_name} {performance_scope} performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"original_p50={original_p50:.4f} ms, "
        f"original_p95={original_p95:.4f} ms, "
        f"variant_p50={variant_p50:.4f} ms, "
        f"variant_p95={variant_p95:.4f} ms, "
        f"variant/original_p50={variant_p50 / original_p50:.4f}x"
    )


def compare_variant_outputs(
    *,
    original: TensorOutputs,
    variant: TensorOutputs,
    reference: TensorOutputs,
    tolerance: float | tuple[float, ...],
    target_name: str,
    kernel_name: str,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    original_errors = require_close(
        actual=original,
        expected=reference,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} original",
    )
    variant_errors = require_close(
        actual=variant,
        expected=reference,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} variant",
    )
    pair_errors = require_close(
        actual=variant,
        expected=original,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} equivalent pair",
    )
    return original_errors, variant_errors, pair_errors


def run_variant(
    *,
    original,
    variant,
    arguments: tuple[object, ...],
    reference: Callable[[], TensorOutputs],
    compiler: str,
    target: Target,
    target_name: str,
    kernel_name: str,
    tolerance: float | tuple[float, ...],
    constexprs: dict[str, object] | None = None,
    cuda_graph: bool = True,
) -> None:
    original_artifact = intent.compile(
        original,
        target=target,
        compiler=compiler,
        constexprs=constexprs,
    )
    variant_artifact = intent.compile(
        variant,
        target=target,
        compiler=compiler,
        constexprs=constexprs,
    )
    original_output = original_artifact.run(*arguments)
    variant_output = variant_artifact.run(*arguments)
    expected = reference()
    torch.cuda.synchronize()
    original_errors, variant_errors, pair_errors = compare_variant_outputs(
        original=original_output,
        variant=variant_output,
        reference=expected,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=kernel_name,
    )
    original_call = prepare_kernel_call(original_artifact, arguments, original_output)
    variant_call = prepare_kernel_call(variant_artifact, arguments, variant_output)
    report_variant_pipeline(
        variant_artifacts=(variant_artifact,),
        original_launch=original_call,
        variant_launch=variant_call,
        original_prepare=None,
        variant_prepare=None,
        original_errors=original_errors,
        variant_errors=variant_errors,
        pair_errors=pair_errors,
        target_name=target_name,
        kernel_name=kernel_name,
        performance_scope="kernel-only",
        cuda_graph=cuda_graph,
    )
