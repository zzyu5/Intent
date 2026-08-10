from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.activation.swiglu import FEATURES as SWIGLU_FORWARD_FEATURES
from kernels.activation.swiglu import TOKENS as SWIGLU_FORWARD_TOKENS
from kernels.activation.swiglu import swiglu_forward
from kernels.contraction.batched_gemm import BATCH as BMM_BATCH
from kernels.contraction.batched_gemm import K as BMM_K
from kernels.contraction.batched_gemm import M as BMM_M
from kernels.contraction.batched_gemm import N as BMM_N
from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.batched_gemm import batched_gemm_nt
from kernels.contraction.batched_gemm import batched_gemm_tn
from kernels.contraction.batched_gemm import batched_gemm_tt
from kernels.backward.layer_norm import FEATURES as BWD_LAYER_FEATURES
from kernels.backward.layer_norm import PARTIAL_GROUPS as BWD_LAYER_PARTIAL_GROUPS
from kernels.backward.layer_norm import ROWS as BWD_LAYER_ROWS
from kernels.backward.layer_norm import layer_norm_backward_reduce
from kernels.backward.layer_norm import layer_norm_backward_rows
from kernels.backward.swiglu import FEATURES as SWIGLU_FEATURES
from kernels.backward.swiglu import TOKENS as SWIGLU_TOKENS
from kernels.backward.swiglu import swiglu_backward
from kernels.contraction.dual_gemm import K as DUAL_K
from kernels.contraction.dual_gemm import M as DUAL_M
from kernels.contraction.dual_gemm import N as DUAL_N
from kernels.contraction.dual_gemm import gated_dual_gemm
from kernels.contraction.gemm import K as GEMM_K
from kernels.contraction.gemm import M as GEMM_M
from kernels.contraction.gemm import N as GEMM_N
from kernels.contraction.gemm import bf16_gemm
from kernels.contraction.gemm import quantized_gemm
from kernels.indexing.relations import GQA_KEY_HEADS
from kernels.indexing.relations import GQA_QUERY_HEADS
from kernels.indexing.relations import GQA_TOKENS
from kernels.indexing.relations import LOOKUP_ENTRIES
from kernels.indexing.relations import LOOKUP_FEATURES
from kernels.indexing.relations import LOOKUP_ROWS
from kernels.indexing.relations import OFFSET_FEATURES
from kernels.indexing.relations import OFFSET_ROWS
from kernels.indexing.relations import grouped_query_head_add
from kernels.indexing.relations import scalar_table_lookup
from kernels.indexing.relations import shifted_row_copy
from kernels.normalization.fused_add_rms_norm import FEATURES as FUSED_RMS_FEATURES
from kernels.normalization.fused_add_rms_norm import ROWS as FUSED_RMS_ROWS
from kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from kernels.normalization.layer_norm import FEATURES as LAYER_FEATURES
from kernels.normalization.layer_norm import ROWS as LAYER_ROWS
from kernels.normalization.layer_norm import weighted_layer_norm
from kernels.normalization.logsumexp import COLUMNS as LSE_COLUMNS
from kernels.normalization.logsumexp import ROWS as LSE_ROWS
from kernels.normalization.logsumexp import row_logsumexp
from kernels.normalization.rms_norm import FEATURES as RMS_FEATURES
from kernels.normalization.rms_norm import ROWS as RMS_ROWS
from kernels.normalization.rms_norm import weighted_rms_norm
from kernels.ragged.grouped_gemm import GROUPS
from kernels.ragged.grouped_gemm import K as GROUPED_K
from kernels.ragged.grouped_gemm import N as GROUPED_N
from kernels.ragged.grouped_gemm import ROWS as GROUPED_ROWS
from kernels.ragged.grouped_gemm import ragged_grouped_gemm
from kernels.streaming.online_softmax import COLUMNS as ONLINE_COLUMNS
from kernels.streaming.online_softmax import ROWS as ONLINE_ROWS
from kernels.streaming.online_softmax import streamed_online_softmax
from kernels.streaming.attention import BATCH as ATTENTION_BATCH
from kernels.streaming.attention import HEADS as ATTENTION_HEADS
from kernels.streaming.attention import HEAD_DIMENSION
from kernels.streaming.attention import SCALE
from kernels.streaming.attention import SEQUENCE as ATTENTION_SEQUENCE
from kernels.streaming.attention import VARLEN_BATCH
from kernels.streaming.attention import VARLEN_TOTAL_TOKENS
from kernels.streaming.attention import flash_attention_bias_fwd
from kernels.streaming.attention import flash_varlen_attention_fwd

from .support import benchmark
from .support import prepare_kernel_call
from .support import print_artifact


Upstream = Callable[[tuple[object, ...]], object]
Runner = Callable[[str, Target, str, Upstream | None], None]


def _compare(
    *,
    artifact,
    arguments: tuple[object, ...],
    reference: Callable[[], torch.Tensor],
    target_name: str,
    kernel_name: str,
    tolerance: float,
    upstream: Upstream | None,
    expected_dtype: torch.dtype | None = None,
    measurement_scope: str = "kernel-only",
    cuda_graph: bool = True,
) -> None:
    generated = artifact.run(*arguments)
    expected = reference()
    upstream_output = upstream(arguments) if upstream is not None else None
    torch.cuda.synchronize()
    if expected_dtype is not None and generated.dtype != expected_dtype:
        raise RuntimeError(
            f"{target_name} {kernel_name} returned {generated.dtype}, "
            f"expected {expected_dtype}"
        )
    error = (generated - expected).abs().max().item()
    if error > tolerance:
        raise RuntimeError(
            f"{target_name} {kernel_name} numerical comparison failed: {error}"
        )
    generated_call = (
        prepare_kernel_call(artifact, arguments, generated)
        if measurement_scope != "end-to-end"
        else lambda: artifact.run(*arguments)
    )
    generated_p50, generated_p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=cuda_graph,
    )
    if upstream_output is not None:
        if expected_dtype is not None and upstream_output.dtype != expected_dtype:
            raise RuntimeError(
                f"{target_name} {kernel_name} upstream returned "
                f"{upstream_output.dtype}, expected {expected_dtype}"
            )
        upstream_error = (upstream_output - expected).abs().max().item()
        if upstream_error > tolerance:
            raise RuntimeError(
                f"{target_name} {kernel_name} upstream numerical comparison "
                f"failed: {upstream_error}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream(arguments),
            warmup=3,
            repetitions=100,
            cuda_graph=cuda_graph,
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} {kernel_name} numerical comparison: PASS "
        f"(generated/reference={error})"
    )
    print(
        f"{target_name} {kernel_name} {measurement_scope} performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} {kernel_name} upstream baseline: unavailable")
    else:
        print(
            f"{target_name} {kernel_name} upstream comparison: PASS "
            f"(upstream/reference={upstream_error}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
        )


def run_gemm_tail_case(artifact, target_name: str) -> None:
    m, k, n = 4093, 4080, 14320
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    a /= math.sqrt(k)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    _compare(
        artifact=artifact,
        arguments=(a, b),
        reference=lambda: torch.matmul(a, b),
        target_name=target_name,
        kernel_name="GEMM M/N/K tail",
        tolerance=2.0e-2,
        upstream=None,
    )


def _run_bf16_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    a = torch.randn((GEMM_M, GEMM_K), device="cuda", dtype=torch.bfloat16)
    a /= math.sqrt(GEMM_K)
    b = torch.randn((GEMM_K, GEMM_N), device="cuda", dtype=torch.bfloat16)
    artifact = intent.compile(bf16_gemm, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(a, b),
        reference=lambda: (a.float() @ b.float()).to(torch.bfloat16),
        target_name=target_name,
        kernel_name="BF16 GEMM",
        tolerance=5.0e-2,
        upstream=upstream,
        expected_dtype=torch.bfloat16,
    )


def _run_batched_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    logical_a = torch.randn(
        (BMM_BATCH, BMM_M, BMM_K), device="cuda", dtype=torch.bfloat16
    )
    logical_a /= math.sqrt(BMM_K)
    logical_b = torch.randn(
        (BMM_BATCH, BMM_K, BMM_N), device="cuda", dtype=torch.bfloat16
    )
    variants = (
        ("NN", False, False, batched_gemm_nn),
        ("TN", True, False, batched_gemm_tn),
        ("NT", False, True, batched_gemm_nt),
        ("TT", True, True, batched_gemm_tt),
    )
    for label, transpose_a, transpose_b, kernel in variants:
        a = (
            logical_a.transpose(-1, -2).contiguous()
            if transpose_a
            else logical_a
        )
        b = (
            logical_b.transpose(-1, -2).contiguous()
            if transpose_b
            else logical_b
        )
        artifact = intent.compile(kernel, target=target, compiler=compiler)
        _compare(
            artifact=artifact,
            arguments=(a, b),
            reference=lambda: torch.bmm(logical_a.float(), logical_b.float()).to(
                torch.bfloat16
            ),
            target_name=target_name,
            kernel_name=f"BF16 batched GEMM {label}",
            tolerance=5.0e-2,
            upstream=(
                (lambda arguments, ta=transpose_a, tb=transpose_b: upstream(
                    (*arguments, ta, tb)
                ))
                if upstream is not None
                else None
            ),
            expected_dtype=torch.bfloat16,
        )


def _run_quantized_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    a = torch.randn((GEMM_M, GEMM_K), device="cuda", dtype=torch.float16)
    a /= math.sqrt(GEMM_K)
    b = torch.randn((GEMM_K, GEMM_N), device="cuda", dtype=torch.float16)
    bias = torch.randn((GEMM_N,), device="cuda", dtype=torch.float32) * 0.25
    residual = (
        torch.randn((GEMM_M, GEMM_N), device="cuda", dtype=torch.float16) * 0.25
    )
    output_scale = torch.linspace(
        1.0 / 48.0,
        1.0 / 24.0,
        GEMM_N,
        device="cuda",
        dtype=torch.float32,
    )
    artifact = intent.compile(quantized_gemm, target=target, compiler=compiler)
    arguments = (a, b, bias, residual, output_scale)

    def reference() -> torch.Tensor:
        accumulator = a.float() @ b.float()
        activated = torch.relu(accumulator + bias)
        fused = activated + residual.float()
        return torch.clamp(fused / output_scale, -128.0, 127.0).to(torch.int8)

    generated = artifact.run(*arguments)
    if generated.dtype != torch.int8:
        raise RuntimeError(
            f"{target_name} quantized GEMM returned {generated.dtype}, expected int8"
        )
    expected = reference()
    error = (
        generated.to(torch.int16) - expected.to(torch.int16)
    ).abs().max().item()
    if error > 2:
        raise RuntimeError(
            f"{target_name} quantized GEMM numerical comparison failed: {error}"
        )
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, arguments, generated),
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} fused quantized GEMM numerical comparison: PASS "
        f"(max_int8_error={error})"
    )
    print(
        f"{target_name} fused quantized GEMM kernel-only performance "
        f"(CUDA Graph): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    print(f"{target_name} fused quantized GEMM upstream baseline: unavailable")


def _run_swiglu_backward(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    shape = (SWIGLU_TOKENS, SWIGLU_FEATURES)
    dc = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    a = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    b = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    artifact = intent.compile(swiglu_backward, target=target, compiler=compiler)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        dc_f32 = dc.float()
        a_f32 = a.float()
        b_f32 = b.float()
        sigmoid = torch.sigmoid(a_f32)
        silu = a_f32 * sigmoid
        da = dc_f32 * (silu * (1.0 - sigmoid) + sigmoid) * b_f32
        db = dc_f32 * silu
        return da.to(torch.bfloat16), db.to(torch.bfloat16)

    generated = artifact.run(dc, a, b)
    if not isinstance(generated, tuple) or len(generated) != 2:
        raise RuntimeError(f"{target_name} SwiGLU backward did not return two gradients")
    expected = reference()
    errors = tuple(
        (actual - wanted).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if any(value > 5.0e-2 for value in errors):
        raise RuntimeError(
            f"{target_name} SwiGLU backward numerical comparison failed: {errors}"
        )
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, (dc, a, b), generated),
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    upstream_output = upstream((dc, a, b)) if upstream is not None else None
    if upstream_output is not None:
        if not isinstance(upstream_output, tuple) or len(upstream_output) != 2:
            raise RuntimeError(
                f"{target_name} SwiGLU backward upstream did not return two gradients"
            )
        upstream_errors = tuple(
            (actual - wanted).abs().max().item()
            for actual, wanted in zip(upstream_output, expected)
        )
        if any(value > 5.0e-2 for value in upstream_errors):
            raise RuntimeError(
                f"{target_name} SwiGLU backward upstream comparison failed: "
                f"{upstream_errors}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream((dc, a, b)),
            warmup=3,
            repetitions=100,
            cuda_graph=True,
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} SwiGLU backward numerical comparison: PASS "
        f"(da/db errors={errors})"
    )
    print(
        f"{target_name} SwiGLU backward kernel-only performance "
        f"(CUDA Graph): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} SwiGLU backward upstream baseline: unavailable")
    else:
        print(
            f"{target_name} SwiGLU backward upstream comparison: PASS "
            f"(da/db errors={upstream_errors}, upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
        )


def _run_swiglu_forward(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    shape = (SWIGLU_FORWARD_TOKENS, SWIGLU_FORWARD_FEATURES)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    up = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    artifact = intent.compile(swiglu_forward, target=target, compiler=compiler)
    packed = torch.cat((gate, up), dim=1) if upstream is not None else None
    adapted_upstream = (
        (lambda _: upstream((gate, up, packed))) if upstream is not None else None
    )
    _compare(
        artifact=artifact,
        arguments=(gate, up),
        reference=lambda: (F.silu(gate.float()) * up.float()).to(torch.bfloat16),
        target_name=target_name,
        kernel_name="SwiGLU forward",
        tolerance=5.0e-2,
        upstream=adapted_upstream,
        expected_dtype=torch.bfloat16,
    )


def _run_layer_norm_backward(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    epsilon = 1.0e-5
    inverse_features = 1.0 / BWD_LAYER_FEATURES
    shape = (BWD_LAYER_ROWS, BWD_LAYER_FEATURES)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    dy = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.05
    weight = torch.randn(
        (BWD_LAYER_FEATURES,), device="cuda", dtype=torch.bfloat16
    )
    bias = torch.randn_like(weight)
    rows_artifact = intent.compile(
        layer_norm_backward_rows, target=target, compiler=compiler
    )
    reduce_artifact = intent.compile(
        layer_norm_backward_reduce, target=target, compiler=compiler
    )

    dx = torch.empty_like(x)
    partial_shape = (BWD_LAYER_PARTIAL_GROUPS, BWD_LAYER_FEATURES)
    dw_partial = torch.zeros(partial_shape, device="cuda", dtype=torch.float32)
    db_partial = torch.zeros_like(dw_partial)
    dw = torch.empty((BWD_LAYER_FEATURES,), device="cuda", dtype=torch.float32)
    db = torch.empty_like(dw)
    rows_call = prepare_kernel_call(
        rows_artifact,
        (x, dy, weight, dw_partial, db_partial, inverse_features, epsilon),
        dx,
    )
    reduce_call = prepare_kernel_call(
        reduce_artifact,
        (dw_partial, db_partial),
        (dw, db),
    )

    def generated_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dw_partial.zero_()
        db_partial.zero_()
        rows_call()
        reduce_call()
        return dx, dw, db

    x_f32 = x.float()
    dy_f32 = dy.float()
    weight_f32 = weight.float()
    mean = x_f32.mean(dim=1, keepdim=True)
    centered = x_f32 - mean
    rstd = torch.rsqrt(centered.square().mean(dim=1, keepdim=True) + epsilon)
    normalized = centered * rstd
    weighted_dy = weight_f32 * dy_f32
    expected = (
        (
            (
                weighted_dy
                - weighted_dy.mean(dim=1, keepdim=True)
                - normalized * (weighted_dy * normalized).mean(dim=1, keepdim=True)
            )
            * rstd
        ).to(torch.bfloat16),
        (dy_f32 * normalized).sum(dim=0),
        dy_f32.sum(dim=0),
    )
    generated = generated_pipeline()
    errors = tuple(
        (actual - wanted).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if any(value > 5.0e-2 for value in errors):
        raise RuntimeError(
            f"{target_name} LayerNorm backward numerical comparison failed: {errors}"
        )
    generated_p50, generated_p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
    )
    upstream_output = (
        upstream((x, dy, weight, bias, epsilon)) if upstream is not None else None
    )
    if upstream_output is not None:
        if not isinstance(upstream_output, tuple) or len(upstream_output) != 3:
            raise RuntimeError(
                f"{target_name} LayerNorm backward upstream did not return three gradients"
            )
        upstream_expected = (
            expected[0],
            expected[1].to(torch.bfloat16).float(),
            expected[2].to(torch.bfloat16).float(),
        )
        upstream_errors = tuple(
            (actual - wanted).abs().max().item()
            for actual, wanted in zip(upstream_output, upstream_expected)
        )
        if upstream_errors[0] > 5.0e-2 or any(
            value > 1.25e-1 for value in upstream_errors[1:]
        ):
            raise RuntimeError(
                f"{target_name} LayerNorm backward upstream comparison failed: "
                f"{upstream_errors}"
            )
    print_artifact(rows_artifact, target_name)
    print_artifact(reduce_artifact, target_name)
    print(
        f"{target_name} LayerNorm backward pipeline numerical comparison: PASS "
        f"(dx/dw/db errors={errors})"
    )
    print(
        f"{target_name} LayerNorm backward end-to-end performance "
        f"(CUDA Event): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} LayerNorm backward upstream baseline: unavailable")
    else:
        print(
            f"{target_name} LayerNorm backward upstream comparison: PASS "
            f"(dx/dw/db errors={upstream_errors}, timing=unavailable: "
            f"the autograd wrapper cannot expose or CUDA-Graph-capture its "
            f"inner kernels)"
        )


def _run_layer_norm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    epsilon = 1.0e-5
    x = torch.randn(
        (LAYER_ROWS, LAYER_FEATURES), device="cuda", dtype=torch.float32
    )
    weight = torch.randn((LAYER_FEATURES,), device="cuda", dtype=torch.float32)
    bias = torch.randn((LAYER_FEATURES,), device="cuda", dtype=torch.float32)
    artifact = intent.compile(
        weighted_layer_norm, target=target, compiler=compiler
    )
    _compare(
        artifact=artifact,
        arguments=(x, weight, bias, 1.0 / LAYER_FEATURES, epsilon),
        reference=lambda: F.layer_norm(
            x, (LAYER_FEATURES,), weight=weight, bias=bias, eps=epsilon
        ),
        target_name=target_name,
        kernel_name="weighted LayerNorm",
        tolerance=5.0e-5,
        upstream=upstream,
    )


def _run_rms_norm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    epsilon = 1.0e-6
    x = torch.randn((RMS_ROWS, RMS_FEATURES), device="cuda", dtype=torch.float32)
    weight = torch.randn((RMS_FEATURES,), device="cuda", dtype=torch.float32)
    artifact = intent.compile(weighted_rms_norm, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x, weight, 1.0 / RMS_FEATURES, epsilon),
        reference=lambda: x
        * torch.rsqrt(x.square().mean(dim=1, keepdim=True) + epsilon)
        * weight,
        target_name=target_name,
        kernel_name="weighted RMSNorm",
        tolerance=5.0e-5,
        upstream=upstream,
        measurement_scope=(
            "end-to-end"
            if target_name == "TileLang" and upstream is not None
            else "kernel-only"
        ),
    )


def _run_fused_add_rms_norm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    inverse_features = 1.0 / FUSED_RMS_FEATURES
    epsilon = 1.0e-6
    weight_offset = 1.0
    shape = (FUSED_RMS_ROWS, FUSED_RMS_FEATURES)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn(
        (FUSED_RMS_FEATURES,), device="cuda", dtype=torch.bfloat16
    )
    artifact = intent.compile(fused_add_rms_norm, target=target, compiler=compiler)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        summed = (x.float() + residual.float()).to(torch.bfloat16)
        summed_f32 = summed.float()
        inverse_rms = torch.rsqrt(
            summed_f32.square().sum(dim=1, keepdim=True) * inverse_features
            + epsilon
        )
        normalized = (
            summed_f32 * inverse_rms * (weight.float() + weight_offset)
        ).to(torch.bfloat16)
        return normalized, summed

    arguments = (x, residual, weight, inverse_features, epsilon, weight_offset)
    generated = artifact.run(*arguments)
    if not isinstance(generated, tuple) or len(generated) != 2:
        raise RuntimeError(
            f"{target_name} fused add RMSNorm did not return two outputs"
        )
    expected = reference()
    for actual, wanted in zip(generated, expected):
        if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
            raise RuntimeError(
                f"{target_name} fused add RMSNorm returned "
                f"shape={tuple(actual.shape)}, dtype={actual.dtype}; expected "
                f"shape={tuple(wanted.shape)}, dtype={wanted.dtype}"
            )
    errors = tuple(
        (actual - wanted).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if any(value > 5.0e-2 for value in errors):
        raise RuntimeError(
            f"{target_name} fused add RMSNorm numerical comparison failed: {errors}"
        )
    generated_p50, generated_p95 = benchmark(
        prepare_kernel_call(artifact, arguments, generated),
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    upstream_output = upstream(arguments) if upstream is not None else None
    if upstream_output is not None:
        if not isinstance(upstream_output, tuple) or len(upstream_output) != 2:
            raise RuntimeError(
                f"{target_name} fused add RMSNorm upstream did not return two outputs"
            )
        for actual, wanted in zip(upstream_output, expected):
            if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
                raise RuntimeError(
                    f"{target_name} fused add RMSNorm upstream returned "
                    f"shape={tuple(actual.shape)}, dtype={actual.dtype}; expected "
                    f"shape={tuple(wanted.shape)}, dtype={wanted.dtype}"
                )
        upstream_errors = tuple(
            (actual - wanted).abs().max().item()
            for actual, wanted in zip(upstream_output, expected)
        )
        if any(value > 5.0e-2 for value in upstream_errors):
            raise RuntimeError(
                f"{target_name} fused add RMSNorm upstream comparison failed: "
                f"{upstream_errors}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream(arguments),
            warmup=3,
            repetitions=100,
            cuda_graph=True,
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} fused add RMSNorm numerical comparison: PASS "
        f"(normalized/residual errors={errors})"
    )
    print(
        f"{target_name} fused add RMSNorm kernel-only performance "
        f"(CUDA Graph): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} fused add RMSNorm upstream baseline: unavailable")
    else:
        print(
            f"{target_name} fused add RMSNorm upstream comparison: PASS "
            f"(normalized/residual errors={upstream_errors}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
        )


def _run_logsumexp(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    x = torch.randn((LSE_ROWS, LSE_COLUMNS), device="cuda", dtype=torch.float32)
    artifact = intent.compile(row_logsumexp, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x,),
        reference=lambda: torch.logsumexp(x, dim=1),
        target_name=target_name,
        kernel_name="row logsumexp",
        tolerance=2.0e-5,
        upstream=upstream,
    )


def _run_dual_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    x = torch.randn((DUAL_M, DUAL_K), device="cuda", dtype=torch.float16)
    x /= math.sqrt(DUAL_K)
    gate_weight = torch.randn(
        (DUAL_K, DUAL_N), device="cuda", dtype=torch.float16
    )
    value_weight = torch.randn_like(gate_weight)
    artifact = intent.compile(gated_dual_gemm, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        gate = x.float() @ gate_weight.float()
        value = x.float() @ value_weight.float()
        return (torch.relu(gate) * value).half()

    _compare(
        artifact=artifact,
        arguments=(x, gate_weight, value_weight),
        reference=reference,
        target_name=target_name,
        kernel_name="gated dual GEMM",
        tolerance=5.0e-2,
        upstream=upstream,
        measurement_scope="end-to-end",
    )


def _run_grouped_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    artifact = intent.compile(ragged_grouped_gemm, target=target, compiler=compiler)

    def run_case(
        rows: int,
        k: int,
        n: int,
        *,
        kernel_name: str,
        case_upstream: Upstream | None,
    ) -> None:
        x = torch.randn((rows, k), device="cuda", dtype=torch.float16)
        x /= math.sqrt(k)
        weight = torch.randn(
            (GROUPS, k, n), device="cuda", dtype=torch.float16
        )
        offsets = torch.tensor(
            [
                0,
                rows // 32,
                rows // 16,
                rows // 8,
                rows // 4,
                rows // 2,
                3 * rows // 4,
                7 * rows // 8,
                rows,
            ],
            device="cuda",
            dtype=torch.int32,
        )

        def reference() -> torch.Tensor:
            result = torch.empty((rows, n), device="cuda", dtype=torch.float32)
            for group in range(GROUPS):
                begin = offsets[group]
                end = offsets[group + 1]
                result[begin:end] = x[begin:end].float() @ weight[group].float()
            return result

        _compare(
            artifact=artifact,
            arguments=(x, offsets, weight),
            reference=reference,
            target_name=target_name,
            kernel_name=kernel_name,
            tolerance=5.0e-2,
            upstream=case_upstream,
            measurement_scope="end-to-end",
            cuda_graph=False,
        )

    run_case(
        GROUPED_ROWS,
        GROUPED_K,
        GROUPED_N,
        kernel_name="ragged grouped GEMM",
        case_upstream=upstream,
    )
    run_case(
        8191,
        4080,
        4080,
        kernel_name="ragged grouped GEMM member/K/N tail",
        case_upstream=None,
    )


def _run_online_softmax(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    x = torch.randn(
        (ONLINE_ROWS, ONLINE_COLUMNS), device="cuda", dtype=torch.float32
    )
    artifact = intent.compile(
        streamed_online_softmax, target=target, compiler=compiler
    )
    _compare(
        artifact=artifact,
        arguments=(x,),
        reference=lambda: torch.softmax(x, dim=1),
        target_name=target_name,
        kernel_name="streamed online softmax",
        tolerance=1.0e-5,
        upstream=upstream,
    )


def _run_varlen_attention(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    lengths = torch.tensor(
        [4093, 3961, 3833, 3701, 3571, 3449, 3319, 3187],
        device="cuda",
        dtype=torch.int32,
    )
    if lengths.numel() != VARLEN_BATCH or lengths.sum().item() != VARLEN_TOTAL_TOKENS:
        raise RuntimeError("varlen attention shape constants do not match its input")
    cu_seqlens = torch.zeros(
        (VARLEN_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    cu_seqlens[1:] = lengths.cumsum(0)
    shape = (VARLEN_TOTAL_TOKENS, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    for causal in (False, True):
        artifact = intent.compile(
            flash_varlen_attention_fwd,
            constexprs={"CAUSAL": causal},
            target=target,
            compiler=compiler,
        )

        def reference() -> torch.Tensor:
            result = torch.empty_like(v)
            for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
                start = int(begin.item())
                stop = int(end.item())
                result[start:stop] = F.scaled_dot_product_attention(
                    q[start:stop][None, None],
                    k[start:stop][None, None],
                    v[start:stop][None, None],
                    is_causal=causal,
                    scale=SCALE,
                )[0, 0]
            return result

        _compare(
            artifact=artifact,
            arguments=(q, k, v, lengths, cu_seqlens, SCALE),
            reference=reference,
            target_name=target_name,
            kernel_name=f"packed varlen attention causal={causal}",
            tolerance=2.0e-2,
            upstream=upstream if causal else None,
            measurement_scope=(
                "kernel-only" if target_name == "TileLang" else "runtime-launch"
            ),
            cuda_graph=target_name == "TileLang",
        )


def _run_attention_bias(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    shape = (
        ATTENTION_BATCH,
        ATTENTION_HEADS,
        ATTENTION_SEQUENCE,
        HEAD_DIMENSION,
    )
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    bias = torch.randn(
        (ATTENTION_BATCH, ATTENTION_HEADS, ATTENTION_SEQUENCE),
        device="cuda",
        dtype=torch.float32,
    ) * 0.125
    artifact = intent.compile(
        flash_attention_bias_fwd, target=target, compiler=compiler
    )
    arguments = (q, k, v, bias, SCALE)
    source_arguments = (
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        bias[:, :, None, :],
        SCALE,
    )
    adapted_upstream = (
        (lambda _: upstream(source_arguments)) if upstream is not None else None
    )
    _compare(
        artifact=artifact,
        arguments=arguments,
        reference=lambda: F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=bias[:, :, None, :],
            scale=SCALE,
        ).to(torch.float16),
        target_name=target_name,
        kernel_name="vector-bias attention",
        tolerance=2.0e-2,
        upstream=adapted_upstream,
        expected_dtype=torch.float16,
    )


def _run_shifted_row_copy(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("shifted row copy has no upstream adapter")
    x = torch.randn(
        (OFFSET_ROWS, OFFSET_FEATURES), device="cuda", dtype=torch.float32
    )
    artifact = intent.compile(shifted_row_copy, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x,),
        reference=lambda: torch.cat((x[1:], torch.zeros_like(x[:1])), dim=0),
        target_name=target_name,
        kernel_name="derived scalar offset index",
        tolerance=0.0,
        upstream=None,
    )


def _run_grouped_query_head_add(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("grouped query head mapping has no upstream adapter")
    query = torch.randn(
        (GQA_QUERY_HEADS, GQA_TOKENS),
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        (GQA_KEY_HEADS, GQA_TOKENS),
        device="cuda",
        dtype=torch.float16,
    )
    artifact = intent.compile(
        grouped_query_head_add, target=target, compiler=compiler
    )
    key_heads = torch.arange(GQA_QUERY_HEADS, device="cuda") // (
        GQA_QUERY_HEADS // GQA_KEY_HEADS
    )
    _compare(
        artifact=artifact,
        arguments=(query, key),
        reference=lambda: query + key[key_heads, :],
        target_name=target_name,
        kernel_name="many-to-one query head mapping",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.float16,
    )


def _run_scalar_table_lookup(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("scalar table lookup has no upstream adapter")
    labels = torch.randint(
        0,
        LOOKUP_ENTRIES,
        (LOOKUP_ROWS,),
        device="cuda",
        dtype=torch.int32,
    )
    table = torch.randn(
        (LOOKUP_ENTRIES, LOOKUP_FEATURES), device="cuda", dtype=torch.float32
    )
    artifact = intent.compile(scalar_table_lookup, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(labels, table),
        reference=lambda: table[labels.long()],
        target_name=target_name,
        kernel_name="tensor-derived scalar index",
        tolerance=0.0,
        upstream=None,
    )


EXTENDED_RUNNERS: dict[str, Runner] = {
    "attention_bias": _run_attention_bias,
    "batched_gemm": _run_batched_gemm,
    "bf16_gemm": _run_bf16_gemm,
    "dual_gemm": _run_dual_gemm,
    "fused_add_rms_norm": _run_fused_add_rms_norm,
    "grouped_gemm": _run_grouped_gemm,
    "grouped_query_head_add": _run_grouped_query_head_add,
    "layer_norm": _run_layer_norm,
    "layer_norm_backward": _run_layer_norm_backward,
    "logsumexp": _run_logsumexp,
    "online_softmax": _run_online_softmax,
    "quantized_gemm": _run_quantized_gemm,
    "rms_norm": _run_rms_norm,
    "scalar_table_lookup": _run_scalar_table_lookup,
    "shifted_row_copy": _run_shifted_row_copy,
    "swiglu_backward": _run_swiglu_backward,
    "swiglu_forward": _run_swiglu_forward,
    "varlen_attention": _run_varlen_attention,
}


def run_extended(
    kernel_name: str,
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    EXTENDED_RUNNERS[kernel_name](compiler, target, target_name, upstream)
