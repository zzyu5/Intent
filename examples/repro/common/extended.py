from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.contraction.batched_gemm import BATCH as BMM_BATCH
from kernels.contraction.batched_gemm import K as BMM_K
from kernels.contraction.batched_gemm import M as BMM_M
from kernels.contraction.batched_gemm import N as BMM_N
from kernels.contraction.batched_gemm import batched_gemm_nn
from kernels.contraction.batched_gemm import batched_gemm_nt
from kernels.contraction.batched_gemm import batched_gemm_tn
from kernels.contraction.batched_gemm import batched_gemm_tt
from kernels.backward.layer_norm import FEATURES as BWD_LAYER_FEATURES
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
from kernels.streaming.attention import HEAD_DIMENSION
from kernels.streaming.attention import SCALE
from kernels.streaming.attention import VARLEN_BATCH
from kernels.streaming.attention import VARLEN_TOTAL_TOKENS
from kernels.streaming.attention import flash_varlen_attention_fwd

from .support import benchmark
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
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(*arguments), warmup=3, repetitions=10
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
            lambda: upstream(arguments), warmup=3, repetitions=10
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} {kernel_name} numerical comparison: PASS "
        f"(generated/reference={error})"
    )
    print(
        f"{target_name} {kernel_name} generated performance: "
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
        lambda: artifact.run(*arguments), warmup=3, repetitions=10
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} fused quantized GEMM numerical comparison: PASS "
        f"(max_int8_error={error})"
    )
    print(
        f"{target_name} fused quantized GEMM generated performance: "
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
        lambda: artifact.run(dc, a, b), warmup=3, repetitions=10
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
            lambda: upstream((dc, a, b)), warmup=3, repetitions=10
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} SwiGLU backward numerical comparison: PASS "
        f"(da/db errors={errors})"
    )
    print(
        f"{target_name} SwiGLU backward generated performance: "
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

    def generated_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dw_partial, db_partial = rows_artifact.run(
            x, dy, weight, inverse_features, epsilon
        )
        dw, db = reduce_artifact.run(dw_partial, db_partial)
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
        generated_pipeline, warmup=3, repetitions=10
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
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream((x, dy, weight, bias, epsilon)),
            warmup=3,
            repetitions=10,
        )
    print_artifact(rows_artifact, target_name)
    print_artifact(reduce_artifact, target_name)
    print(
        f"{target_name} LayerNorm backward pipeline numerical comparison: PASS "
        f"(dx/dw/db errors={errors})"
    )
    print(
        f"{target_name} LayerNorm backward pipeline performance: "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} LayerNorm backward upstream baseline: unavailable")
    else:
        print(
            f"{target_name} LayerNorm backward upstream comparison: PASS "
            f"(dx/dw/db errors={upstream_errors}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
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
        )


EXTENDED_RUNNERS: dict[str, Runner] = {
    "batched_gemm": _run_batched_gemm,
    "bf16_gemm": _run_bf16_gemm,
    "dual_gemm": _run_dual_gemm,
    "grouped_gemm": _run_grouped_gemm,
    "layer_norm": _run_layer_norm,
    "layer_norm_backward": _run_layer_norm_backward,
    "logsumexp": _run_logsumexp,
    "online_softmax": _run_online_softmax,
    "quantized_gemm": _run_quantized_gemm,
    "rms_norm": _run_rms_norm,
    "swiglu_backward": _run_swiglu_backward,
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
