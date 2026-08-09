from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.contraction.dual_gemm import K as DUAL_K
from kernels.contraction.dual_gemm import M as DUAL_M
from kernels.contraction.dual_gemm import N as DUAL_N
from kernels.contraction.dual_gemm import gated_dual_gemm
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


Upstream = Callable[[tuple[object, ...]], torch.Tensor]
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
) -> None:
    generated = artifact.run(*arguments)
    expected = reference()
    upstream_output = upstream(arguments) if upstream is not None else None
    torch.cuda.synchronize()
    error = (generated - expected).abs().max().item()
    if error > tolerance:
        raise RuntimeError(
            f"{target_name} {kernel_name} numerical comparison failed: {error}"
        )
    generated_p50, generated_p95 = benchmark(
        lambda: artifact.run(*arguments), warmup=3, repetitions=10
    )
    if upstream_output is not None:
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
    x = torch.randn(
        (GROUPED_ROWS, GROUPED_K), device="cuda", dtype=torch.float16
    )
    x /= math.sqrt(GROUPED_K)
    weight = torch.randn(
        (GROUPS, GROUPED_K, GROUPED_N), device="cuda", dtype=torch.float16
    )
    offsets = torch.tensor(
        [
            0,
            GROUPED_ROWS // 32,
            GROUPED_ROWS // 16,
            GROUPED_ROWS // 8,
            GROUPED_ROWS // 4,
            GROUPED_ROWS // 2,
            3 * GROUPED_ROWS // 4,
            7 * GROUPED_ROWS // 8,
            GROUPED_ROWS,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    artifact = intent.compile(ragged_grouped_gemm, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        result = torch.empty(
            (GROUPED_ROWS, GROUPED_N), device="cuda", dtype=torch.float32
        )
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
        kernel_name="ragged grouped GEMM",
        tolerance=5.0e-2,
        upstream=upstream,
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
    artifact = intent.compile(
        flash_varlen_attention_fwd,
        constexprs={"CAUSAL": False},
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
                is_causal=False,
                scale=SCALE,
            )[0, 0]
        return result

    _compare(
        artifact=artifact,
        arguments=(q, k, v, lengths, cu_seqlens, SCALE),
        reference=reference,
        target_name=target_name,
        kernel_name="packed varlen attention",
        tolerance=2.0e-2,
        upstream=upstream,
    )


EXTENDED_RUNNERS: dict[str, Runner] = {
    "dual_gemm": _run_dual_gemm,
    "grouped_gemm": _run_grouped_gemm,
    "layer_norm": _run_layer_norm,
    "logsumexp": _run_logsumexp,
    "online_softmax": _run_online_softmax,
    "rms_norm": _run_rms_norm,
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
