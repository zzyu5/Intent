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
from kernels.backward.embedding import FEATURES as EMBEDDING_FEATURES
from kernels.backward.embedding import TOKENS as EMBEDDING_TOKENS
from kernels.backward.embedding import VOCABULARY as EMBEDDING_VOCABULARY
from kernels.backward.embedding import embedding_backward_atomic
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
from kernels.convolution.direct import CONV1D_BATCH
from kernels.convolution.direct import CONV1D_FILTER
from kernels.convolution.direct import CONV1D_LENGTH
from kernels.convolution.direct import CONV2D_BATCH
from kernels.convolution.direct import CONV2D_FILTER_HEIGHT
from kernels.convolution.direct import CONV2D_FILTER_WIDTH
from kernels.convolution.direct import CONV2D_HEIGHT
from kernels.convolution.direct import CONV2D_WIDTH
from kernels.convolution.direct import conv1d_same
from kernels.convolution.direct import conv2d_same
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
from kernels.loss.cross_entropy import IGNORE_INDEX as CROSS_ENTROPY_IGNORE_INDEX
from kernels.loss.cross_entropy import TOKENS as CROSS_ENTROPY_TOKENS
from kernels.loss.cross_entropy import VOCABULARY as CROSS_ENTROPY_VOCABULARY
from kernels.loss.cross_entropy import cross_entropy_backward
from kernels.loss.cross_entropy import cross_entropy_forward
from kernels.normalization.fused_add_rms_norm import FEATURES as FUSED_RMS_FEATURES
from kernels.normalization.fused_add_rms_norm import ROWS as FUSED_RMS_ROWS
from kernels.normalization.fused_add_rms_norm import fused_add_rms_norm
from kernels.normalization.dropout_residual_rms_norm import (
    FEATURES as DROPOUT_RMS_FEATURES,
)
from kernels.normalization.dropout_residual_rms_norm import (
    KEEP_PROBABILITY as DROPOUT_KEEP_PROBABILITY,
)
from kernels.normalization.dropout_residual_rms_norm import (
    ROWS as DROPOUT_RMS_ROWS,
)
from kernels.normalization.dropout_residual_rms_norm import SEED as DROPOUT_SEED
from kernels.normalization.dropout_residual_rms_norm import (
    dropout_residual_rms_norm_backward_data,
)
from kernels.normalization.dropout_residual_rms_norm import (
    dropout_residual_rms_norm_forward,
)
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
from kernels.sampling.nucleus import CANDIDATES as NUCLEUS_CANDIDATES
from kernels.sampling.nucleus import ROWS as NUCLEUS_ROWS
from kernels.sampling.nucleus import THRESHOLD as NUCLEUS_THRESHOLD
from kernels.sampling.nucleus import sorted_nucleus_cutoff
from kernels.sampling.top_k import ROWS as TOP_K_ROWS
from kernels.sampling.top_k import TOP_K
from kernels.sampling.top_k import VOCABULARY as TOP_K_VOCABULARY
from kernels.sampling.top_k import insertion_top_k
from kernels.streaming.online_softmax import COLUMNS as ONLINE_COLUMNS
from kernels.streaming.online_softmax import ROWS as ONLINE_ROWS
from kernels.streaming.online_softmax import streamed_online_softmax
from kernels.streaming.attention import BATCH as ATTENTION_BATCH
from kernels.streaming.attention import HEADS as ATTENTION_HEADS
from kernels.streaming.attention import HEAD_DIMENSION
from kernels.streaming.attention import SCALE
from kernels.streaming.attention import SEQUENCE as ATTENTION_SEQUENCE
from kernels.streaming.attention import VARLEN_BATCH
from kernels.streaming.attention import VARLEN_GQA_HEAD_GROUP
from kernels.streaming.attention import VARLEN_GQA_KV_HEADS
from kernels.streaming.attention import VARLEN_GQA_QUERY_HEADS
from kernels.streaming.attention import VARLEN_GQA_TOTAL_TOKENS
from kernels.streaming.attention import VARLEN_TOTAL_TOKENS
from kernels.streaming.attention import flash_attention_bias_fwd
from kernels.streaming.attention import flash_varlen_attention_fwd
from kernels.streaming.attention import flash_varlen_gqa_prefill
from kernels.position.rope import rotary_embedding_flat
from kernels.streaming.paged_attention import BATCH as PAGED_BATCH
from kernels.streaming.paged_attention import HEAD_DIMENSION as PAGED_HEAD_DIMENSION
from kernels.streaming.paged_attention import HEAD_GROUP as PAGED_HEAD_GROUP
from kernels.streaming.paged_attention import KV_HEADS as PAGED_KV_HEADS
from kernels.streaming.paged_attention import PAGE_SIZE as PAGED_PAGE_SIZE
from kernels.streaming.paged_attention import QUERY_HEADS as PAGED_QUERY_HEADS
from kernels.streaming.paged_attention import SCALE as PAGED_SCALE
from kernels.streaming.paged_attention import SEQUENCE_LENGTHS as PAGED_SEQUENCE_LENGTHS
from kernels.streaming.paged_attention import paged_gqa_decode_attention
from kernels.streaming.selective_scan import BATCH as SELECTIVE_SCAN_BATCH
from kernels.streaming.selective_scan import LENGTH as SELECTIVE_SCAN_LENGTH
from kernels.streaming.selective_scan import selective_state_scan
from kernels.contraction.weight_only_int4 import GROUP_SIZE as W4_GROUP_SIZE
from kernels.contraction.weight_only_int4 import K as W4_K
from kernels.contraction.weight_only_int4 import M as W4_M
from kernels.contraction.weight_only_int4 import N as W4_N
from kernels.contraction.weight_only_int4 import PACK_FACTOR as W4_PACK_FACTOR
from kernels.contraction.weight_only_int4 import weight_only_int4_matmul

from .support import benchmark
from .support import prepare_kernel_call
from .support import print_artifact


ROPE_HALF_DIMENSION = HEAD_DIMENSION // 2


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
    if measurement_scope not in {
        "kernel-only",
        "end-to-end",
        "runtime-metadata",
    }:
        raise RuntimeError(f"unknown measurement scope: {measurement_scope}")
    if measurement_scope != "kernel-only" and cuda_graph:
        raise RuntimeError(
            f"{measurement_scope} measurement cannot use CUDA Graph replay"
        )
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
        if measurement_scope == "kernel-only"
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


def _run_conv1d(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    x = torch.randn(
        (CONV1D_BATCH, CONV1D_LENGTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    weight = torch.randn(
        (CONV1D_FILTER,),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    artifact = intent.compile(conv1d_same, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x, weight),
        reference=lambda: F.conv1d(
            x[:, None, :],
            weight[None, None, :],
            padding=CONV1D_FILTER // 2,
        )[:, 0, :],
        target_name=target_name,
        kernel_name="conv1d same-padding",
        tolerance=2.0e-3,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_conv2d(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    x = torch.randn(
        (CONV2D_BATCH, CONV2D_HEIGHT, CONV2D_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    weight = torch.randn(
        (CONV2D_FILTER_HEIGHT, CONV2D_FILTER_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    artifact = intent.compile(conv2d_same, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x, weight),
        reference=lambda: F.conv2d(
            x[:, None, :, :],
            weight[None, None, :, :],
            padding=(CONV2D_FILTER_HEIGHT // 2, CONV2D_FILTER_WIDTH // 2),
        )[:, 0, :, :],
        target_name=target_name,
        kernel_name="conv2d same-padding",
        tolerance=3.0e-3,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_selective_scan(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    x = torch.randn(
        (SELECTIVE_SCAN_BATCH, SELECTIVE_SCAN_LENGTH),
        device="cuda",
        dtype=torch.float32,
    ) * 0.05
    decay = 0.9 + 0.09 * torch.rand_like(x)
    drive = torch.randn_like(x) * 0.05
    artifact = intent.compile(selective_state_scan, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        state = torch.zeros(
            (SELECTIVE_SCAN_BATCH,), device="cuda", dtype=torch.float32
        )
        expected = torch.empty_like(x)
        for position in range(SELECTIVE_SCAN_LENGTH):
            state = (
                decay[:, position] * state
                + drive[:, position] * x[:, position]
            )
            expected[:, position] = state
        return expected

    _compare(
        artifact=artifact,
        arguments=(x, decay, drive),
        reference=reference,
        target_name=target_name,
        kernel_name="selective state scan",
        tolerance=2.0e-5,
        upstream=upstream,
        expected_dtype=torch.float32,
        cuda_graph=False,
    )


def _run_weight_only_int4(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    activation = torch.randn(
        (W4_M, W4_K), device="cuda", dtype=torch.float16
    ) * 0.05
    quantized = torch.randint(
        0,
        16,
        (W4_K, W4_N),
        device="cuda",
        dtype=torch.int32,
    )
    packed = torch.zeros(
        (W4_K // W4_PACK_FACTOR, W4_N),
        device="cuda",
        dtype=torch.int32,
    )
    unsigned = quantized
    for lane in range(W4_PACK_FACTOR):
        packed |= unsigned[lane::W4_PACK_FACTOR] << (4 * lane)
    scales = (
        0.01
        + 0.02
        * torch.rand(
            (W4_K // W4_GROUP_SIZE, W4_N),
            device="cuda",
            dtype=torch.float16,
        )
    )
    artifact = intent.compile(
        weight_only_int4_matmul,
        target=target,
        compiler=compiler,
    )

    def reference() -> torch.Tensor:
        dequantized = quantized.float() * scales.float().repeat_interleave(
            W4_GROUP_SIZE,
            dim=0,
        )
        return (activation.float() @ dequantized).to(torch.float16)

    _compare(
        artifact=artifact,
        arguments=(activation, packed, scales),
        reference=reference,
        target_name=target_name,
        kernel_name="W4A16 groupwise matmul",
        tolerance=4.0e-2,
        upstream=upstream,
        expected_dtype=torch.float16,
        cuda_graph=False,
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
    one = torch.randn((1, 1), device="cuda", dtype=torch.float16)
    generated = artifact.run(one, one)
    expected = one @ one
    torch.cuda.synchronize()
    error = (generated - expected).abs().max().item()
    if error > 2.0e-2:
        raise RuntimeError(
            f"{target_name} 1x1x1 GEMM numerical comparison failed: {error}"
        )
    print(
        f"{target_name} 1x1x1 GEMM numerical comparison: PASS "
        f"(error={error})"
    )
    zero_extent_cases = (
        (torch.empty((1, 0), device="cuda", dtype=torch.float16),
         torch.empty((0, 1), device="cuda", dtype=torch.float16), "K=0"),
        (torch.empty((0, 1), device="cuda", dtype=torch.float16),
         torch.empty((1, 1), device="cuda", dtype=torch.float16), "M=0"),
        (torch.empty((1, 1), device="cuda", dtype=torch.float16),
         torch.empty((1, 0), device="cuda", dtype=torch.float16), "N=0"),
    )
    for lhs, rhs, label in zero_extent_cases:
        try:
            artifact.run(lhs, rhs)
        except NotImplementedError as error:
            if "zero-extent external views" not in str(error):
                raise
        else:
            raise RuntimeError(f"{target_name} GEMM {label} was not rejected")
    print(f"{target_name} GEMM zero-extent capability checks: PASS")


def run_wide_attention_index_case(artifact, target_name: str) -> None:
    wide_stride = 1 << 31
    storage = torch.empty(
        (wide_stride + 64,), device="cuda", dtype=torch.float16
    )
    query = torch.as_strided(
        storage, (2, 1, 1, 64), (wide_stride, 64, 64, 1)
    )
    query[0] = torch.linspace(
        -0.5, 0.5, 64, device="cuda", dtype=torch.float16
    )
    query[1] = torch.linspace(
        0.75, -0.25, 64, device="cuda", dtype=torch.float16
    )
    key = torch.randn((2, 1, 2, 64), device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    scale = 1.0 / math.sqrt(64)
    if target_name in {"cuTile", "TileLang"}:
        try:
            artifact.run(query, key, value, scale)
        except NotImplementedError as error:
            if "64-bit external-buffer address" not in str(error):
                raise
            print(
                f"{target_name} >32-bit element-offset capability check: PASS "
                f"(offset={wide_stride}, explicitly unsupported)"
            )
            return
        raise RuntimeError(
            f"{target_name} accepted an external address beyond its declared "
            "32-bit surface capability"
        )
    generated = artifact.run(query, key, value, scale)
    expected = F.scaled_dot_product_attention(
        query, key, value, is_causal=False, scale=scale
    )
    torch.cuda.synchronize()
    error = (generated - expected).abs().max().item()
    if error > 2.0e-2:
        raise RuntimeError(
            f"{target_name} >32-bit element-offset comparison failed: {error}"
        )
    print(
        f"{target_name} >32-bit element-offset numerical comparison: PASS "
        f"(offset={wide_stride}, error={error})"
    )


def run_attention_shape_cases(artifact, target_name: str) -> None:
    cases = [(127, 131, dimension) for dimension in (64, 80, 96, 256)]
    cases.append((1, 1, 80))
    for query_length, key_length, dimension in cases:
        query = torch.randn(
            (1, 1, query_length, dimension),
            device="cuda",
            dtype=torch.float16,
        ) * 0.5
        key = torch.randn(
            (1, 1, key_length, dimension),
            device="cuda",
            dtype=torch.float16,
        ) * 0.5
        value = torch.randn_like(key) * 0.5
        scale = 1.0 / math.sqrt(dimension)
        generated = artifact.run(query, key, value, scale)
        expected = F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            scale=scale,
        )
        torch.cuda.synchronize()
        if generated.shape != expected.shape or generated.dtype != expected.dtype:
            raise RuntimeError(
                f"{target_name} attention Q={query_length}, K={key_length}, "
                f"D={dimension} returned shape={tuple(generated.shape)}, "
                f"dtype={generated.dtype}; expected shape={tuple(expected.shape)}, "
                f"dtype={expected.dtype}"
            )
        error = (generated - expected).abs().max().item()
        if not torch.isfinite(generated).all().item() or error > 2.0e-2:
            raise RuntimeError(
                f"{target_name} attention Q={query_length}, K={key_length}, "
                f"D={dimension} failed: {error}"
            )
        print(
            f"{target_name} attention Q={query_length}, K={key_length}, "
            f"D={dimension}: PASS (error={error})"
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


def _run_cross_entropy(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    logits = torch.randn(
        (CROSS_ENTROPY_TOKENS, CROSS_ENTROPY_VOCABULARY),
        device="cuda",
        dtype=torch.float32,
    ) * 0.5
    labels = torch.randint(
        0,
        CROSS_ENTROPY_VOCABULARY,
        (CROSS_ENTROPY_TOKENS,),
        device="cuda",
        dtype=torch.int32,
    )
    labels[::17] = CROSS_ENTROPY_IGNORE_INDEX
    dloss = torch.ones(
        (CROSS_ENTROPY_TOKENS,), device="cuda", dtype=torch.float32
    )
    forward_artifact = intent.compile(
        cross_entropy_forward,
        constexprs={
            "IGNORE_INDEX": CROSS_ENTROPY_IGNORE_INDEX,
            "VOCABULARY": CROSS_ENTROPY_VOCABULARY,
        },
        target=target,
        compiler=compiler,
    )
    backward_artifact = intent.compile(
        cross_entropy_backward,
        constexprs={"IGNORE_INDEX": CROSS_ENTROPY_IGNORE_INDEX},
        target=target,
        compiler=compiler,
    )

    expected_loss = F.cross_entropy(
        logits,
        labels.long(),
        reduction="none",
        ignore_index=CROSS_ENTROPY_IGNORE_INDEX,
    )
    expected_prediction = logits.argmax(dim=1)
    expected_prediction = torch.where(
        labels == CROSS_ENTROPY_IGNORE_INDEX,
        -1,
        expected_prediction,
    )
    valid = labels != CROSS_ENTROPY_IGNORE_INDEX
    safe_labels = torch.where(valid, labels, 0).long()
    expected_gradient = torch.softmax(logits, dim=1)
    expected_gradient[
        torch.arange(CROSS_ENTROPY_TOKENS, device="cuda"), safe_labels
    ] -= valid.float()
    expected_gradient *= valid[:, None].float() * dloss[:, None]

    generated_forward = forward_artifact.run(logits, labels)
    if not isinstance(generated_forward, tuple) or len(generated_forward) != 2:
        raise RuntimeError(
            f"{target_name} cross entropy forward did not return two outputs"
        )
    generated_loss, generated_prediction = generated_forward
    generated_gradient = backward_artifact.run(logits, labels, dloss)
    loss_error = (generated_loss - expected_loss).abs().max().item()
    prediction_matches = torch.equal(
        generated_prediction.long(), expected_prediction.long()
    )
    gradient_error = (generated_gradient - expected_gradient).abs().max().item()
    if loss_error > 2.0e-4 or not prediction_matches or gradient_error > 2.0e-5:
        raise RuntimeError(
            f"{target_name} cross entropy comparison failed: "
            f"loss={loss_error}, prediction={prediction_matches}, "
            f"gradient={gradient_error}"
        )

    forward_call = prepare_kernel_call(
        forward_artifact, (logits, labels), generated_forward
    )
    backward_call = prepare_kernel_call(
        backward_artifact, (logits, labels, dloss), generated_gradient
    )

    def generated_pipeline():
        forward_call()
        backward_call()

    generated_p50, generated_p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    upstream_output = upstream((logits, labels, dloss)) if upstream else None
    if upstream_output is not None:
        if not isinstance(upstream_output, tuple) or len(upstream_output) != 3:
            raise RuntimeError(
                f"{target_name} cross entropy upstream did not return three outputs"
            )
        upstream_loss, upstream_prediction, upstream_gradient = upstream_output
        upstream_errors = (
            (upstream_loss.float()[valid] - expected_loss[valid])
            .abs()
            .max()
            .item(),
            torch.equal(upstream_prediction.long(), expected_prediction.long()),
            (upstream_gradient.float() - expected_gradient).abs().max().item(),
        )
        if upstream_errors[0] > 2.0e-4 or not upstream_errors[1] or upstream_errors[2] > 2.0e-5:
            raise RuntimeError(
                f"{target_name} cross entropy upstream comparison failed: "
                f"{upstream_errors}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream((logits, labels, dloss)),
            warmup=3,
            repetitions=100,
            cuda_graph=False,
        )
    print_artifact(forward_artifact, target_name)
    print_artifact(backward_artifact, target_name)
    print(
        f"{target_name} cross entropy forward/backward numerical comparison: PASS "
        f"(loss={loss_error}, prediction={prediction_matches}, "
        f"gradient={gradient_error})"
    )
    print(
        f"{target_name} cross entropy end-to-end performance (CUDA Event): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_output is None:
        print(f"{target_name} cross entropy upstream baseline: unavailable")
    else:
        print(
            f"{target_name} cross entropy upstream comparison: PASS "
            f"(loss/prediction/gradient={upstream_errors}, "
            f"upstream_p50={upstream_p50:.4f} ms, upstream_p95={upstream_p95:.4f} ms, "
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
        cuda_graph=False,
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
            repetitions=100,
            cuda_graph=False,
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
            f"(dx/dw/db errors={upstream_errors}, "
            f"end-to-end upstream_p50={upstream_p50:.4f} ms, "
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
        measurement_scope=(
            "end-to-end"
            if target_name == "TileLang" and upstream is not None
            else "kernel-only"
        ),
        cuda_graph=not (target_name == "TileLang" and upstream is not None),
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


def _run_dropout_residual_rms_norm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("dropout residual RMSNorm has no algorithm-matched baseline")
    inverse_features = 1.0 / DROPOUT_RMS_FEATURES
    inverse_keep_probability = 1.0 / DROPOUT_KEEP_PROBABILITY
    epsilon = 1.0e-6
    weight_offset = 1.0
    shape = (DROPOUT_RMS_ROWS, DROPOUT_RMS_FEATURES)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    residual = torch.randn_like(x) * 0.5
    weight = torch.randn(
        (DROPOUT_RMS_FEATURES,), device="cuda", dtype=torch.bfloat16
    )
    dnormalized = torch.randn_like(x) * 0.25
    dresidual_out = torch.randn_like(x) * 0.25
    forward = intent.compile(
        dropout_residual_rms_norm_forward,
        target=target,
        compiler=compiler,
    )
    backward = intent.compile(
        dropout_residual_rms_norm_backward_data,
        target=target,
        compiler=compiler,
    )

    counters = torch.arange(
        DROPOUT_RMS_ROWS * DROPOUT_RMS_FEATURES,
        device="cuda",
        dtype=torch.int64,
    ).reshape(shape)
    bit_mask = (1 << 32) - 1
    random_bits = (counters ^ DROPOUT_SEED ^ 1831565813) & bit_mask
    random_bits = (random_bits ^ (random_bits << 13)) & bit_mask
    random_bits = (random_bits ^ (random_bits >> 17)) & bit_mask
    random_bits = (random_bits ^ (random_bits << 5)) & bit_mask
    uniform = (random_bits >> 8).float() * 5.960464477539063e-08
    keep = uniform < DROPOUT_KEEP_PROBABILITY
    keep_f32 = keep.float()
    summed = (
        x.float() * keep_f32 / DROPOUT_KEEP_PROBABILITY + residual.float()
    ).to(torch.bfloat16)
    summed_f32 = summed.float()
    inverse_rms = torch.rsqrt(
        summed_f32.square().sum(dim=1, keepdim=True) * inverse_features + epsilon
    )
    expected_normalized = (
        summed_f32 * inverse_rms * (weight.float() + weight_offset)
    ).to(torch.bfloat16)
    normalized_gradient = dnormalized.float() * (weight.float() + weight_offset)
    projection = (
        (normalized_gradient * summed_f32).sum(dim=1, keepdim=True)
        * inverse_features
    )
    summed_gradient = (
        normalized_gradient * inverse_rms
        - summed_f32 * inverse_rms.pow(3) * projection
        + dresidual_out.float()
    )
    expected_dx = (
        summed_gradient * keep_f32 / DROPOUT_KEEP_PROBABILITY
    ).to(torch.bfloat16)
    expected_dresidual = summed_gradient.to(torch.bfloat16)

    forward_arguments = (
        x,
        residual,
        weight,
        DROPOUT_SEED,
        DROPOUT_KEEP_PROBABILITY,
        inverse_keep_probability,
        inverse_features,
        epsilon,
        weight_offset,
    )
    backward_arguments = (
        x,
        residual,
        weight,
        dnormalized,
        dresidual_out,
        DROPOUT_SEED,
        DROPOUT_KEEP_PROBABILITY,
        inverse_keep_probability,
        inverse_features,
        epsilon,
        weight_offset,
    )
    generated_forward = forward.run(*forward_arguments)
    generated_backward = backward.run(*backward_arguments)
    if not isinstance(generated_forward, tuple) or len(generated_forward) != 2:
        raise RuntimeError(
            f"{target_name} dropout residual RMSNorm forward returned wrong ABI"
        )
    if not isinstance(generated_backward, tuple) or len(generated_backward) != 2:
        raise RuntimeError(
            f"{target_name} dropout residual RMSNorm backward returned wrong ABI"
        )
    expected = (
        expected_normalized,
        summed,
        expected_dx,
        expected_dresidual,
    )
    actual = (*generated_forward, *generated_backward)
    errors = tuple(
        (value.float() - wanted.float()).abs().max().item()
        for value, wanted in zip(actual, expected)
    )
    if any(error > 6.5e-2 for error in errors):
        raise RuntimeError(
            f"{target_name} dropout residual RMSNorm comparison failed: {errors}"
        )

    forward_call = prepare_kernel_call(
        forward, forward_arguments, generated_forward
    )
    backward_call = prepare_kernel_call(
        backward, backward_arguments, generated_backward
    )

    def generated_pipeline():
        forward_call()
        backward_call()

    generated_p50, generated_p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    print_artifact(forward, target_name)
    print_artifact(backward, target_name)
    print(
        f"{target_name} dropout residual RMSNorm forward/backward-data "
        f"numerical comparison: PASS (errors={errors})"
    )
    print(
        f"{target_name} dropout residual RMSNorm end-to-end performance "
        f"(CUDA Graph): p50={generated_p50:.4f} ms, "
        f"p95={generated_p95:.4f} ms"
    )
    print(f"{target_name} dropout residual RMSNorm upstream baseline: unavailable")


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
        cuda_graph=False,
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
        offset_values: list[int] | None = None,
    ) -> None:
        x = torch.randn((rows, k), device="cuda", dtype=torch.float16)
        x /= math.sqrt(k)
        weight = torch.randn(
            (GROUPS, k, n), device="cuda", dtype=torch.float16
        )
        offsets = torch.tensor(
            offset_values
            if offset_values is not None
            else [
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
    run_case(
        257,
        80,
        96,
        kernel_name="ragged grouped GEMM with empty groups",
        case_upstream=None,
        offset_values=[0, 0, 1, 17, 65, 129, 193, 257, 257],
    )
    empty_x = torch.empty((0, 80), device="cuda", dtype=torch.float16)
    empty_offsets = torch.zeros((GROUPS + 1,), device="cuda", dtype=torch.int32)
    empty_weight = torch.empty(
        (GROUPS, 80, 96), device="cuda", dtype=torch.float16
    )
    try:
        artifact.run(empty_x, empty_offsets, empty_weight)
    except NotImplementedError as error:
        if "zero-extent external views" not in str(error):
            raise
    else:
        raise RuntimeError(f"{target_name} accepted an all-empty ragged launch")
    print(f"{target_name} all-empty ragged capability check: PASS")


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
            measurement_scope="runtime-metadata",
            cuda_graph=False,
        )

    tail_dimension = 80
    tail_scale = 1.0 / math.sqrt(tail_dimension)
    tail_lengths = torch.tensor(
        [0, 1, 79, 131], device="cuda", dtype=torch.int32
    )
    tail_offsets = torch.zeros(
        (tail_lengths.numel() + 1,), device="cuda", dtype=torch.int32
    )
    tail_offsets[1:] = tail_lengths.cumsum(0)
    tail_tokens = int(tail_offsets[-1].item())
    tail_shape = (tail_tokens, tail_dimension)
    tail_q = torch.randn(tail_shape, device="cuda", dtype=torch.float16) * 0.5
    tail_k = torch.randn(tail_shape, device="cuda", dtype=torch.float16) * 0.5
    tail_v = torch.randn(tail_shape, device="cuda", dtype=torch.float16) * 0.5
    generated = artifact.run(
        tail_q,
        tail_k,
        tail_v,
        tail_lengths,
        tail_offsets,
        tail_scale,
    )
    expected = torch.empty_like(tail_v)
    for begin, end in zip(tail_offsets[:-1], tail_offsets[1:]):
        start = int(begin.item())
        stop = int(end.item())
        if start == stop:
            continue
        expected[start:stop] = F.scaled_dot_product_attention(
            tail_q[start:stop][None, None],
            tail_k[start:stop][None, None],
            tail_v[start:stop][None, None],
            is_causal=True,
            scale=tail_scale,
        )[0, 0]
    torch.cuda.synchronize()
    if generated.shape != expected.shape or generated.dtype != expected.dtype:
        raise RuntimeError(
            f"{target_name} packed varlen attention D=80 returned "
            f"shape={tuple(generated.shape)}, dtype={generated.dtype}; expected "
            f"shape={tuple(expected.shape)}, dtype={expected.dtype}"
        )
    error = (generated - expected).abs().max().item()
    if not torch.isfinite(generated).all().item() or error > 2.0e-2:
        raise RuntimeError(
            f"{target_name} packed varlen attention D=80 failed: {error}"
        )
    print(
        f"{target_name} packed varlen attention D=80 with an empty sequence: "
        f"PASS (error={error})"
    )


def _run_varlen_gqa_prefill(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    lengths = torch.tensor(
        [4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200],
        device="cuda",
        dtype=torch.int32,
    )
    if (
        lengths.numel() != VARLEN_BATCH
        or lengths.sum().item() != VARLEN_GQA_TOTAL_TOKENS
    ):
        raise RuntimeError("varlen GQA prefill shape constants do not match its input")
    cu_seqlens = torch.zeros(
        (VARLEN_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    cu_seqlens[1:] = lengths.cumsum(0)
    q = torch.randn(
        (VARLEN_GQA_TOTAL_TOKENS, VARLEN_GQA_QUERY_HEADS, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k = torch.randn(
        (VARLEN_GQA_TOTAL_TOKENS, VARLEN_GQA_KV_HEADS, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    artifact = intent.compile(
        flash_varlen_gqa_prefill,
        constexprs={"HEAD_GROUP": VARLEN_GQA_HEAD_GROUP},
        target=target,
        compiler=compiler,
    )
    arguments = (q, k, v, lengths, cu_seqlens, SCALE)

    def reference() -> torch.Tensor:
        result = torch.empty_like(q)
        key_heads = torch.arange(
            VARLEN_GQA_QUERY_HEADS, device="cuda"
        ) // VARLEN_GQA_HEAD_GROUP
        for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            start = int(begin.item())
            stop = int(end.item())
            query = q[start:stop].permute(1, 0, 2)[None]
            key = k[start:stop, key_heads, :].permute(1, 0, 2)[None]
            value = v[start:stop, key_heads, :].permute(1, 0, 2)[None]
            result[start:stop] = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=SCALE,
            )[0].permute(1, 0, 2)
        return result

    _compare(
        artifact=artifact,
        arguments=arguments,
        reference=reference,
        target_name=target_name,
        kernel_name="packed varlen GQA causal prefill",
        tolerance=3.0e-2,
        upstream=upstream,
        measurement_scope="runtime-metadata",
        cuda_graph=False,
        expected_dtype=torch.float16,
    )


def _run_varlen_gqa_rope_prefill(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("varlen GQA RoPE prefill has no algorithm-matched baseline")
    lengths = torch.tensor(
        [4096, 3968, 3840, 3712, 3584, 3456, 3328, 3200],
        device="cuda",
        dtype=torch.int32,
    )
    if (
        lengths.numel() != VARLEN_BATCH
        or lengths.sum().item() != VARLEN_GQA_TOTAL_TOKENS
    ):
        raise RuntimeError("varlen GQA RoPE prefill constants do not match its input")
    cu_seqlens = torch.zeros(
        (VARLEN_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    cu_seqlens[1:] = lengths.cumsum(0)
    q = torch.randn(
        (VARLEN_GQA_TOTAL_TOKENS, VARLEN_GQA_QUERY_HEADS, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k = torch.randn(
        (VARLEN_GQA_TOTAL_TOKENS, VARLEN_GQA_KV_HEADS, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    positions = torch.cat(
        [
            torch.arange(int(length.item()), device="cuda", dtype=torch.float32)
            for length in lengths
        ]
    )
    frequency = torch.arange(
        ROPE_HALF_DIMENSION, device="cuda", dtype=torch.float32
    )
    inverse_frequency = 1.0 / (10000.0 ** (frequency / ROPE_HALF_DIMENSION))
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = torch.cos(angles).to(torch.float16)
    sine = torch.sin(angles).to(torch.float16)
    q_rope = intent.compile(
        rotary_embedding_flat,
        constexprs={"HEADS": VARLEN_GQA_QUERY_HEADS},
        target=target,
        compiler=compiler,
    )
    k_rope = intent.compile(
        rotary_embedding_flat,
        constexprs={"HEADS": VARLEN_GQA_KV_HEADS},
        target=target,
        compiler=compiler,
    )
    attention = intent.compile(
        flash_varlen_gqa_prefill,
        constexprs={"HEAD_GROUP": VARLEN_GQA_HEAD_GROUP},
        target=target,
        compiler=compiler,
    )

    def rotate(values: torch.Tensor) -> torch.Tensor:
        paired = torch.cat(
            (
                -values[..., ROPE_HALF_DIMENSION:],
                values[..., :ROPE_HALF_DIMENSION],
            ),
            dim=-1,
        )
        cosine_full = torch.cat((cosine, cosine), dim=-1)
        sine_full = torch.cat((sine, sine), dim=-1)
        return values * cosine_full[:, None, :] + paired * sine_full[:, None, :]

    expected_q = rotate(q)
    expected_k = rotate(k)

    def attention_reference() -> torch.Tensor:
        result = torch.empty_like(q)
        key_heads = torch.arange(
            VARLEN_GQA_QUERY_HEADS, device="cuda"
        ) // VARLEN_GQA_HEAD_GROUP
        for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            start = int(begin.item())
            stop = int(end.item())
            query = expected_q[start:stop].permute(1, 0, 2)[None]
            key = expected_k[start:stop, key_heads, :].permute(1, 0, 2)[None]
            value = v[start:stop, key_heads, :].permute(1, 0, 2)[None]
            result[start:stop] = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=SCALE,
            )[0].permute(1, 0, 2)
        return result

    q_flat = q.reshape(-1, HEAD_DIMENSION)
    k_flat = k.reshape(-1, HEAD_DIMENSION)
    generated_q_flat = q_rope.run(q_flat, cosine, sine)
    generated_k_flat = k_rope.run(k_flat, cosine, sine)
    generated_q = generated_q_flat.reshape_as(q)
    generated_k = generated_k_flat.reshape_as(k)
    attention_arguments = (
        generated_q,
        generated_k,
        v,
        lengths,
        cu_seqlens,
        SCALE,
    )
    generated_output = attention.run(*attention_arguments)
    expected_output = attention_reference()
    errors = (
        (generated_q - expected_q).abs().max().item(),
        (generated_k - expected_k).abs().max().item(),
        (generated_output - expected_output).abs().max().item(),
    )
    if errors[0] > 2.0e-3 or errors[1] > 2.0e-3 or errors[2] > 4.0e-2:
        raise RuntimeError(
            f"{target_name} varlen GQA RoPE prefill comparison failed: {errors}"
        )

    q_call = prepare_kernel_call(
        q_rope, (q_flat, cosine, sine), generated_q_flat
    )
    k_call = prepare_kernel_call(
        k_rope, (k_flat, cosine, sine), generated_k_flat
    )
    attention_call = prepare_kernel_call(
        attention, attention_arguments, generated_output
    )

    def generated_pipeline():
        q_call()
        k_call()
        attention_call()

    generated_p50, generated_p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    print_artifact(q_rope, target_name)
    print_artifact(k_rope, target_name)
    print_artifact(attention, target_name)
    print(
        f"{target_name} packed varlen GQA causal prefill with RoPE "
        f"numerical comparison: PASS (q/k/output errors={errors})"
    )
    print(
        f"{target_name} packed varlen GQA causal prefill with RoPE "
        f"end-to-end performance (CUDA Event): p50={generated_p50:.4f} ms, "
        f"p95={generated_p95:.4f} ms"
    )
    print(f"{target_name} packed varlen GQA RoPE upstream baseline: unavailable")


def _run_paged_attention(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    sequence_lengths = torch.tensor(
        PAGED_SEQUENCE_LENGTHS,
        device="cuda",
        dtype=torch.int32,
    )
    if sequence_lengths.numel() != PAGED_BATCH:
        raise RuntimeError("paged attention batch constant does not match lengths")
    page_counts = torch.div(
        sequence_lengths + PAGED_PAGE_SIZE - 1,
        PAGED_PAGE_SIZE,
        rounding_mode="floor",
    )
    page_offsets = torch.zeros(
        (PAGED_BATCH + 1,),
        device="cuda",
        dtype=torch.int32,
    )
    page_offsets[1:] = page_counts.cumsum(0)
    total_pages = int(page_offsets[-1].item())
    page_indices = torch.randperm(total_pages, device="cuda", dtype=torch.int64).to(
        torch.int32
    )
    q = torch.randn(
        (PAGED_BATCH, PAGED_QUERY_HEADS, PAGED_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    key_cache = torch.randn(
        (
            total_pages,
            PAGED_PAGE_SIZE,
            PAGED_KV_HEADS,
            PAGED_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    value_cache = torch.randn_like(key_cache) * 0.5
    artifact = intent.compile(
        paged_gqa_decode_attention,
        constexprs={
            "PAGE_SIZE": PAGED_PAGE_SIZE,
            "HEAD_GROUP": PAGED_HEAD_GROUP,
        },
        target=target,
        compiler=compiler,
    )
    arguments = (
        q,
        key_cache,
        value_cache,
        page_offsets,
        page_indices,
        sequence_lengths,
        PAGED_SCALE,
    )

    def reference() -> torch.Tensor:
        result = torch.empty_like(q)
        key_heads = torch.arange(PAGED_QUERY_HEADS, device="cuda") // PAGED_HEAD_GROUP
        for batch in range(PAGED_BATCH):
            begin = int(page_offsets[batch].item())
            end = int(page_offsets[batch + 1].item())
            length = int(sequence_lengths[batch].item())
            physical_pages = page_indices[begin:end].long()
            key = key_cache[physical_pages].reshape(
                -1, PAGED_KV_HEADS, PAGED_HEAD_DIMENSION
            )[:length]
            value = value_cache[physical_pages].reshape(
                -1, PAGED_KV_HEADS, PAGED_HEAD_DIMENSION
            )[:length]
            expanded_key = key[:, key_heads, :].permute(1, 0, 2)[None]
            expanded_value = value[:, key_heads, :].permute(1, 0, 2)[None]
            result[batch] = F.scaled_dot_product_attention(
                q[batch][None, :, None, :],
                expanded_key,
                expanded_value,
                is_causal=False,
                scale=PAGED_SCALE,
            )[0, :, 0, :]
        return result

    _compare(
        artifact=artifact,
        arguments=arguments,
        reference=reference,
        target_name=target_name,
        kernel_name="paged GQA causal decode attention",
        tolerance=3.0e-2,
        upstream=upstream,
        expected_dtype=torch.float16,
    )
    masked_dimension = 80
    masked_query_heads = PAGED_HEAD_GROUP
    masked_query = torch.randn(
        (1, masked_query_heads, masked_dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    masked_key = torch.randn(
        (1, PAGED_PAGE_SIZE, 1, masked_dimension),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    masked_value = torch.randn_like(masked_key) * 0.5
    masked_page_offsets = torch.tensor(
        [0, 1], device="cuda", dtype=torch.int32
    )
    masked_page_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    masked_sequence_lengths = torch.tensor(
        [0], device="cuda", dtype=torch.int32
    )
    masked_output = artifact.run(
        masked_query,
        masked_key,
        masked_value,
        masked_page_offsets,
        masked_page_indices,
        masked_sequence_lengths,
        1.0 / math.sqrt(masked_dimension),
    )
    torch.cuda.synchronize()
    if not torch.isfinite(masked_output).all().item() or torch.count_nonzero(
        masked_output
    ).item():
        raise RuntimeError(
            f"{target_name} paged attention produced a nonzero or non-finite "
            "fully masked row"
        )
    print(
        f"{target_name} paged attention D=80 fully masked row: PASS "
        "(defined output=0)"
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
    masked_dimension = 80
    masked_q = torch.randn(
        (1, 1, 3, masked_dimension), device="cuda", dtype=torch.float16
    ) * 0.5
    masked_k = torch.randn(
        (1, 1, 5, masked_dimension), device="cuda", dtype=torch.float16
    ) * 0.5
    masked_v = torch.randn_like(masked_k) * 0.5
    masked_bias = torch.full(
        (1, 1, 5), -math.inf, device="cuda", dtype=torch.float32
    )
    masked_output = artifact.run(
        masked_q,
        masked_k,
        masked_v,
        masked_bias,
        1.0 / math.sqrt(masked_dimension),
    )
    torch.cuda.synchronize()
    if not torch.isfinite(masked_output).all().item() or torch.count_nonzero(
        masked_output
    ).item():
        raise RuntimeError(
            f"{target_name} vector-bias attention produced a nonzero or "
            "non-finite fully masked row"
        )
    print(
        f"{target_name} vector-bias attention D=80 fully masked row: PASS "
        "(defined output=0)"
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


def _run_sorted_nucleus_cutoff(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("sorted nucleus cutoff has no algorithm-matched baseline")
    ranks = torch.arange(
        NUCLEUS_CANDIDATES, device="cuda", dtype=torch.float32
    )
    probabilities = torch.softmax(-ranks / 256.0, dim=0)[None, :].expand(
        NUCLEUS_ROWS, -1
    ).contiguous()
    artifact = intent.compile(sorted_nucleus_cutoff, target=target, compiler=compiler)
    generated_cumulative, generated_cutoff = artifact.run(
        probabilities, NUCLEUS_THRESHOLD
    )
    expected_cumulative = probabilities.cumsum(dim=1)
    expected_cutoff = (
        (expected_cumulative < NUCLEUS_THRESHOLD).sum(dim=1, dtype=torch.int32)
        + 1
    )
    cumulative_error = (
        generated_cumulative - expected_cumulative
    ).abs().max().item()
    cutoff_error = (generated_cutoff - expected_cutoff).abs().max().item()
    if cumulative_error > 2.0e-4 or cutoff_error != 0:
        raise RuntimeError(
            f"{target_name} sorted nucleus cutoff comparison failed: "
            f"cumulative={cumulative_error}, cutoff={cutoff_error}"
        )
    call = prepare_kernel_call(
        artifact,
        (probabilities, NUCLEUS_THRESHOLD),
        (generated_cumulative, generated_cutoff),
    )
    p50, p95 = benchmark(call, warmup=25, repetitions=100, cuda_graph=True)
    print_artifact(artifact, target_name)
    print(
        f"{target_name} sorted nucleus cutoff numerical comparison: PASS "
        f"(cumulative error={cumulative_error}, cutoff error={cutoff_error})"
    )
    print(
        f"{target_name} sorted nucleus cutoff performance (CUDA Graph): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} sorted nucleus cutoff upstream baseline: unavailable")


def _run_insertion_top_k(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("insertion top-k has no algorithm-matched baseline")
    logits = torch.randn(
        (TOP_K_ROWS, TOP_K_VOCABULARY), device="cuda", dtype=torch.float32
    )
    artifact = intent.compile(insertion_top_k, target=target, compiler=compiler)
    generated_values, generated_indices = artifact.run(logits)
    expected_values, expected_indices = torch.topk(
        logits, TOP_K, dim=1, largest=True, sorted=True
    )
    value_error = (generated_values - expected_values).abs().max().item()
    index_matches = torch.equal(generated_indices.long(), expected_indices.long())
    if value_error != 0.0 or not index_matches:
        raise RuntimeError(
            f"{target_name} insertion top-k comparison failed: "
            f"values={value_error}, indices={index_matches}"
        )
    call = prepare_kernel_call(
        artifact, (logits,), (generated_values, generated_indices)
    )
    p50, p95 = benchmark(call, warmup=3, repetitions=100, cuda_graph=True)
    print_artifact(artifact, target_name)
    print(
        f"{target_name} insertion top-k numerical comparison: PASS "
        f"(value error={value_error}, indices={index_matches})"
    )
    print(
        f"{target_name} insertion top-k kernel-only performance (CUDA Graph): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} insertion top-k upstream baseline: unavailable")


def _run_embedding_backward_atomic(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("embedding backward atomic has no algorithm-matched baseline")
    token_ids = torch.arange(
        EMBEDDING_TOKENS, device="cuda", dtype=torch.int32
    )
    indices = (token_ids * 17) % EMBEDDING_VOCABULARY
    grad_output = torch.randn(
        (EMBEDDING_TOKENS, EMBEDDING_FEATURES),
        device="cuda",
        dtype=torch.float32,
    ) * 1.0e-3
    grad_weight = torch.zeros(
        (EMBEDDING_VOCABULARY, EMBEDDING_FEATURES),
        device="cuda",
        dtype=torch.float32,
    )
    expected = torch.zeros_like(grad_weight)
    expected.index_add_(0, indices.long(), grad_output)

    artifact = intent.compile(
        embedding_backward_atomic, target=target, compiler=compiler
    )
    artifact.run(indices, grad_output, grad_weight)
    torch.cuda.synchronize()
    error = (grad_weight - expected).abs().max().item()
    if error > 2.0e-6:
        raise RuntimeError(
            f"{target_name} embedding backward atomic comparison failed: {error}"
        )
    call = prepare_kernel_call(
        artifact,
        (indices, grad_output, grad_weight),
        (),
    )
    p50, p95 = benchmark(call, warmup=3, repetitions=100, cuda_graph=True)
    print_artifact(artifact, target_name)
    print(
        f"{target_name} embedding backward atomic numerical comparison: PASS "
        f"(max error={error})"
    )
    print(
        f"{target_name} embedding backward atomic kernel-only performance "
        f"(CUDA Graph): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} embedding backward atomic upstream baseline: unavailable")


EXTENDED_RUNNERS: dict[str, Runner] = {
    "attention_bias": _run_attention_bias,
    "batched_gemm": _run_batched_gemm,
    "bf16_gemm": _run_bf16_gemm,
    "conv1d": _run_conv1d,
    "conv2d": _run_conv2d,
    "cross_entropy": _run_cross_entropy,
    "dropout_residual_rms_norm": _run_dropout_residual_rms_norm,
    "dual_gemm": _run_dual_gemm,
    "embedding_backward_atomic": _run_embedding_backward_atomic,
    "fused_add_rms_norm": _run_fused_add_rms_norm,
    "grouped_gemm": _run_grouped_gemm,
    "grouped_query_head_add": _run_grouped_query_head_add,
    "insertion_top_k": _run_insertion_top_k,
    "layer_norm": _run_layer_norm,
    "layer_norm_backward": _run_layer_norm_backward,
    "logsumexp": _run_logsumexp,
    "online_softmax": _run_online_softmax,
    "paged_attention": _run_paged_attention,
    "quantized_gemm": _run_quantized_gemm,
    "rms_norm": _run_rms_norm,
    "scalar_table_lookup": _run_scalar_table_lookup,
    "selective_scan": _run_selective_scan,
    "shifted_row_copy": _run_shifted_row_copy,
    "sorted_nucleus_cutoff": _run_sorted_nucleus_cutoff,
    "swiglu_backward": _run_swiglu_backward,
    "swiglu_forward": _run_swiglu_forward,
    "varlen_attention": _run_varlen_attention,
    "varlen_gqa_prefill": _run_varlen_gqa_prefill,
    "varlen_gqa_rope_prefill": _run_varlen_gqa_rope_prefill,
    "weight_only_int4": _run_weight_only_int4,
}


def run_extended(
    kernel_name: str,
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    EXTENDED_RUNNERS[kernel_name](compiler, target, target_name, upstream)
