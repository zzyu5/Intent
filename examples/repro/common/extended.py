from __future__ import annotations

import math
from collections.abc import Callable
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.backward.attention import BATCH as BWD_ATTENTION_BATCH
from kernels.backward.attention import HEAD_DIMENSION as BWD_ATTENTION_DIMENSION
from kernels.backward.attention import HEAD_GROUP as BWD_ATTENTION_HEAD_GROUP
from kernels.backward.attention import KV_HEADS as BWD_ATTENTION_KV_HEADS
from kernels.backward.attention import QUERY_HEADS as BWD_ATTENTION_QUERY_HEADS
from kernels.backward.attention import SCALE as BWD_ATTENTION_SCALE
from kernels.backward.attention import SEQUENCE as BWD_ATTENTION_SEQUENCE
from kernels.backward.attention import attention_backward_delta
from kernels.backward.attention import attention_backward_dkdv
from kernels.backward.attention import attention_backward_dq
from kernels.backward.causal_conv import BATCH as BWD_CAUSAL_CONV_BATCH
from kernels.backward.causal_conv import CHANNELS as BWD_CAUSAL_CONV_CHANNELS
from kernels.backward.causal_conv import LENGTH as BWD_CAUSAL_CONV_LENGTH
from kernels.backward.causal_conv import WIDTH as BWD_CAUSAL_CONV_WIDTH
from kernels.backward.causal_conv import causal_conv1d_backward_partials
from kernels.backward.causal_conv import causal_conv1d_backward_reduce
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
from kernels.contraction.block_scaled import block_scaled_matmul
from kernels.backward.embedding import FEATURES as EMBEDDING_FEATURES
from kernels.backward.embedding import TOKENS as EMBEDDING_TOKENS
from kernels.backward.embedding import VOCABULARY as EMBEDDING_VOCABULARY
from kernels.backward.embedding import embedding_backward_atomic
from kernels.backward.embedding import embedding_forward_lookup
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
from kernels.contraction.sparse_2to4 import K as SPARSE_2TO4_K
from kernels.contraction.sparse_2to4 import M as SPARSE_2TO4_M
from kernels.contraction.sparse_2to4 import N as SPARSE_2TO4_N
from kernels.contraction.sparse_2to4 import sparse_2to4_gemm
from kernels.contraction.mla import mla_head_projection
from kernels.convolution.direct import CONV1D_BATCH
from kernels.convolution.direct import CONV1D_FILTER
from kernels.convolution.direct import CONV1D_LENGTH
from kernels.convolution.direct import CONV2D_BATCH
from kernels.convolution.direct import CONV2D_FILTER_HEIGHT
from kernels.convolution.direct import CONV2D_FILTER_WIDTH
from kernels.convolution.direct import CONV2D_HEIGHT
from kernels.convolution.direct import CONV2D_WIDTH
from kernels.convolution.direct import CAUSAL_CONV_BATCH
from kernels.convolution.direct import CAUSAL_CONV_CHANNELS
from kernels.convolution.direct import CAUSAL_CONV_LENGTH
from kernels.convolution.direct import CAUSAL_CONV_WIDTH
from kernels.convolution.direct import causal_depthwise_conv1d
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
from kernels.indexing.relations import INDEX_SELECT_FEATURES
from kernels.indexing.relations import INDEX_SELECT_ROWS
from kernels.indexing.relations import INDEX_SELECT_SOURCE_ROWS
from kernels.indexing.relations import SCALED_ADD_DESTINATION_ROWS
from kernels.indexing.relations import SCALED_ADD_FEATURES
from kernels.indexing.relations import SCALED_ADD_INNER_ROWS
from kernels.indexing.relations import SCALED_ADD_SOURCE_ROWS
from kernels.indexing.relations import index_select_rows
from kernels.indexing.relations import scaled_index_add_unique
from kernels.layout.transpose import COLUMNS as TRANSPOSE_COLUMNS
from kernels.layout.transpose import ROWS as TRANSPOSE_ROWS
from kernels.layout.transpose import matrix_transpose
from kernels.loss.cross_entropy import IGNORE_INDEX as CROSS_ENTROPY_IGNORE_INDEX
from kernels.loss.cross_entropy import TOKENS as CROSS_ENTROPY_TOKENS
from kernels.loss.cross_entropy import VOCABULARY as CROSS_ENTROPY_VOCABULARY
from kernels.loss.cross_entropy import fused_cross_entropy
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
from kernels.routing.mqa_logits import MQA_HEAD_DIMENSION
from kernels.routing.mqa_logits import MQA_HEADS
from kernels.routing.mqa_logits import MQA_KEYS
from kernels.routing.mqa_logits import MQA_QUERIES
from kernels.routing.mqa_logits import fp8_mqa_logits
from kernels.reduction.boolean import COLUMNS as BOOLEAN_REDUCTION_COLUMNS
from kernels.reduction.boolean import ROWS as BOOLEAN_REDUCTION_ROWS
from kernels.reduction.boolean import row_boolean_reduction
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
from kernels.streaming.ordered_prefix import BATCH as ORDERED_PREFIX_BATCH
from kernels.streaming.ordered_prefix import COLUMNS as ORDERED_PREFIX_COLUMNS
from kernels.streaming.ordered_prefix import ROWS as ORDERED_PREFIX_ROWS
from kernels.streaming.ordered_prefix import ordered_product_prefix
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
from kernels.streaming.attention import VARLEN_GQA_DECODE_BATCH
from kernels.streaming.attention import VARLEN_GQA_DECODE_BLOCK_SIZE
from kernels.streaming.attention import VARLEN_GQA_DECODE_HEAD_DIMENSION
from kernels.streaming.attention import VARLEN_GQA_DECODE_HEAD_GROUP
from kernels.streaming.attention import VARLEN_GQA_DECODE_KV_HEADS
from kernels.streaming.attention import VARLEN_GQA_DECODE_MAX_BLOCKS
from kernels.streaming.attention import VARLEN_GQA_DECODE_QUERY_HEADS
from kernels.streaming.attention import VARLEN_GQA_DECODE_SCALE
from kernels.streaming.attention import VARLEN_TOTAL_TOKENS
from kernels.streaming.attention import flash_attention_bias_fwd
from kernels.streaming.attention import flash_varlen_attention_fwd
from kernels.streaming.attention import flash_varlen_gqa_prefill
from kernels.streaming.attention import GQA_DECODE_BATCH
from kernels.streaming.attention import GQA_DECODE_HEAD_DIMENSION
from kernels.streaming.attention import GQA_DECODE_HEAD_GROUP
from kernels.streaming.attention import GQA_DECODE_KV_HEADS
from kernels.streaming.attention import GQA_DECODE_QUERY_HEADS
from kernels.streaming.attention import GQA_DECODE_SCALE
from kernels.streaming.attention import GQA_DECODE_SEQUENCE
from kernels.streaming.attention import continuous_gqa_decode
from kernels.streaming.attention import varlen_gqa_decode_with_sink_logits
from kernels.streaming.attention import MLA_PREFILL_BATCH
from kernels.streaming.attention import MLA_PREFILL_CONTENT_DIMENSION
from kernels.streaming.attention import MLA_PREFILL_HEAD_GROUP
from kernels.streaming.attention import MLA_PREFILL_KV_HEADS
from kernels.streaming.attention import MLA_PREFILL_POSITION_DIMENSION
from kernels.streaming.attention import MLA_PREFILL_QUERY_HEADS
from kernels.streaming.attention import MLA_PREFILL_SCALE
from kernels.streaming.attention import MLA_PREFILL_SEQUENCE
from kernels.streaming.attention import mla_prefill
from kernels.streaming.mla import ABSORBED_MLA_BATCH
from kernels.streaming.mla import ABSORBED_MLA_HEADS
from kernels.streaming.mla import ABSORBED_MLA_LATENT_DIMENSION
from kernels.streaming.mla import ABSORBED_MLA_NOPE_DIMENSION
from kernels.streaming.mla import ABSORBED_MLA_ROPE_DIMENSION
from kernels.streaming.mla import ABSORBED_MLA_SCALE
from kernels.streaming.mla import ABSORBED_MLA_SEQUENCE
from kernels.streaming.mla import ABSORBED_MLA_VALUE_DIMENSION
from kernels.streaming.mla import SPARSE_MLA_HEADS
from kernels.streaming.mla import SPARSE_MLA_KEYS
from kernels.streaming.mla import SPARSE_MLA_LATENT_DIMENSION
from kernels.streaming.mla import SPARSE_MLA_QUERIES
from kernels.streaming.mla import SPARSE_MLA_ROPE_DIMENSION
from kernels.streaming.mla import SPARSE_MLA_SCALE
from kernels.streaming.mla import PAGED_MLA_BATCH
from kernels.streaming.mla import PAGED_MLA_HEAD_GROUP
from kernels.streaming.mla import PAGED_MLA_KV_HEADS
from kernels.streaming.mla import PAGED_MLA_LATENT_DIMENSION
from kernels.streaming.mla import PAGED_MLA_PAGE_SIZE
from kernels.streaming.mla import PAGED_MLA_QUERY_HEADS
from kernels.streaming.mla import PAGED_MLA_ROPE_DIMENSION
from kernels.streaming.mla import PAGED_MLA_SCALE
from kernels.streaming.mla import PAGED_MLA_SEQUENCE_LENGTHS
from kernels.streaming.mla import paged_mla_decode
from kernels.streaming.mla import SPARSE_MLA_SELECTED_KEYS
from kernels.streaming.mla import absorbed_mla_prefill
from kernels.streaming.mla import token_sparse_mla_prefill
from kernels.position.rope import rotary_embedding_flat
from kernels.pointwise.batched_affine import BATCH as AFFINE_BATCH
from kernels.pointwise.batched_affine import COLUMNS as AFFINE_COLUMNS
from kernels.pointwise.batched_affine import ROWS as AFFINE_ROWS
from kernels.pointwise.batched_affine import batched_row_affine
from kernels.pointwise.record import ELEMENTS as RECORD_ELEMENTS
from kernels.pointwise.record import paired_sum_product
from kernels.pointwise.select import COLUMNS as SELECT_COLUMNS
from kernels.pointwise.select import ROWS as SELECT_ROWS
from kernels.pointwise.select import alternating_signed_indices
from kernels.pointwise.while_loop import ELEMENTS as WHILE_ELEMENTS
from kernels.pointwise.while_loop import integer_log2_floor
from kernels.streaming.paged_attention import BATCH as PAGED_BATCH
from kernels.streaming.paged_attention import HEAD_DIMENSION as PAGED_HEAD_DIMENSION
from kernels.streaming.paged_attention import HEAD_GROUP as PAGED_HEAD_GROUP
from kernels.streaming.paged_attention import KV_HEADS as PAGED_KV_HEADS
from kernels.streaming.paged_attention import PAGE_SIZE as PAGED_PAGE_SIZE
from kernels.streaming.paged_attention import QUERY_HEADS as PAGED_QUERY_HEADS
from kernels.streaming.paged_attention import SCALE as PAGED_SCALE
from kernels.streaming.paged_attention import SPLITS as PAGED_SPLITS
from kernels.streaming.paged_attention import paged_gqa_decode_partials
from kernels.streaming.paged_attention import SEQUENCE_LENGTHS as PAGED_SEQUENCE_LENGTHS
from kernels.streaming.paged_attention import paged_gqa_decode_attention
from kernels.streaming.block_sparse_attention import BATCH as BLOCK_SPARSE_BATCH
from kernels.streaming.block_sparse_attention import BLOCK_SIZE as BLOCK_SPARSE_BLOCK_SIZE
from kernels.streaming.block_sparse_attention import HEAD_DIMENSION as BLOCK_SPARSE_DIMENSION
from kernels.streaming.block_sparse_attention import HEAD_GROUP as BLOCK_SPARSE_HEAD_GROUP
from kernels.streaming.block_sparse_attention import KV_HEADS as BLOCK_SPARSE_KV_HEADS
from kernels.streaming.block_sparse_attention import QUERY_HEADS as BLOCK_SPARSE_QUERY_HEADS
from kernels.streaming.block_sparse_attention import SCALE as BLOCK_SPARSE_SCALE
from kernels.streaming.block_sparse_attention import SELECTED_BLOCKS as BLOCK_SPARSE_SELECTED_BLOCKS
from kernels.streaming.block_sparse_attention import SEQUENCE as BLOCK_SPARSE_SEQUENCE
from kernels.streaming.block_sparse_attention import SPLITS as BLOCK_SPARSE_SPLITS
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_combine
from kernels.streaming.block_sparse_attention import block_sparse_gqa_decode_partials
from kernels.streaming.selective_scan import BATCH as SELECTIVE_SCAN_BATCH
from kernels.streaming.selective_scan import LENGTH as SELECTIVE_SCAN_LENGTH
from kernels.streaming.selective_scan import selective_state_scan
from kernels.streaming.selective_scan import MAMBA_BATCH
from kernels.streaming.selective_scan import MAMBA_CHUNKS
from kernels.streaming.selective_scan import MAMBA_GROUPS
from kernels.streaming.selective_scan import MAMBA_CHUNK_SIZE
from kernels.streaming.selective_scan import MAMBA_HEADS
from kernels.streaming.selective_scan import MAMBA_HEAD_DIMENSION
from kernels.streaming.selective_scan import MAMBA_STATE_DIMENSION
from kernels.streaming.selective_scan import mamba_chunk_scan_fwd
from kernels.streaming.splitk_reduce import BATCH as SPLITK_REDUCE_BATCH
from kernels.streaming.splitk_reduce import HEAD_DIMENSION as SPLITK_REDUCE_DIMENSION
from kernels.streaming.splitk_reduce import HEADS as SPLITK_REDUCE_HEADS
from kernels.streaming.splitk_reduce import SPLITS as SPLITK_REDUCE_SPLITS
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from kernels.synchronization.compare_exchange import SLOTS as CAS_SLOTS
from kernels.synchronization.compare_exchange import claim_zero_slots
from kernels.contraction.weight_only_int4 import GROUP_SIZE as W4_GROUP_SIZE
from kernels.contraction.weight_only_int4 import K as W4_K
from kernels.contraction.weight_only_int4 import M as W4_M
from kernels.contraction.weight_only_int4 import N as W4_N
from kernels.contraction.weight_only_int4 import PACK_FACTOR as W4_PACK_FACTOR
from kernels.contraction.weight_only_int4 import weight_only_int4_matmul
from kernels.contraction.weight_only_int4 import fp8_e4m3_matmul
from kernels.contraction.weight_only_int4 import fp8_e5m2_matmul
from kernels.contraction.weight_only_int4 import W4A8_K
from kernels.contraction.weight_only_int4 import W4A8_M
from kernels.contraction.weight_only_int4 import W4A8_N
from kernels.contraction.weight_only_int4 import W4A8_PACK_FACTOR
from kernels.contraction.weight_only_int4 import w4a8_packed_matmul

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
    comparison_error: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
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
    error = (
        comparison_error(generated, expected)
        if comparison_error is not None
        else (generated - expected).abs().max().item()
    )
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
        upstream_error = (
            comparison_error(upstream_output, expected)
            if comparison_error is not None
            else (upstream_output - expected).abs().max().item()
        )
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


def _run_causal_conv1d(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError(
            "causal conv source checkout contains only the C++ dispatch wrapper"
        )
    x = torch.randn(
        (CAUSAL_CONV_BATCH, CAUSAL_CONV_CHANNELS, CAUSAL_CONV_LENGTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    weight = torch.randn(
        (CAUSAL_CONV_CHANNELS, CAUSAL_CONV_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    bias = torch.randn(
        (CAUSAL_CONV_CHANNELS,), device="cuda", dtype=torch.float16
    ) * 0.1
    artifact = intent.compile(
        causal_depthwise_conv1d,
        constexprs={"SILU": True},
        target=target,
        compiler=compiler,
    )
    _compare(
        artifact=artifact,
        arguments=(x, weight, bias),
        reference=lambda: F.silu(
            F.conv1d(
                F.pad(x, (CAUSAL_CONV_WIDTH - 1, 0)),
                weight[:, None, :],
                bias,
                groups=CAUSAL_CONV_CHANNELS,
            )
        ),
        target_name=target_name,
        kernel_name="causal depthwise conv1d",
        tolerance=3.0e-3,
        upstream=None,
        expected_dtype=torch.float16,
    )


def _run_continuous_gqa_decode(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    q = torch.randn(
        (
            GQA_DECODE_BATCH,
            GQA_DECODE_QUERY_HEADS,
            GQA_DECODE_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k = torch.randn(
        (
            GQA_DECODE_BATCH,
            GQA_DECODE_SEQUENCE,
            GQA_DECODE_KV_HEADS,
            GQA_DECODE_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    valid = torch.ones(
        (
            GQA_DECODE_BATCH,
            GQA_DECODE_SEQUENCE,
            GQA_DECODE_KV_HEADS,
        ),
        device="cuda",
        dtype=torch.uint8,
    )
    artifact = intent.compile(
        continuous_gqa_decode,
        constexprs={"HEAD_GROUP": GQA_DECODE_HEAD_GROUP},
        target=target,
        compiler=compiler,
    )

    def reference() -> torch.Tensor:
        expected = torch.empty_like(q)
        for key_head in range(GQA_DECODE_KV_HEADS):
            begin = key_head * GQA_DECODE_HEAD_GROUP
            end = begin + GQA_DECODE_HEAD_GROUP
            expected[:, begin:end] = F.scaled_dot_product_attention(
                q[:, begin:end, None, :],
                k[:, :, key_head, :][:, None, :, :],
                v[:, :, key_head, :][:, None, :, :],
                scale=GQA_DECODE_SCALE,
            )[:, :, 0, :]
        return expected

    _compare(
        artifact=artifact,
        arguments=(q, k, v, valid, GQA_DECODE_SCALE),
        reference=reference,
        target_name=target_name,
        kernel_name="continuous GQA decode",
        tolerance=3.0e-2,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_mla_prefill(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    q = torch.randn(
        (MLA_PREFILL_BATCH, MLA_PREFILL_QUERY_HEADS, MLA_PREFILL_SEQUENCE,
         MLA_PREFILL_CONTENT_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    qpe = torch.randn(
        (MLA_PREFILL_BATCH, MLA_PREFILL_QUERY_HEADS, MLA_PREFILL_SEQUENCE,
         MLA_PREFILL_POSITION_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    k = torch.randn(
        (MLA_PREFILL_BATCH, MLA_PREFILL_KV_HEADS, MLA_PREFILL_SEQUENCE,
         MLA_PREFILL_CONTENT_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    kpe = torch.randn(
        (MLA_PREFILL_BATCH, 1, MLA_PREFILL_SEQUENCE,
         MLA_PREFILL_POSITION_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    v = torch.randn_like(k)
    artifact = intent.compile(
        mla_prefill,
        target=target,
        compiler=compiler,
        constexprs={"HEAD_GROUP": MLA_PREFILL_HEAD_GROUP},
    )

    def reference() -> torch.Tensor:
        key = k.repeat_interleave(MLA_PREFILL_HEAD_GROUP, dim=1).float()
        value = v.repeat_interleave(MLA_PREFILL_HEAD_GROUP, dim=1).float()
        position_key = kpe.expand(
            MLA_PREFILL_BATCH,
            MLA_PREFILL_QUERY_HEADS,
            MLA_PREFILL_SEQUENCE,
            MLA_PREFILL_POSITION_DIMENSION,
        ).float()
        score = torch.matmul(q.float(), key.transpose(-1, -2))
        score += torch.matmul(qpe.float(), position_key.transpose(-1, -2))
        score *= MLA_PREFILL_SCALE
        causal = torch.triu(
            torch.ones(
                (MLA_PREFILL_SEQUENCE, MLA_PREFILL_SEQUENCE),
                device="cuda",
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        score.masked_fill_(causal, -torch.inf)
        probability = torch.softmax(score, dim=-1)
        return torch.matmul(probability, value).to(torch.float16)

    _compare(
        artifact=artifact,
        arguments=(q, qpe, k, kpe, v, MLA_PREFILL_SCALE),
        reference=reference,
        target_name=target_name,
        kernel_name="MLA causal prefill",
        tolerance=3e-2,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_absorbed_mla_prefill(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    q_latent = torch.randn(
        (
            ABSORBED_MLA_BATCH,
            ABSORBED_MLA_SEQUENCE,
            ABSORBED_MLA_HEADS,
            ABSORBED_MLA_LATENT_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    q_rope = torch.randn(
        (
            ABSORBED_MLA_BATCH,
            ABSORBED_MLA_SEQUENCE,
            ABSORBED_MLA_HEADS,
            ABSORBED_MLA_ROPE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    latent_cache = torch.randn(
        (
            ABSORBED_MLA_BATCH,
            ABSORBED_MLA_SEQUENCE,
            ABSORBED_MLA_LATENT_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    rope_cache = torch.randn(
        (
            ABSORBED_MLA_BATCH,
            ABSORBED_MLA_SEQUENCE,
            ABSORBED_MLA_ROPE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    artifact = intent.compile(absorbed_mla_prefill, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        scores = torch.einsum(
            "bqhc,bkc->bqhk",
            q_latent.float(),
            latent_cache.float(),
        )
        scores += torch.einsum(
            "bqhr,bkr->bqhk",
            q_rope.float(),
            rope_cache.float(),
        )
        scores *= ABSORBED_MLA_SCALE
        causal = torch.triu(
            torch.ones(
                (ABSORBED_MLA_SEQUENCE, ABSORBED_MLA_SEQUENCE),
                device="cuda",
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        scores.masked_fill_(causal[None, :, None, :], -torch.inf)
        probability = torch.softmax(scores, dim=-1)
        return torch.einsum(
            "bqhk,bkc->bqhc",
            probability,
            latent_cache.float(),
        ).half()

    _compare(
        artifact=artifact,
        arguments=(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            ABSORBED_MLA_SCALE,
        ),
        reference=reference,
        target_name=target_name,
        kernel_name="absorbed MLA causal prefill",
        tolerance=4.0e-2,
        upstream=None,
        expected_dtype=torch.float16,
    )


def _run_mla_head_projection(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    artifact = intent.compile(mla_head_projection, target=target, compiler=compiler)
    cases = (
        (
            "query absorb",
            ABSORBED_MLA_NOPE_DIMENSION,
            ABSORBED_MLA_LATENT_DIMENSION,
        ),
        (
            "value reconstruct",
            ABSORBED_MLA_LATENT_DIMENSION,
            ABSORBED_MLA_VALUE_DIMENSION,
        ),
    )
    for label, input_dimension, output_dimension in cases:
        source = torch.randn(
            (
                ABSORBED_MLA_BATCH,
                ABSORBED_MLA_SEQUENCE,
                ABSORBED_MLA_HEADS,
                input_dimension,
            ),
            device="cuda",
            dtype=torch.float16,
        ) * 0.1
        weight = torch.randn(
            (ABSORBED_MLA_HEADS, output_dimension, input_dimension),
            device="cuda",
            dtype=torch.float16,
        ) * 0.05
        _compare(
            artifact=artifact,
            arguments=(source, weight),
            reference=lambda source=source, weight=weight: torch.einsum(
                "bqhi,hoi->bqho",
                source.float(),
                weight.float(),
            ).half(),
            target_name=target_name,
            kernel_name=f"MLA head projection {label}",
            tolerance=3.0e-2,
            upstream=None,
            expected_dtype=torch.float16,
        )


def _run_token_sparse_mla_prefill(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    q_latent = torch.randn(
        (SPARSE_MLA_QUERIES, SPARSE_MLA_HEADS, SPARSE_MLA_LATENT_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    q_rope = torch.randn(
        (SPARSE_MLA_QUERIES, SPARSE_MLA_HEADS, SPARSE_MLA_ROPE_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    latent_cache = torch.randn(
        (SPARSE_MLA_KEYS, SPARSE_MLA_LATENT_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    rope_cache = torch.randn(
        (SPARSE_MLA_KEYS, SPARSE_MLA_ROPE_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    selected = torch.randint(
        0,
        SPARSE_MLA_KEYS,
        (SPARSE_MLA_QUERIES, SPARSE_MLA_SELECTED_KEYS),
        device="cuda",
        dtype=torch.int32,
    )
    selected[:, -2] = -1
    selected[:, -1] = SPARSE_MLA_KEYS
    artifact = intent.compile(token_sparse_mla_prefill, target=target, compiler=compiler)

    def reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = (selected >= 0) & (selected < SPARSE_MLA_KEYS)
        safe = selected.clamp(0, SPARSE_MLA_KEYS - 1).long()
        focused_latent = latent_cache[safe].float()
        focused_rope = rope_cache[safe].float()
        scores = torch.einsum("qhc,qtc->qht", q_latent.float(), focused_latent)
        scores += torch.einsum("qhr,qtr->qht", q_rope.float(), focused_rope)
        scores *= SPARSE_MLA_SCALE * math.log2(math.e)
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        maximum = scores.max(dim=2).values
        denominator = torch.exp2(scores - maximum[:, :, None]).sum(dim=2)
        lse = maximum + torch.log2(denominator)
        probability = torch.exp2(scores - lse[:, :, None])
        output = torch.einsum("qht,qtc->qhc", probability, focused_latent)
        return output.bfloat16(), maximum, lse

    arguments = (
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        selected,
        SPARSE_MLA_SCALE,
    )
    generated = artifact.run(*arguments)
    expected = reference()
    if not isinstance(generated, tuple) or len(generated) != 3:
        raise RuntimeError(f"{target_name} token-sparse MLA returned wrong ABI")
    errors = tuple(
        (actual.float() - wanted.float()).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if errors[0] > 5.0e-2 or errors[1] > 5.0e-3 or errors[2] > 5.0e-3:
        raise RuntimeError(
            f"{target_name} token-sparse MLA numerical comparison failed: {errors}"
        )
    generated_call = prepare_kernel_call(artifact, arguments, generated)
    generated_p50, generated_p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} token-sparse MLA prefill numerical comparison: PASS "
        f"(output/max/lse={errors})"
    )
    print(
        f"{target_name} token-sparse MLA prefill kernel-only performance "
        f"(CUDA Graph): p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    print(f"{target_name} token-sparse MLA prefill upstream baseline: unavailable")


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


def _run_mamba_chunk_scan(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    length = MAMBA_CHUNKS * MAMBA_CHUNK_SIZE
    heads_per_group = MAMBA_HEADS // MAMBA_GROUPS
    cb = torch.randn(
        (
            MAMBA_BATCH,
            MAMBA_CHUNKS,
            MAMBA_GROUPS,
            MAMBA_CHUNK_SIZE,
            MAMBA_CHUNK_SIZE,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    x = torch.randn(
        (MAMBA_BATCH, length, MAMBA_HEADS, MAMBA_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    dt = torch.rand(
        (MAMBA_BATCH, MAMBA_HEADS, MAMBA_CHUNKS, MAMBA_CHUNK_SIZE),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    dA = -torch.rand_like(dt) * 0.1
    state_matrix = torch.randn(
        (MAMBA_BATCH, length, MAMBA_GROUPS, MAMBA_STATE_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    previous = torch.randn(
        (
            MAMBA_BATCH,
            MAMBA_CHUNKS,
            MAMBA_HEADS,
            MAMBA_HEAD_DIMENSION,
            MAMBA_STATE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.05
    residual_scale = torch.randn(
        (MAMBA_HEADS,), device="cuda", dtype=torch.float16
    ) * 0.1
    artifact = intent.compile(
        mamba_chunk_scan_fwd,
        constexprs={"HEADS_PER_GROUP": heads_per_group},
        target=target,
        compiler=compiler,
    )

    def reference() -> torch.Tensor:
        group_index = torch.arange(MAMBA_HEADS, device="cuda") // heads_per_group
        chunk_x = x.reshape(
            MAMBA_BATCH,
            MAMBA_CHUNKS,
            MAMBA_CHUNK_SIZE,
            MAMBA_HEADS,
            MAMBA_HEAD_DIMENSION,
        )
        state_c = state_matrix.reshape(
            MAMBA_BATCH,
            MAMBA_CHUNKS,
            MAMBA_CHUNK_SIZE,
            MAMBA_GROUPS,
            MAMBA_STATE_DIMENSION,
        )[:, :, :, group_index, :]
        state = torch.einsum(
            "bcshn,bchpn->bcshp",
            state_c.float(),
            previous.float(),
        )
        state *= torch.exp(dA.float()).permute(0, 2, 3, 1)[..., None]
        decay = torch.exp(
            dA.float()[:, :, :, :, None] - dA.float()[:, :, :, None, :]
        ).permute(0, 2, 3, 4, 1)
        coefficients = cb[:, :, group_index].permute(0, 1, 3, 4, 2).float()
        coefficients *= decay
        coefficients *= dt.float().permute(0, 2, 3, 1)[:, :, None, :, :]
        causal = torch.tril(
            torch.ones(
                (MAMBA_CHUNK_SIZE, MAMBA_CHUNK_SIZE),
                device="cuda",
                dtype=torch.bool,
            )
        )
        coefficients.masked_fill_(~causal[None, None, :, :, None], 0.0)
        scan = torch.einsum(
            "bcskh,bckhp->bcshp",
            coefficients,
            chunk_x.float(),
        )
        result = state + scan + chunk_x.float() * residual_scale.float()[None, None, None, :, None]
        return result.reshape_as(x).half()

    _compare(
        artifact=artifact,
        arguments=(cb, x, dt, dA, state_matrix, previous, residual_scale),
        reference=reference,
        target_name=target_name,
        kernel_name="Mamba chunk scan forward",
        tolerance=5.0e-2,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_w4a8_packed(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    activation = torch.randint(
        -4, 5, (W4A8_M, W4A8_K), device="cuda", dtype=torch.int8
    )
    quantized = torch.randint(
        -8, 8, (W4A8_N, W4A8_K), device="cuda", dtype=torch.int32
    )
    packed = torch.zeros(
        (W4A8_N, W4A8_K // W4A8_PACK_FACTOR),
        device="cuda",
        dtype=torch.uint8,
    )
    unsigned = quantized & 15
    for lane in range(W4A8_PACK_FACTOR):
        packed |= (unsigned[:, lane::W4A8_PACK_FACTOR] << (4 * lane)).to(
            torch.uint8
        )
    artifact = intent.compile(
        w4a8_packed_matmul,
        target=target,
        compiler=compiler,
    )

    def reference() -> torch.Tensor:
        return torch.matmul(
            quantized.float(),
            activation.float().transpose(0, 1),
        ).to(torch.int32)

    _compare(
        artifact=artifact,
        arguments=(activation, packed),
        reference=reference,
        target_name=target_name,
        kernel_name="packed W4A8 int8-by-int4 matmul",
        tolerance=0.0,
        upstream=upstream,
        expected_dtype=torch.int32,
    )


def _run_embedding_forward_lookup(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    lookup_tokens = 8192
    embedding_table = torch.randn(
        (EMBEDDING_VOCABULARY, EMBEDDING_FEATURES),
        device="cuda",
        dtype=torch.float32,
    )
    indices = torch.randint(
        0,
        EMBEDDING_VOCABULARY,
        (lookup_tokens,),
        device="cuda",
        dtype=torch.int32,
    )
    artifact = intent.compile(
        embedding_forward_lookup,
        target=target,
        compiler=compiler,
    )
    _compare(
        artifact=artifact,
        arguments=(embedding_table, indices),
        reference=lambda: embedding_table[indices.long()],
        target_name=target_name,
        kernel_name="embedding forward lookup",
        tolerance=0.0,
        upstream=upstream,
        expected_dtype=torch.float32,
    )


def _run_block_scaled_matmul(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    lhs = torch.randn(
        (512, 24, 32), device="cuda", dtype=torch.float16
    ).to(torch.float8_e4m3fn)
    rhs = torch.randn(
        (24, 32, 512), device="cuda", dtype=torch.float16
    ).to(torch.float8_e4m3fn)
    lhs_scale = torch.ones(
        (512, 24), device="cuda", dtype=torch.float32
    ).to(torch.float8_e8m0fnu)
    rhs_scale = torch.ones(
        (24, 512), device="cuda", dtype=torch.float32
    ).to(torch.float8_e8m0fnu)
    if target_name == "Triton":
        lhs_scale = lhs_scale.view(torch.uint8)
        rhs_scale = rhs_scale.view(torch.uint8)
    artifact = intent.compile(
        block_scaled_matmul, target=target, compiler=compiler
    )

    def reference() -> torch.Tensor:
        decoded_lhs_scale = (
            torch.pow(2.0, lhs_scale.float() - 127.0)
            if lhs_scale.dtype == torch.uint8
            else lhs_scale.float()
        )
        decoded_rhs_scale = (
            torch.pow(2.0, rhs_scale.float() - 127.0)
            if rhs_scale.dtype == torch.uint8
            else rhs_scale.float()
        )
        scaled_lhs = lhs.float().reshape(512, 24 * 32) * decoded_lhs_scale.repeat_interleave(
            32, dim=1
        )
        scaled_rhs = rhs.float().reshape(24 * 32, 512) * decoded_rhs_scale.repeat_interleave(
            32, dim=0
        )
        return scaled_lhs @ scaled_rhs

    _compare(
        artifact=artifact,
        arguments=(lhs, lhs_scale, rhs, rhs_scale),
        reference=reference,
        target_name=target_name,
        kernel_name="block-scaled matmul",
        tolerance=4.0e-1,
        upstream=upstream,
        expected_dtype=torch.float32,
    )


def _run_fp8_mqa_logits(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    q = (
        torch.randn(
            (MQA_QUERIES, MQA_HEADS, MQA_HEAD_DIMENSION),
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn(
            (MQA_KEYS, MQA_HEAD_DIMENSION),
            device="cuda",
            dtype=torch.float16,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    kv_scale = 0.5 + torch.rand(
        (MQA_KEYS,),
        device="cuda",
        dtype=torch.float32,
    )
    head_weight = torch.randn(
        (MQA_QUERIES, MQA_HEADS),
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    key_start = torch.arange(MQA_QUERIES, device="cuda", dtype=torch.int32) % 17
    key_end = MQA_KEYS - (
        torch.arange(MQA_QUERIES, device="cuda", dtype=torch.int32) % 19
    )
    artifact = intent.compile(fp8_mqa_logits, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        per_head = torch.einsum("qhd,kd->qhk", q.float(), kv.float())
        result = (
            torch.relu(per_head) * head_weight[:, :, None]
        ).sum(dim=1) * kv_scale[None, :]
        key = torch.arange(MQA_KEYS, device="cuda", dtype=torch.int32)
        valid = (
            key[None, :] >= key_start[:, None]
        ) & (
            key[None, :] < key_end[:, None]
        )
        return result.masked_fill(~valid, -torch.inf)

    def finite_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        finite = torch.isfinite(expected)
        if not torch.equal(torch.isfinite(actual), finite):
            return float("inf")
        return (actual[finite] - expected[finite]).abs().max().item()

    _compare(
        artifact=artifact,
        arguments=(q, kv, kv_scale, head_weight, key_start, key_end),
        reference=reference,
        target_name=target_name,
        kernel_name="FP8 MQA weighted logits",
        tolerance=2.0e-2,
        upstream=None,
        expected_dtype=torch.float32,
        comparison_error=finite_error,
    )

def _run_splitk_attention_reduce(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    partial = torch.randn(
        (
            SPLITK_REDUCE_BATCH,
            SPLITK_REDUCE_HEADS,
            SPLITK_REDUCE_SPLITS,
            SPLITK_REDUCE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    partial_lse = torch.randn(
        (SPLITK_REDUCE_BATCH, SPLITK_REDUCE_HEADS, SPLITK_REDUCE_SPLITS),
        device="cuda",
        dtype=torch.float32,
    )
    artifact = intent.compile(
        splitk_attention_reduce,
        target=target,
        compiler=compiler,
    )

    def reference() -> torch.Tensor:
        maximum = partial_lse.max(dim=2, keepdim=True).values
        weights = torch.exp2(partial_lse - maximum)
        return (
            torch.sum(weights[..., None] * partial.float(), dim=2)
            / weights.sum(dim=2)[..., None]
        ).bfloat16()

    _compare(
        artifact=artifact,
        arguments=(partial, partial_lse),
        reference=reference,
        target_name=target_name,
        kernel_name="split-K attention reducer",
        tolerance=2.0e-2,
        upstream=upstream,
        expected_dtype=torch.bfloat16,
    )


def _run_fp8_gemm(
    compiler: str,
    target: Target,
    target_name: str,
    upstream: Upstream | None,
) -> None:
    for label, dtype, kernel in (
        ("e4m3", torch.float8_e4m3fn, fp8_e4m3_matmul),
        ("e5m2", torch.float8_e5m2, fp8_e5m2_matmul),
    ):
        lhs = torch.randn((1024, 1024), device="cuda", dtype=torch.float16).to(
            dtype
        )
        rhs = torch.randn((1024, 1024), device="cuda", dtype=torch.float16).to(
            dtype
        )
        artifact = intent.compile(kernel, target=target, compiler=compiler)

        def reference(lhs=lhs, rhs=rhs, dtype=dtype) -> torch.Tensor:
            return torch.matmul(lhs.float(), rhs.float().transpose(0, 1)).to(dtype)

        def similarity_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
            actual64 = actual.double()
            expected64 = expected.double()
            denominator = (actual64 * actual64 + expected64 * expected64).sum()
            return abs(1.0 - (2.0 * (actual64 * expected64).sum() / denominator).item())

        _compare(
            artifact=artifact,
            arguments=(lhs, rhs),
            reference=reference,
            target_name=target_name,
            kernel_name=f"FP8 GEMM {label}",
            tolerance=1.0e-3,
            upstream=upstream,
            expected_dtype=dtype,
            comparison_error=similarity_error,
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
        -8,
        8,
        (W4_K, W4_N),
        device="cuda",
        dtype=torch.int32,
    )
    packed = torch.zeros(
        (W4_K // W4_PACK_FACTOR, W4_N),
        device="cuda",
        dtype=torch.int32,
    )
    unsigned = quantized & 15
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
        kernel_name="signed W4A16 groupwise matmul",
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


def _run_sparse_2to4_gemm(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    artifact = intent.compile(sparse_2to4_gemm, target=target, compiler=compiler)
    dense = torch.randn(
        (SPARSE_2TO4_M, SPARSE_2TO4_K), device="cuda", dtype=torch.float16
    )
    dense = dense.view(SPARSE_2TO4_M, -1, 4)
    keep = dense.abs().topk(2, dim=-1).indices
    sparse_dense = torch.zeros_like(dense).scatter(-1, keep, dense.gather(-1, keep))
    sparse_dense = sparse_dense.view(SPARSE_2TO4_M, SPARSE_2TO4_K)
    source_path = (
        Path(__file__).parents[3]
        / "source/tilelang/tilelang/gemm/sparse_2to4/sparse_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "intent_sparse_2to4_utils", source_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compressed, metadata = module.torch_compress(
        sparse_dense, meta_dtype=torch.int16
    )
    rhs = torch.randn(
        (SPARSE_2TO4_K, SPARSE_2TO4_N), device="cuda", dtype=torch.float16
    )
    rhs /= math.sqrt(SPARSE_2TO4_K)
    _compare(
        artifact=artifact,
        arguments=(compressed, metadata, rhs),
        reference=lambda: sparse_dense.float() @ rhs.float(),
        target_name=target_name,
        kernel_name="2:4 structured sparse GEMM",
        tolerance=1.0e-1,
        upstream=upstream,
        expected_dtype=torch.float32,
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
    artifact = intent.compile(
        fused_cross_entropy,
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
    expected_gradient *= valid[:, None].float()

    generated_logits = logits.clone()
    generated_outputs = artifact.run(generated_logits, labels)
    if not isinstance(generated_outputs, tuple) or len(generated_outputs) != 2:
        raise RuntimeError(
            f"{target_name} fused cross entropy did not return two outputs"
        )
    generated_loss, generated_prediction = generated_outputs
    generated_gradient = generated_logits
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

    generated_call = prepare_kernel_call(
        artifact, (generated_logits, labels), generated_outputs
    )
    generated_p50, generated_p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=True,
    )
    upstream_output = upstream((logits, labels)) if upstream else None
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
            lambda: upstream((logits, labels)),
            warmup=3,
            repetitions=100,
            cuda_graph=True,
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} fused cross entropy numerical comparison: PASS "
        f"(loss={loss_error}, prediction={prediction_matches}, "
        f"gradient={gradient_error})"
    )
    print(
        f"{target_name} fused cross entropy kernel-only performance (CUDA Graph): "
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
    x_f32 = x.float()
    mean = x_f32.mean(dim=1)
    centered = x_f32 - mean[:, None]
    rstd = torch.rsqrt(centered.square().mean(dim=1) + epsilon)
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
        (x, dy, weight, mean, rstd, dw_partial, db_partial, inverse_features),
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

    dy_f32 = dy.float()
    weight_f32 = weight.float()
    normalized = centered * rstd[:, None]
    weighted_dy = weight_f32 * dy_f32
    expected = (
        (
            (
                weighted_dy
                - weighted_dy.mean(dim=1, keepdim=True)
                - normalized * (weighted_dy * normalized).mean(dim=1, keepdim=True)
            )
            * rstd[:, None]
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


def _run_attention_backward(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    shape_q = (
        BWD_ATTENTION_BATCH,
        BWD_ATTENTION_QUERY_HEADS,
        BWD_ATTENTION_SEQUENCE,
        BWD_ATTENTION_DIMENSION,
    )
    shape_kv = (
        BWD_ATTENTION_BATCH,
        BWD_ATTENTION_KV_HEADS,
        BWD_ATTENTION_SEQUENCE,
        BWD_ATTENTION_DIMENSION,
    )
    q = torch.randn(shape_q, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape_kv, device="cuda", dtype=torch.float16) * 0.5
    grad_output = torch.randn_like(q) * 0.05

    q_reference = q.float().detach().requires_grad_(True)
    k_reference = k.float().detach().requires_grad_(True)
    v_reference = v.float().detach().requires_grad_(True)
    expanded_k = k_reference.repeat_interleave(BWD_ATTENTION_HEAD_GROUP, dim=1)
    expanded_v = v_reference.repeat_interleave(BWD_ATTENTION_HEAD_GROUP, dim=1)
    scores = torch.matmul(q_reference, expanded_k.transpose(-2, -1))
    scores = scores * BWD_ATTENTION_SCALE
    causal = torch.ones(
        (BWD_ATTENTION_SEQUENCE, BWD_ATTENTION_SEQUENCE),
        device="cuda",
        dtype=torch.bool,
    ).tril()
    scores = scores.masked_fill(~causal, -float("inf"))
    probability = torch.softmax(scores, dim=-1)
    reference_output_f32 = torch.matmul(probability, expanded_v)
    reference_output_f32.backward(grad_output.float())
    expected = (
        q_reference.grad.to(torch.float16),
        k_reference.grad.to(torch.float16),
        v_reference.grad.to(torch.float16),
    )
    output = reference_output_f32.detach().to(torch.float16)
    lse = (
        torch.logsumexp(scores.detach(), dim=-1) * math.log2(math.e)
    ).to(torch.float32)

    delta_artifact = intent.compile(
        attention_backward_delta, target=target, compiler=compiler
    )
    constexprs = {"HEAD_GROUP": BWD_ATTENTION_HEAD_GROUP, "CAUSAL": True}
    dkdv_artifact = intent.compile(
        attention_backward_dkdv,
        target=target,
        compiler=compiler,
        constexprs=constexprs,
    )
    dq_artifact = intent.compile(
        attention_backward_dq,
        target=target,
        compiler=compiler,
        constexprs=constexprs,
    )
    delta = torch.empty(
        shape_q[:-1], device="cuda", dtype=torch.float32
    )
    grad_q = torch.empty_like(q)
    grad_k = torch.empty_like(k)
    grad_v = torch.empty_like(v)
    delta_call = prepare_kernel_call(
        delta_artifact, (output, grad_output), delta
    )
    common = (q, k, v, grad_output, lse, delta, BWD_ATTENTION_SCALE)
    dkdv_call = prepare_kernel_call(
        dkdv_artifact, common, (grad_k, grad_v)
    )
    dq_call = prepare_kernel_call(dq_artifact, common, grad_q)

    def generated_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta_call()
        dkdv_call()
        dq_call()
        return grad_q, grad_k, grad_v

    generated = generated_pipeline()
    torch.cuda.synchronize()
    errors = tuple(
        (actual - wanted).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if errors[0] > 1.25e-1 or errors[1] > 2.5e-1 or errors[2] > 1.25e-1:
        raise RuntimeError(
            f"{target_name} attention backward numerical comparison failed: {errors}"
        )
    p50, p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    for artifact in (delta_artifact, dkdv_artifact, dq_artifact):
        print_artifact(artifact, target_name)
    print(
        f"{target_name} attention backward numerical comparison: PASS "
        f"(dq/dk/dv errors={errors})"
    )
    print(
        f"{target_name} attention backward end-to-end GPU pipeline performance "
        f"(CUDA Event): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} attention backward upstream baseline: unavailable")


def _run_causal_conv1d_backward(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    shape = (
        BWD_CAUSAL_CONV_BATCH,
        BWD_CAUSAL_CONV_CHANNELS,
        BWD_CAUSAL_CONV_LENGTH,
    )
    x = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    weight = torch.randn(
        (BWD_CAUSAL_CONV_CHANNELS, BWD_CAUSAL_CONV_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    grad_output = torch.randn_like(x) * 0.05
    x_reference = x.float().detach().requires_grad_(True)
    weight_reference = weight.float().detach().requires_grad_(True)
    bias_reference = torch.zeros(
        (BWD_CAUSAL_CONV_CHANNELS,),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    expected_output = F.conv1d(
        x_reference,
        weight_reference[:, None, :],
        bias=bias_reference,
        padding=BWD_CAUSAL_CONV_WIDTH - 1,
        groups=BWD_CAUSAL_CONV_CHANNELS,
    )[..., :BWD_CAUSAL_CONV_LENGTH]
    expected_output.backward(grad_output.float())
    expected = (
        x_reference.grad.to(torch.float16),
        weight_reference.grad,
        bias_reference.grad,
    )

    partial_artifact = intent.compile(
        causal_conv1d_backward_partials, target=target, compiler=compiler
    )
    reduce_artifact = intent.compile(
        causal_conv1d_backward_reduce, target=target, compiler=compiler
    )
    grad_x = torch.empty_like(x)
    partial_weight = torch.empty(
        (
            BWD_CAUSAL_CONV_BATCH,
            BWD_CAUSAL_CONV_CHANNELS,
            BWD_CAUSAL_CONV_WIDTH,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    partial_bias = torch.empty(
        (BWD_CAUSAL_CONV_BATCH, BWD_CAUSAL_CONV_CHANNELS),
        device="cuda",
        dtype=torch.float32,
    )
    grad_weight = torch.empty_like(weight, dtype=torch.float32)
    grad_bias = torch.empty(
        (BWD_CAUSAL_CONV_CHANNELS,), device="cuda", dtype=torch.float32
    )
    partial_call = prepare_kernel_call(
        partial_artifact,
        (x, weight, grad_output),
        (grad_x, partial_weight, partial_bias),
    )
    reduce_call = prepare_kernel_call(
        reduce_artifact,
        (partial_weight, partial_bias),
        (grad_weight, grad_bias),
    )

    def generated_pipeline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        partial_call()
        reduce_call()
        return grad_x, grad_weight, grad_bias

    generated = generated_pipeline()
    torch.cuda.synchronize()
    errors = tuple(
        (actual - wanted).abs().max().item()
        for actual, wanted in zip(generated, expected)
    )
    if errors[0] > 2.5e-3 or errors[1] > 2.5e-1 or errors[2] > 2.5e-1:
        raise RuntimeError(
            f"{target_name} causal conv1d backward numerical comparison failed: {errors}"
        )
    p50, p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    for artifact in (partial_artifact, reduce_artifact):
        print_artifact(artifact, target_name)
    print(
        f"{target_name} causal conv1d backward numerical comparison: PASS "
        f"(dx/dw/db errors={errors})"
    )
    print(
        f"{target_name} causal conv1d backward end-to-end GPU pipeline performance "
        f"(CUDA Event): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} causal conv1d backward upstream baseline: unavailable")


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


def _run_varlen_gqa_decode_logits(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError(
            "varlen GQA decode logits has no target-matched generated comparison"
        )
    lengths = torch.tensor(
        [4096, 3904, 3584, 3328, 3008, 2752, 2432, 2176],
        device="cuda",
        dtype=torch.int32,
    )
    if lengths.numel() != VARLEN_GQA_DECODE_BATCH:
        raise RuntimeError("varlen GQA decode batch constant does not match its input")
    cu_seqlens = torch.zeros(
        (VARLEN_GQA_DECODE_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    cu_seqlens[1:] = lengths.cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())
    q = torch.randn(
        (
            VARLEN_GQA_DECODE_BATCH,
            VARLEN_GQA_DECODE_QUERY_HEADS,
            VARLEN_GQA_DECODE_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k = torch.randn(
        (
            total_tokens,
            VARLEN_GQA_DECODE_KV_HEADS,
            VARLEN_GQA_DECODE_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    sink = torch.rand(
        (VARLEN_GQA_DECODE_QUERY_HEADS,), device="cuda", dtype=torch.float32
    ) * 0.125
    artifact = intent.compile(
        varlen_gqa_decode_with_sink_logits,
        constexprs={
            "HEAD_GROUP": VARLEN_GQA_DECODE_HEAD_GROUP,
            "MAX_BLOCKS": VARLEN_GQA_DECODE_MAX_BLOCKS,
        },
        target=target,
        compiler=compiler,
    )
    output = torch.empty_like(q)
    block_logits = torch.empty(
        (
            VARLEN_GQA_DECODE_BATCH,
            VARLEN_GQA_DECODE_QUERY_HEADS,
            VARLEN_GQA_DECODE_MAX_BLOCKS,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    arguments = (
        q,
        k,
        v,
        cu_seqlens,
        sink,
        VARLEN_GQA_DECODE_SCALE,
    )
    generated_call = prepare_kernel_call(
        artifact, arguments, (output, block_logits)
    )

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        expected_output = torch.empty_like(q)
        expected_logits = torch.zeros_like(block_logits)
        for batch in range(VARLEN_GQA_DECODE_BATCH):
            begin = int(cu_seqlens[batch].item())
            end = int(cu_seqlens[batch + 1].item())
            for query_head in range(VARLEN_GQA_DECODE_QUERY_HEADS):
                key_head = query_head // VARLEN_GQA_DECODE_HEAD_GROUP
                scores = (
                    q[batch, query_head].float()
                    @ k[begin:end, key_head].float().transpose(0, 1)
                ) * VARLEN_GQA_DECODE_SCALE
                maximum = scores.max()
                probability = torch.exp(scores - maximum)
                denominator = probability.sum() + sink[query_head]
                expected_output[batch, query_head] = (
                    probability @ v[begin:end, key_head].float() / denominator
                ).to(torch.float16)
                block_count = math.ceil(
                    (end - begin) / VARLEN_GQA_DECODE_BLOCK_SIZE
                )
                for block in range(block_count):
                    block_begin = block * VARLEN_GQA_DECODE_BLOCK_SIZE
                    block_end = min(
                        block_begin + VARLEN_GQA_DECODE_BLOCK_SIZE,
                        end - begin,
                    )
                    expected_logits[batch, query_head, block] = torch.exp(
                        scores[block_begin:block_end].max() - maximum
                    ) / denominator
        return expected_output, expected_logits

    generated_call()
    expected_output, expected_logits = reference()
    torch.cuda.synchronize()
    errors = (
        (output - expected_output).abs().max().item(),
        (block_logits - expected_logits).abs().max().item(),
    )
    if errors[0] > 3.0e-2 or errors[1] > 3.0e-3:
        raise RuntimeError(
            f"{target_name} varlen GQA decode logits comparison failed: {errors}"
        )
    p50, p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} varlen GQA decode with sink/logits numerical comparison: "
        f"PASS (output/logits errors={errors})"
    )
    print(
        f"{target_name} varlen GQA decode with sink/logits runtime-metadata "
        f"performance (CUDA Event): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(
        f"{target_name} varlen GQA decode with sink/logits upstream baseline: "
        "unavailable"
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


def _run_paged_splitk_attention(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("paged split-K attention has no matched multi-kernel adapter")
    sequence_lengths = torch.tensor(
        PAGED_SEQUENCE_LENGTHS, device="cuda", dtype=torch.int32
    )
    page_counts = torch.div(
        sequence_lengths + PAGED_PAGE_SIZE - 1,
        PAGED_PAGE_SIZE,
        rounding_mode="floor",
    )
    page_offsets = torch.zeros(
        (PAGED_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    page_offsets[1:] = page_counts.cumsum(0)
    total_pages = int(page_offsets[-1].item())
    page_indices = torch.randperm(
        total_pages, device="cuda", dtype=torch.int64
    ).to(torch.int32)
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
    host_page_counts = tuple(int(count) for count in page_counts.cpu())
    host_page_offsets = tuple(int(offset) for offset in page_offsets.cpu())
    split_offsets = []
    for batch, page_count in enumerate(host_page_counts):
        page_begin = host_page_offsets[batch]
        for split in range(PAGED_SPLITS):
            split_offsets.append(
                page_begin + page_count * split // PAGED_SPLITS
            )
    split_offsets.append(total_pages)
    split_offsets = torch.tensor(
        split_offsets, device="cuda", dtype=torch.int32
    )
    partial_artifact = intent.compile(
        paged_gqa_decode_partials,
        constexprs={
            "PAGE_SIZE": PAGED_PAGE_SIZE,
            "HEAD_GROUP": PAGED_HEAD_GROUP,
            "SPLITS": PAGED_SPLITS,
            "BATCH_SIZE": PAGED_BATCH,
        },
        target=target,
        compiler=compiler,
    )
    reduce_artifact = intent.compile(
        splitk_attention_reduce, target=target, compiler=compiler
    )
    partial_lse = torch.empty(
        (PAGED_BATCH, PAGED_QUERY_HEADS, PAGED_SPLITS),
        device="cuda",
        dtype=torch.float32,
    )
    partial_output = torch.empty(
        (
            PAGED_BATCH,
            PAGED_QUERY_HEADS,
            PAGED_SPLITS,
            PAGED_HEAD_DIMENSION,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (PAGED_BATCH, PAGED_QUERY_HEADS, PAGED_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.bfloat16,
    )
    partial_call = prepare_kernel_call(
        partial_artifact,
        (
            q,
            key_cache,
            value_cache,
            page_offsets,
            page_indices,
            sequence_lengths,
            split_offsets,
            PAGED_SCALE,
        ),
        (partial_lse, partial_output),
    )
    reduce_call = prepare_kernel_call(
        reduce_artifact, (partial_output, partial_lse), output
    )

    def pipeline():
        partial_call()
        reduce_call()

    def reference() -> torch.Tensor:
        expected = torch.empty_like(output)
        key_heads = torch.arange(
            PAGED_QUERY_HEADS, device="cuda"
        ) // PAGED_HEAD_GROUP
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
            expected[batch] = F.scaled_dot_product_attention(
                q[batch][None, :, None, :],
                key[:, key_heads].permute(1, 0, 2)[None],
                value[:, key_heads].permute(1, 0, 2)[None],
                scale=PAGED_SCALE,
            )[0, :, 0].to(torch.bfloat16)
        return expected

    pipeline()
    expected = reference()
    torch.cuda.synchronize()
    error = (output.float() - expected.float()).abs().max().item()
    if error > 4.0e-2:
        raise RuntimeError(
            f"{target_name} paged split-K attention comparison failed: {error}"
        )
    p50, p95 = benchmark(
        pipeline, warmup=3, repetitions=100, cuda_graph=False
    )
    for artifact in (partial_artifact, reduce_artifact):
        print_artifact(artifact, target_name)
    print(
        f"{target_name} paged split-K GQA decode numerical comparison: PASS "
        f"(generated/reference={error})"
    )
    print(
        f"{target_name} paged split-K GQA decode end-to-end GPU pipeline "
        f"performance (CUDA Event): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} paged split-K GQA decode upstream baseline: unavailable")


def _run_paged_mla_decode(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("paged MLA decode has no same-ABI upstream adapter")
    sequence_lengths = torch.tensor(
        PAGED_MLA_SEQUENCE_LENGTHS, device="cuda", dtype=torch.int32
    )
    page_counts = torch.div(
        sequence_lengths + PAGED_MLA_PAGE_SIZE - 1,
        PAGED_MLA_PAGE_SIZE,
        rounding_mode="floor",
    )
    page_offsets = torch.zeros(
        (PAGED_MLA_BATCH + 1,), device="cuda", dtype=torch.int32
    )
    page_offsets[1:] = page_counts.cumsum(0)
    total_pages = int(page_offsets[-1].item())
    page_indices = torch.randperm(
        total_pages, device="cuda", dtype=torch.int64
    ).to(torch.int32)
    q_latent = torch.randn(
        (
            PAGED_MLA_BATCH,
            PAGED_MLA_QUERY_HEADS,
            PAGED_MLA_LATENT_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    q_rope = torch.randn(
        (
            PAGED_MLA_BATCH,
            PAGED_MLA_QUERY_HEADS,
            PAGED_MLA_ROPE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    latent_cache = torch.randn(
        (
            total_pages,
            PAGED_MLA_PAGE_SIZE,
            PAGED_MLA_KV_HEADS,
            PAGED_MLA_LATENT_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    rope_cache = torch.randn(
        (
            total_pages,
            PAGED_MLA_PAGE_SIZE,
            PAGED_MLA_KV_HEADS,
            PAGED_MLA_ROPE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    artifact = intent.compile(
        paged_mla_decode,
        constexprs={
            "PAGE_SIZE": PAGED_MLA_PAGE_SIZE,
            "HEAD_GROUP": PAGED_MLA_HEAD_GROUP,
        },
        target=target,
        compiler=compiler,
    )
    arguments = (
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        page_offsets,
        page_indices,
        sequence_lengths,
        PAGED_MLA_SCALE,
    )

    def reference() -> torch.Tensor:
        expected = torch.empty_like(q_latent)
        key_heads = torch.arange(
            PAGED_MLA_QUERY_HEADS, device="cuda"
        ) // PAGED_MLA_HEAD_GROUP
        for batch in range(PAGED_MLA_BATCH):
            begin = int(page_offsets[batch].item())
            end = int(page_offsets[batch + 1].item())
            length = int(sequence_lengths[batch].item())
            physical_pages = page_indices[begin:end].long()
            latent = latent_cache[physical_pages].reshape(
                -1, PAGED_MLA_KV_HEADS, PAGED_MLA_LATENT_DIMENSION
            )[:length]
            rope = rope_cache[physical_pages].reshape(
                -1, PAGED_MLA_KV_HEADS, PAGED_MLA_ROPE_DIMENSION
            )[:length]
            content = torch.einsum(
                "hd,khd->hk", q_latent[batch].float(), latent[:, key_heads].float()
            )
            position = torch.einsum(
                "hd,khd->hk", q_rope[batch].float(), rope[:, key_heads].float()
            )
            probability = torch.softmax(
                (content + position) * PAGED_MLA_SCALE, dim=1
            )
            expected[batch] = torch.einsum(
                "hk,khd->hd", probability, latent[:, key_heads].float()
            ).to(torch.float16)
        return expected

    _compare(
        artifact=artifact,
        arguments=arguments,
        reference=reference,
        target_name=target_name,
        kernel_name="paged MLA decode",
        tolerance=4.0e-2,
        upstream=None,
        expected_dtype=torch.float16,
    )


def _run_block_sparse_attention(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("block-sparse attention has no comparable upstream adapter")
    q = torch.randn(
        (BLOCK_SPARSE_BATCH, BLOCK_SPARSE_QUERY_HEADS, BLOCK_SPARSE_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    k = torch.randn(
        (
            BLOCK_SPARSE_BATCH,
            BLOCK_SPARSE_SEQUENCE,
            BLOCK_SPARSE_KV_HEADS,
            BLOCK_SPARSE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float16,
    ) * 0.5
    v = torch.randn_like(k) * 0.5
    total_blocks = BLOCK_SPARSE_SEQUENCE // BLOCK_SPARSE_BLOCK_SIZE
    selected_base = torch.arange(
        total_blocks - 1,
        total_blocks - BLOCK_SPARSE_SELECTED_BLOCKS - 1,
        -1,
        device="cuda",
        dtype=torch.int32,
    )
    block_indices = torch.empty(
        (
            BLOCK_SPARSE_BATCH,
            BLOCK_SPARSE_KV_HEADS,
            BLOCK_SPARSE_SELECTED_BLOCKS,
        ),
        device="cuda",
        dtype=torch.int32,
    )
    for batch in range(BLOCK_SPARSE_BATCH):
        for key_head in range(BLOCK_SPARSE_KV_HEADS):
            block_indices[batch, key_head] = torch.roll(
                selected_base, shifts=(batch + key_head) % BLOCK_SPARSE_SELECTED_BLOCKS
            )
    cache_lengths = torch.tensor(
        [BLOCK_SPARSE_SEQUENCE - 3 * index for index in range(BLOCK_SPARSE_BATCH)],
        device="cuda",
        dtype=torch.int32,
    )
    split_offsets = torch.arange(
        0,
        BLOCK_SPARSE_SELECTED_BLOCKS + 1,
        BLOCK_SPARSE_SELECTED_BLOCKS // BLOCK_SPARSE_SPLITS,
        device="cuda",
        dtype=torch.int32,
    )
    partial_artifact = intent.compile(
        block_sparse_gqa_decode_partials,
        target=target,
        compiler=compiler,
        constexprs={
            "HEAD_GROUP": BLOCK_SPARSE_HEAD_GROUP,
            "BLOCK_SIZE": BLOCK_SPARSE_BLOCK_SIZE,
            "SPLITS": BLOCK_SPARSE_SPLITS,
        },
    )
    combine_artifact = intent.compile(
        block_sparse_gqa_decode_combine,
        target=target,
        compiler=compiler,
    )
    partial_lse = torch.empty(
        (
            BLOCK_SPARSE_BATCH,
            BLOCK_SPARSE_QUERY_HEADS,
            BLOCK_SPARSE_SPLITS,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    partial_output = torch.empty(
        (
            BLOCK_SPARSE_BATCH,
            BLOCK_SPARSE_QUERY_HEADS,
            BLOCK_SPARSE_SPLITS,
            BLOCK_SPARSE_DIMENSION,
        ),
        device="cuda",
        dtype=torch.float32,
    )
    output = torch.empty_like(q)
    partial_call = prepare_kernel_call(
        partial_artifact,
        (q, k, v, block_indices, cache_lengths, split_offsets, BLOCK_SPARSE_SCALE),
        (partial_lse, partial_output),
    )
    combine_call = prepare_kernel_call(
        combine_artifact,
        (partial_lse, partial_output),
        output,
    )

    def generated_pipeline() -> torch.Tensor:
        partial_call()
        combine_call()
        return output

    def reference() -> torch.Tensor:
        expected = torch.empty_like(q)
        token_offsets = torch.arange(
            BLOCK_SPARSE_BLOCK_SIZE, device="cuda", dtype=torch.int64
        )
        for batch in range(BLOCK_SPARSE_BATCH):
            cache_length = int(cache_lengths[batch].item())
            for query_head in range(BLOCK_SPARSE_QUERY_HEADS):
                key_head = query_head // BLOCK_SPARSE_HEAD_GROUP
                tokens = (
                    block_indices[batch, key_head].long()[:, None]
                    * BLOCK_SPARSE_BLOCK_SIZE
                    + token_offsets[None, :]
                ).reshape(-1)
                tokens = tokens[tokens < cache_length]
                scores = (
                    q[batch, query_head].float()
                    @ k[batch, tokens, key_head].float().transpose(0, 1)
                ) * BLOCK_SPARSE_SCALE
                probability = torch.softmax(scores, dim=0)
                expected[batch, query_head] = (
                    probability @ v[batch, tokens, key_head].float()
                ).to(torch.float16)
        return expected

    actual = generated_pipeline()
    expected = reference()
    torch.cuda.synchronize()
    error = (actual - expected).abs().max().item()
    if error > 7.5e-2:
        raise RuntimeError(
            f"{target_name} block-sparse GQA decode numerical comparison failed: {error}"
        )
    p50, p95 = benchmark(
        generated_pipeline,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    for artifact in (partial_artifact, combine_artifact):
        print_artifact(artifact, target_name)
    print(
        f"{target_name} block-sparse GQA decode numerical comparison: PASS "
        f"(generated/reference={error})"
    )
    print(
        f"{target_name} block-sparse GQA decode end-to-end GPU pipeline performance "
        f"(CUDA Event): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} block-sparse GQA decode upstream baseline: unavailable")


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
        reference=lambda: torch.cat((x[1:], x[:1]), dim=0),
        target_name=target_name,
        kernel_name="derived scalar offset index",
        tolerance=0.0,
        upstream=None,
    )


def _run_matrix_transpose(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("matrix transpose has no upstream adapter")
    x = torch.randn(
        (TRANSPOSE_ROWS, TRANSPOSE_COLUMNS),
        device="cuda",
        dtype=torch.float16,
    )
    artifact = intent.compile(matrix_transpose, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x,),
        reference=lambda: x.transpose(0, 1).contiguous(),
        target_name=target_name,
        kernel_name="canonical matrix transpose",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.float16,
    )


def _run_batched_row_affine(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("batched row affine has no upstream adapter")
    x = torch.randn(
        (AFFINE_BATCH, AFFINE_ROWS, AFFINE_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    )
    scale = torch.randn(
        (AFFINE_BATCH, AFFINE_ROWS), device="cuda", dtype=torch.float32
    )
    bias = torch.randn_like(scale)
    artifact = intent.compile(batched_row_affine, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x, scale, bias),
        reference=lambda: x * scale[:, :, None] + bias[:, :, None],
        target_name=target_name,
        kernel_name="parallel domain product row affine",
        tolerance=2.0e-6,
        upstream=None,
        expected_dtype=torch.float32,
    )


def _run_atomic_compare_exchange(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("atomic compare exchange has no upstream adapter")
    state = torch.where(
        torch.arange(CAS_SLOTS, device="cuda") % 3 == 0,
        torch.zeros((), device="cuda", dtype=torch.int32),
        torch.full((), 2, device="cuda", dtype=torch.int32),
    )
    initial = state.clone()
    artifact = intent.compile(claim_zero_slots, target=target, compiler=compiler)

    def reference():
        expected_state = torch.where(initial == 0, 1, initial)
        state_error = (state - expected_state).abs().max().item()
        if state_error != 0:
            raise RuntimeError(
                f"{target_name} compare-and-swap state transition failed: "
                f"{state_error}"
            )
        return initial

    _compare(
        artifact=artifact,
        arguments=(state,),
        reference=reference,
        target_name=target_name,
        kernel_name="scalar atomic compare exchange",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.int32,
    )


def _run_record_fields(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("record fields have no upstream adapter")
    x = torch.randn((RECORD_ELEMENTS,), device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    artifact = intent.compile(paired_sum_product, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x, y),
        reference=lambda: (x + y) + (x * y),
        target_name=target_name,
        kernel_name="named SSA record fields",
        tolerance=2.0e-6,
        upstream=None,
        expected_dtype=torch.float32,
    )


def _run_scalar_while(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("scalar while has no upstream adapter")
    values = torch.randint(
        1,
        1 << 30,
        (WHILE_ELEMENTS,),
        device="cuda",
        dtype=torch.int32,
    )
    artifact = intent.compile(integer_log2_floor, target=target, compiler=compiler)

    def reference() -> torch.Tensor:
        working = values.clone()
        exponent = torch.zeros_like(values)
        while torch.any(working > 1).item():
            active = working > 1
            working = torch.where(active, working // 2, working)
            exponent += active.to(torch.int32)
        return exponent

    _compare(
        artifact=artifact,
        arguments=(values,),
        reference=reference,
        target_name=target_name,
        kernel_name="scalar carried-state while",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.int32,
    )


def _run_ordered_prefix(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("ordered prefix has no upstream adapter")
    x = torch.randn(
        (ORDERED_PREFIX_BATCH, ORDERED_PREFIX_ROWS, ORDERED_PREFIX_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    artifact = intent.compile(ordered_product_prefix, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(x,),
        reference=lambda: torch.cumsum(x.flatten(1), dim=1).reshape_as(x),
        target_name=target_name,
        kernel_name="ordered Cartesian prefix",
        tolerance=2.0e-5,
        upstream=None,
        expected_dtype=torch.float32,
        cuda_graph=False,
    )


def _run_boolean_reduction(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("boolean reduction has no upstream adapter")
    shape_source = torch.empty(
        (BOOLEAN_REDUCTION_ROWS, BOOLEAN_REDUCTION_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    )
    artifact = intent.compile(
        row_boolean_reduction, target=target, compiler=compiler
    )
    _compare(
        artifact=artifact,
        arguments=(shape_source,),
        reference=lambda: torch.full(
            (BOOLEAN_REDUCTION_ROWS,), 3, device="cuda", dtype=torch.int32
        ),
        target_name=target_name,
        kernel_name="boolean any/all reduction",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.int32,
    )


def _run_value_select(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    if upstream is not None:
        raise RuntimeError("value select has no upstream adapter")
    shape_source = torch.empty(
        (SELECT_ROWS, SELECT_COLUMNS), device="cuda", dtype=torch.float32
    )
    artifact = intent.compile(
        alternating_signed_indices, target=target, compiler=compiler
    )

    def reference() -> torch.Tensor:
        columns = torch.arange(
            SELECT_COLUMNS, device="cuda", dtype=torch.int32
        ).unsqueeze(0)
        rows = torch.arange(SELECT_ROWS, device="cuda", dtype=torch.int32)
        signs = torch.where(rows % 2 == 0, 1, -1).unsqueeze(1)
        return columns * signs

    _compare(
        artifact=artifact,
        arguments=(shape_source,),
        reference=reference,
        target_name=target_name,
        kernel_name="canonical SSA value select",
        tolerance=0.0,
        upstream=None,
        expected_dtype=torch.int32,
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
    upstream_error = None
    upstream_p50 = None
    upstream_p95 = None
    if upstream is not None:
        upstream_weight = torch.zeros_like(grad_weight)
        upstream_output = upstream((indices, grad_output, upstream_weight))
        torch.cuda.synchronize()
        upstream_error = (upstream_output - expected).abs().max().item()
        if upstream_error > 2.0e-6:
            raise RuntimeError(
                f"{target_name} embedding backward upstream comparison failed: "
                f"{upstream_error}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream((indices, grad_output, upstream_weight)),
            warmup=3,
            repetitions=100,
            cuda_graph=False,
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} embedding backward atomic numerical comparison: PASS "
        f"(max error={error})"
    )
    print(
        f"{target_name} embedding backward atomic kernel-only performance "
        f"(CUDA Graph): p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    if upstream_p50 is None:
        print(f"{target_name} embedding backward atomic upstream baseline: unavailable")
    else:
        print(
            f"{target_name} embedding backward atomic upstream comparison: PASS "
            f"(upstream/reference={upstream_error}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={p50 / upstream_p50:.4f}x)"
        )


def _run_index_select_rows(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    source = torch.randn(
        (INDEX_SELECT_SOURCE_ROWS, INDEX_SELECT_FEATURES),
        device="cuda",
        dtype=torch.float16,
    )
    indices = torch.arange(
        0,
        INDEX_SELECT_ROWS * 2,
        2,
        device="cuda",
        dtype=torch.int64,
    )
    artifact = intent.compile(index_select_rows, target=target, compiler=compiler)
    _compare(
        artifact=artifact,
        arguments=(source, indices),
        reference=lambda: source[indices],
        target_name=target_name,
        kernel_name="row index-select",
        tolerance=0.0,
        upstream=upstream,
        expected_dtype=torch.float16,
    )


def _run_scaled_index_add(
    compiler: str, target: Target, target_name: str, upstream: Upstream | None
) -> None:
    initial = torch.randn(
        (
            SCALED_ADD_DESTINATION_ROWS,
            SCALED_ADD_INNER_ROWS,
            SCALED_ADD_FEATURES,
        ),
        device="cuda",
        dtype=torch.float16,
    )
    source = torch.randn(
        (SCALED_ADD_SOURCE_ROWS, SCALED_ADD_INNER_ROWS, SCALED_ADD_FEATURES),
        device="cuda",
        dtype=torch.float16,
    )
    indices = torch.arange(
        0,
        SCALED_ADD_SOURCE_ROWS * 2,
        2,
        device="cuda",
        dtype=torch.int64,
    )
    scaling = torch.randn(
        (SCALED_ADD_FEATURES,), device="cuda", dtype=torch.float16
    )
    alpha = 1.0
    generated_destination = initial.clone()
    expected_selected = torch.empty_like(source)
    reference_chunk = 1024
    for begin in range(0, SCALED_ADD_SOURCE_ROWS, reference_chunk):
        end = min(begin + reference_chunk, SCALED_ADD_SOURCE_ROWS)
        expected_selected[begin:end] = (
            initial[indices[begin:end]].float()
            + alpha
            * scaling.float()[None, None, :]
            * source[begin:end].float()
        ).half()
    artifact = intent.compile(
        scaled_index_add_unique, target=target, compiler=compiler
    )
    arguments = (
        generated_destination,
        indices,
        source,
        scaling,
        alpha,
    )
    artifact.run(*arguments)
    torch.cuda.synchronize()
    error = 0.0
    for begin in range(0, SCALED_ADD_SOURCE_ROWS, reference_chunk):
        end = min(begin + reference_chunk, SCALED_ADD_SOURCE_ROWS)
        error = max(
            error,
            (
                generated_destination[indices[begin:end]]
                - expected_selected[begin:end]
            ).abs().max().item(),
        )
    if error != 0.0:
        raise RuntimeError(
            f"{target_name} scaled index-add comparison failed: {error}"
        )
    generated_call = prepare_kernel_call(artifact, arguments, ())
    generated_p50, generated_p95 = benchmark(
        generated_call,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
        prepare=lambda: generated_destination.copy_(initial),
    )
    upstream_error = None
    upstream_p50 = None
    upstream_p95 = None
    if upstream is not None:
        upstream_destination = initial.clone()
        upstream_output = upstream(
            (upstream_destination, indices, source, scaling, alpha)
        )
        torch.cuda.synchronize()
        upstream_error = 0.0
        for begin in range(0, SCALED_ADD_SOURCE_ROWS, reference_chunk):
            end = min(begin + reference_chunk, SCALED_ADD_SOURCE_ROWS)
            upstream_error = max(
                upstream_error,
                (
                    upstream_output[indices[begin:end]]
                    - expected_selected[begin:end]
                ).abs().max().item(),
            )
        if upstream_error != 0.0:
            raise RuntimeError(
                f"{target_name} scaled index-add upstream comparison failed: "
                f"{upstream_error}"
            )
        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream(
                (upstream_destination, indices, source, scaling, alpha)
            ),
            warmup=3,
            repetitions=100,
            cuda_graph=False,
            prepare=lambda: upstream_destination.copy_(initial),
        )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} scaled index-add numerical comparison: PASS "
        f"(generated/reference={error})"
    )
    print(
        f"{target_name} scaled index-add kernel-only performance (CUDA Event): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_p50 is None:
        print(f"{target_name} scaled index-add upstream baseline: unavailable")
    else:
        print(
            f"{target_name} scaled index-add upstream comparison: PASS "
            f"(upstream/reference={upstream_error}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
        )


EXTENDED_RUNNERS: dict[str, Runner] = {
    "absorbed_mla_prefill": _run_absorbed_mla_prefill,
    "attention_bias": _run_attention_bias,
    "attention_backward": _run_attention_backward,
    "atomic_compare_exchange": _run_atomic_compare_exchange,
    "batched_row_affine": _run_batched_row_affine,
    "batched_gemm": _run_batched_gemm,
    "bf16_gemm": _run_bf16_gemm,
    "block_scaled_matmul": _run_block_scaled_matmul,
    "block_sparse_attention": _run_block_sparse_attention,
    "boolean_reduction": _run_boolean_reduction,
    "causal_conv1d": _run_causal_conv1d,
    "causal_conv1d_backward": _run_causal_conv1d_backward,
    "conv1d": _run_conv1d,
    "conv2d": _run_conv2d,
    "continuous_gqa_decode": _run_continuous_gqa_decode,
    "cross_entropy": _run_cross_entropy,
    "dropout_residual_rms_norm": _run_dropout_residual_rms_norm,
    "dual_gemm": _run_dual_gemm,
    "embedding_backward_atomic": _run_embedding_backward_atomic,
    "embedding_forward_lookup": _run_embedding_forward_lookup,
    "fp8_gemm": _run_fp8_gemm,
    "fp8_mqa_logits": _run_fp8_mqa_logits,
    "fused_add_rms_norm": _run_fused_add_rms_norm,
    "grouped_gemm": _run_grouped_gemm,
    "grouped_query_head_add": _run_grouped_query_head_add,
    "insertion_top_k": _run_insertion_top_k,
    "index_select_rows": _run_index_select_rows,
    "layer_norm": _run_layer_norm,
    "layer_norm_backward": _run_layer_norm_backward,
    "logsumexp": _run_logsumexp,
    "matrix_transpose": _run_matrix_transpose,
    "mamba_chunk_scan": _run_mamba_chunk_scan,
    "splitk_attention_reduce": _run_splitk_attention_reduce,
    "mla_prefill": _run_mla_prefill,
    "mla_head_projection": _run_mla_head_projection,
    "online_softmax": _run_online_softmax,
    "ordered_prefix": _run_ordered_prefix,
    "paged_attention": _run_paged_attention,
    "paged_mla_decode": _run_paged_mla_decode,
    "paged_splitk_attention": _run_paged_splitk_attention,
    "quantized_gemm": _run_quantized_gemm,
    "rms_norm": _run_rms_norm,
    "record_fields": _run_record_fields,
    "scalar_while": _run_scalar_while,
    "sparse_2to4_gemm": _run_sparse_2to4_gemm,
    "scaled_index_add": _run_scaled_index_add,
    "scalar_table_lookup": _run_scalar_table_lookup,
    "selective_scan": _run_selective_scan,
    "shifted_row_copy": _run_shifted_row_copy,
    "sorted_nucleus_cutoff": _run_sorted_nucleus_cutoff,
    "swiglu_backward": _run_swiglu_backward,
    "swiglu_forward": _run_swiglu_forward,
    "token_sparse_mla_prefill": _run_token_sparse_mla_prefill,
    "varlen_attention": _run_varlen_attention,
    "varlen_gqa_prefill": _run_varlen_gqa_prefill,
    "varlen_gqa_decode_logits": _run_varlen_gqa_decode_logits,
    "varlen_gqa_rope_prefill": _run_varlen_gqa_rope_prefill,
    "value_select": _run_value_select,
    "w4a8_packed": _run_w4a8_packed,
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
