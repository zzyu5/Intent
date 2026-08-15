from __future__ import annotations

from collections.abc import Callable
import math

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
from kernels.backward.group_norm_silu import BATCH as GROUP_NORM_BATCH
from kernels.backward.group_norm_silu import CHANNELS as GROUP_NORM_CHANNELS
from kernels.backward.group_norm_silu import CHANNELS_PER_GROUP
from kernels.backward.group_norm_silu import GROUPS as GROUP_NORM_GROUPS
from kernels.backward.group_norm_silu import SPATIAL as GROUP_NORM_SPATIAL
from kernels.backward.group_norm_silu import group_norm_silu_backward
from kernels.cache.reshape_and_cache import BLOCKS as CACHE_BLOCKS
from kernels.cache.reshape_and_cache import BLOCK_SIZE as CACHE_BLOCK_SIZE
from kernels.cache.reshape_and_cache import HEAD_DIMENSION as CACHE_HEAD_DIMENSION
from kernels.cache.reshape_and_cache import HEADS as CACHE_HEADS
from kernels.cache.reshape_and_cache import TOKENS as CACHE_TOKENS
from kernels.cache.reshape_and_cache import reshape_and_cache
from kernels.compaction.nonzero import ROWS as NONZERO_ROWS
from kernels.compaction.nonzero import VALUES as NONZERO_VALUES
from kernels.compaction.nonzero import compact_nonzero_rows
from kernels.compaction.unique_consecutive import ROWS as UNIQUE_ROWS
from kernels.compaction.unique_consecutive import VALUES as UNIQUE_VALUES
from kernels.compaction.unique_consecutive import unique_consecutive_rows
from kernels.factorization.cholesky import BATCH as CHOLESKY_BATCH
from kernels.factorization.cholesky import SIZE as CHOLESKY_SIZE
from kernels.factorization.cholesky import batched_cholesky_lower
from kernels.factorization.householder_qr import BATCH as QR_BATCH
from kernels.factorization.householder_qr import COLUMNS as QR_COLUMNS
from kernels.factorization.householder_qr import ROWS as QR_ROWS
from kernels.factorization.householder_qr import batched_householder_qr
from kernels.optimization.adafactor import COLUMNS as ADAFACTOR_COLUMNS
from kernels.optimization.adafactor import ROWS as ADAFACTOR_ROWS
from kernels.optimization.adafactor import adafactor_apply
from kernels.optimization.adafactor import adafactor_update_columns
from kernels.optimization.adafactor import adafactor_update_rows
from kernels.optimization.adamw import PARAMETERS as ADAMW_PARAMETERS
from kernels.optimization.adamw import adamw_update
from kernels.ragged.nested_pool import DOCUMENTS as NESTED_DOCUMENTS
from kernels.ragged.nested_pool import FEATURES as NESTED_FEATURES
from kernels.ragged.nested_pool import SENTENCES as NESTED_SENTENCES
from kernels.ragged.nested_pool import TOKENS as NESTED_TOKENS
from kernels.ragged.nested_pool import nested_jagged_mean_pool
from kernels.routing.moe_align import BLOCK_SIZE as MOE_ALIGN_BLOCK_SIZE
from kernels.routing.moe_align import EXPERTS as MOE_ALIGN_EXPERTS
from kernels.routing.moe_align import PADDED_ROUTES as MOE_ALIGN_PADDED_ROUTES
from kernels.routing.moe_align import TOKENS as MOE_ALIGN_TOKENS
from kernels.routing.moe_align import TOP_K as MOE_ALIGN_TOP_K
from kernels.routing.moe_align import moe_count_routes
from kernels.routing.moe_align import moe_mark_expert_blocks
from kernels.routing.moe_align import moe_prefix_routes
from kernels.routing.moe_align import moe_scatter_routes
from kernels.activation.swiglu import FEATURES as SWIGLU_FEATURES
from kernels.activation.swiglu import TOKENS as SWIGLU_TOKENS
from kernels.activation.swiglu import swiglu_forward
from kernels.clustering.kmeans import CLUSTERS
from kernels.clustering.kmeans import FEATURES as KMEANS_FEATURES
from kernels.clustering.kmeans import POINTS
from kernels.clustering.kmeans import kmeans_assign
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import K as GEMM_K
from kernels.contraction.gemm import M as GEMM_M
from kernels.contraction.gemm import N as GEMM_N
from kernels.contraction.gemm import gemm
from kernels.convolution.direct import CONV2D_BATCH
from kernels.convolution.direct import CONV2D_FILTER_HEIGHT
from kernels.convolution.direct import CONV2D_FILTER_WIDTH
from kernels.convolution.direct import CONV2D_HEIGHT
from kernels.convolution.direct import CONV2D_WIDTH
from kernels.convolution.direct import CAUSAL_CONV_WIDTH
from kernels.convolution.direct import CAUSAL_UPDATE_BATCH
from kernels.convolution.direct import CAUSAL_UPDATE_CHANNELS
from kernels.convolution.direct import causal_depthwise_conv1d_update
from kernels.convolution.direct import conv2d_same
from kernels.dynamic_programming.smith_waterman import BATCH as SW_BATCH
from kernels.dynamic_programming.smith_waterman import GAP_SCORE
from kernels.dynamic_programming.smith_waterman import MATCH_SCORE
from kernels.dynamic_programming.smith_waterman import MISMATCH_SCORE
from kernels.dynamic_programming.smith_waterman import QUERY_LENGTH
from kernels.dynamic_programming.smith_waterman import REFERENCE_LENGTH
from kernels.dynamic_programming.smith_waterman import smith_waterman_score
from kernels.dynamic_programming.viterbi import BATCH as VITERBI_BATCH
from kernels.dynamic_programming.viterbi import STATES
from kernels.dynamic_programming.viterbi import TIME_STEPS
from kernels.dynamic_programming.viterbi import viterbi_decode
from kernels.layout.transpose import COLUMNS as TRANSPOSE_COLUMNS
from kernels.layout.transpose import ROWS as TRANSPOSE_ROWS
from kernels.layout.transpose import matrix_transpose
from kernels.normalization.layer_norm import FEATURES as LAYER_FEATURES
from kernels.normalization.layer_norm import ROWS as LAYER_ROWS
from kernels.normalization.layer_norm import weighted_layer_norm
from kernels.normalization.softmax import COLUMNS as SOFTMAX_COLUMNS
from kernels.normalization.softmax import ROWS as SOFTMAX_ROWS
from kernels.normalization.softmax import stable_softmax
from kernels.position.rope import HALF_DIMENSION
from kernels.position.rope import HEAD_DIMENSION
from kernels.position.rope import ROPE_BATCH
from kernels.position.rope import ROPE_KEY_HEADS
from kernels.position.rope import ROPE_QUERY_HEADS
from kernels.position.rope import ROPE_SEQUENCE
from kernels.position.rope import rotary_embedding_flat
from kernels.position.rope import rotary_qk_inplace
from kernels.position.rope import rotary_qk_partial_inplace
from kernels.simulation.monte_carlo import PATHS
from kernels.simulation.monte_carlo import STEPS
from kernels.simulation.monte_carlo import barrier_option_paths
from kernels.sorting.bitonic import ROWS as SORT_ROWS
from kernels.sorting.bitonic import VALUES as SORT_VALUES
from kernels.sorting.bitonic import bitonic_sort_rows
from kernels.sparse.csr_spmv import COLUMNS as SPMV_COLUMNS
from kernels.sparse.csr_spmv import NONZEROS
from kernels.sparse.csr_spmv import NONZEROS_PER_ROW
from kernels.sparse.csr_spmv import ROWS as SPMV_ROWS
from kernels.sparse.csr_spmv import csr_spmv
from kernels.spectral.fft import BATCH as FFT_BATCH
from kernels.spectral.fft import FFT_SIZE
from kernels.spectral.fft import LOG_FFT_SIZE
from kernels.spectral.fft import radix2_fft
from kernels.statistics.histogram import BINS
from kernels.statistics.histogram import SAMPLES
from kernels.statistics.histogram import histogram_256
from kernels.streaming.attention import SCALE as ATTENTION_SCALE
from kernels.streaming.attention import flash_attention_fwd
from kernels.streaming.online_softmax import COLUMNS as ONLINE_COLUMNS
from kernels.streaming.online_softmax import ROWS as ONLINE_ROWS
from kernels.streaming.online_softmax import streamed_online_softmax
from kernels.variants.activation import swiglu_forward_helper
from kernels.variants.contraction import conv2d_reduce_order
from kernels.variants.contraction import gemm_loop_interchange
from kernels.variants.indexing import rotary_embedding_equivalent_index
from kernels.variants.layout import matrix_transpose_scalar_domains
from kernels.variants.normalization import stable_softmax_online
from kernels.variants.normalization import weighted_layer_norm_second_moment
from kernels.variants.streaming import flash_attention_inline_fwd
from kernels.variants.streaming import flash_attention_select_fwd
from kernels.variants.streaming import streamed_online_softmax_inline
from kernels.vision.nms import BATCH as NMS_BATCH
from kernels.vision.nms import BOXES
from kernels.vision.nms import greedy_nms
from kernels.vision.roi_align import BATCH as ROI_BATCH
from kernels.vision.roi_align import CHANNELS
from kernels.vision.roi_align import HEIGHT
from kernels.vision.roi_align import POOLED_HEIGHT
from kernels.vision.roi_align import POOLED_WIDTH
from kernels.vision.roi_align import ROIS
from kernels.vision.roi_align import WIDTH
from kernels.vision.roi_align import roi_align_center_sample

from .support import benchmark
from .support import prepare_kernel_call
from .support import print_artifact
from .evaluation import Runner
from .evaluation import outputs as _outputs
from .evaluation import report_pipeline as _report_pipeline
from .evaluation import require_close as _require_close
from .evaluation import run_generated as _run_generated
from .evaluation import run_variant as _run_variant
from .decomposition import DECOMPOSITION_RUNNERS
from .small_operators import SMALL_OPERATOR_RUNNERS


def _run_histogram(compiler: str, target: Target, target_name: str) -> None:
    samples = torch.randint(0, BINS, (SAMPLES,), device="cuda", dtype=torch.uint8)
    histogram = torch.zeros((BINS,), device="cuda", dtype=torch.int32)
    artifact = intent.compile(histogram_256, target=target, compiler=compiler)
    artifact.run(samples, histogram)
    expected = torch.bincount(samples.long(), minlength=BINS).to(torch.int32)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=histogram,
        expected=expected,
        tolerance=0.0,
        target_name=target_name,
        kernel_name="256-bin histogram",
    )
    launch = prepare_kernel_call(artifact, (samples, histogram), ())

    def clear_and_launch() -> None:
        histogram.zero_()
        launch()

    p50, p95 = benchmark(
        clear_and_launch,
        warmup=3,
        repetitions=100,
        cuda_graph=False,
    )
    print_artifact(artifact, target_name)
    print(
        f"{target_name} 256-bin histogram numerical comparison: PASS "
        f"(generated/reference={errors})"
    )
    print(
        f"{target_name} 256-bin histogram end-to-end performance (CUDA Event): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} 256-bin histogram upstream baseline: unavailable")


def _run_csr_spmv(compiler: str, target: Target, target_name: str) -> None:
    row_offsets = (
        torch.arange(SPMV_ROWS + 1, device="cuda", dtype=torch.int32)
        * NONZEROS_PER_ROW
    )
    column_indices = torch.randint(
        0, SPMV_COLUMNS, (NONZEROS,), device="cuda", dtype=torch.int32
    )
    values = torch.randn((NONZEROS,), device="cuda", dtype=torch.float32) * 0.1
    vector = torch.randn((SPMV_COLUMNS,), device="cuda", dtype=torch.float32)
    _run_generated(
        definition=csr_spmv,
        arguments=(row_offsets, column_indices, values, vector),
        reference=lambda: (
            values.reshape(SPMV_ROWS, NONZEROS_PER_ROW)
            * vector[column_indices.long()].reshape(SPMV_ROWS, NONZEROS_PER_ROW)
        ).sum(dim=1),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="CSR sparse matrix-vector product",
        tolerance=2.0e-5,
        cuda_graph=False,
    )


def _run_fft(compiler: str, target: Target, target_name: str) -> None:
    input_real = torch.randn(
        (FFT_BATCH, FFT_SIZE), device="cuda", dtype=torch.float32
    )
    input_imag = torch.randn_like(input_real)
    lanes = torch.arange(FFT_SIZE // 2, device="cuda", dtype=torch.float32)
    twiddle_real = torch.empty(
        (LOG_FFT_SIZE, FFT_SIZE // 2), device="cuda", dtype=torch.float32
    )
    twiddle_imag = torch.empty_like(twiddle_real)
    for stage in range(LOG_FFT_SIZE):
        span = 1 << (stage + 1)
        angle = -2.0 * math.pi * lanes / span
        twiddle_real[stage] = torch.cos(angle)
        twiddle_imag[stage] = torch.sin(angle)

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        transformed = torch.fft.fft(torch.complex(input_real, input_imag), dim=1)
        return transformed.real, transformed.imag

    _run_generated(
        definition=radix2_fft,
        arguments=(input_real, input_imag, twiddle_real, twiddle_imag),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="radix-2 FFT",
        tolerance=(3.0e-4, 3.0e-4),
        cuda_graph=False,
    )


def _run_bitonic_sort(compiler: str, target: Target, target_name: str) -> None:
    values = torch.randn(
        (SORT_ROWS, SORT_VALUES), device="cuda", dtype=torch.float32
    )
    _run_generated(
        definition=bitonic_sort_rows,
        arguments=(values,),
        reference=lambda: torch.sort(values, dim=1).values,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="row-wise bitonic sort",
        tolerance=0.0,
        cuda_graph=False,
    )


def _run_kmeans(compiler: str, target: Target, target_name: str) -> None:
    points = torch.randn(
        (POINTS, KMEANS_FEATURES), device="cuda", dtype=torch.float16
    ) * 0.1
    centroids = torch.randn(
        (CLUSTERS, KMEANS_FEATURES), device="cuda", dtype=torch.float16
    ) * 0.1

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        distances = (
            points.float().square().sum(dim=1, keepdim=True)
            + centroids.float().square().sum(dim=1)[None, :]
            - 2.0 * points.float() @ centroids.float().transpose(0, 1)
        )
        minimum, assignment = distances.min(dim=1)
        return assignment.to(torch.int32), minimum

    _run_generated(
        definition=kmeans_assign,
        arguments=(points, centroids),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="Lloyd k-means assignment",
        tolerance=(0.0, 2.0e-4),
        cuda_graph=False,
    )


def _run_viterbi(compiler: str, target: Target, target_name: str) -> None:
    emissions = torch.randn(
        (VITERBI_BATCH, TIME_STEPS, STATES),
        device="cuda",
        dtype=torch.float32,
    ) * 0.1
    transitions = torch.randn(
        (STATES, STATES), device="cuda", dtype=torch.float32
    ) * 0.1

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        previous = emissions[:, 0]
        predecessors = torch.empty(
            (VITERBI_BATCH, TIME_STEPS, STATES),
            device="cuda",
            dtype=torch.int64,
        )
        for time in range(1, TIME_STEPS):
            values, sources = (previous[:, :, None] + transitions[None, :, :]).max(
                dim=1
            )
            predecessors[:, time] = sources
            previous = values + emissions[:, time]
        score, state = previous.max(dim=1)
        path = torch.empty(
            (VITERBI_BATCH, TIME_STEPS), device="cuda", dtype=torch.int32
        )
        path[:, -1] = state.to(torch.int32)
        batch = torch.arange(VITERBI_BATCH, device="cuda")
        for time in range(TIME_STEPS - 1, 0, -1):
            state = predecessors[batch, time, state]
            path[:, time - 1] = state.to(torch.int32)
        return path, score

    _run_generated(
        definition=viterbi_decode,
        arguments=(emissions, transitions),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="Viterbi trellis decode",
        tolerance=(0.0, 2.0e-5),
        cuda_graph=False,
    )


def _run_smith_waterman(compiler: str, target: Target, target_name: str) -> None:
    query = torch.randint(
        0,
        20,
        (SW_BATCH, QUERY_LENGTH),
        device="cuda",
        dtype=torch.int32,
    )
    reference_sequence = torch.randint(
        0,
        20,
        (SW_BATCH, REFERENCE_LENGTH),
        device="cuda",
        dtype=torch.int32,
    )

    def reference() -> torch.Tensor:
        query_cpu = query.cpu()
        reference_cpu = reference_sequence.cpu()
        previous = torch.zeros((SW_BATCH, REFERENCE_LENGTH + 1), dtype=torch.int32)
        maximum = torch.zeros((SW_BATCH,), dtype=torch.int32)
        for row in range(1, QUERY_LENGTH + 1):
            current = torch.zeros_like(previous)
            for column in range(1, REFERENCE_LENGTH + 1):
                substitution = torch.where(
                    query_cpu[:, row - 1] == reference_cpu[:, column - 1],
                    MATCH_SCORE,
                    MISMATCH_SCORE,
                )
                cell = torch.maximum(
                    torch.zeros_like(maximum),
                    torch.maximum(
                        previous[:, column - 1] + substitution,
                        torch.maximum(
                            previous[:, column] + GAP_SCORE,
                            current[:, column - 1] + GAP_SCORE,
                        ),
                    ),
                )
                current[:, column] = cell
                maximum = torch.maximum(maximum, cell)
            previous = current
        return maximum.to(torch.int32).cuda()

    _run_generated(
        definition=smith_waterman_score,
        arguments=(query, reference_sequence),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="Smith-Waterman local alignment score",
        tolerance=0.0,
        cuda_graph=False,
    )


def _nms_reference(boxes: torch.Tensor, threshold: float) -> torch.Tensor:
    boxes_cpu = boxes.cpu()
    keep = torch.zeros((NMS_BATCH, BOXES), dtype=torch.bool)
    for batch in range(NMS_BATCH):
        suppressed = torch.zeros((BOXES,), dtype=torch.bool)
        for candidate in range(BOXES):
            if suppressed[candidate]:
                continue
            keep[batch, candidate] = True
            candidate_box = boxes_cpu[batch, candidate]
            remaining = boxes_cpu[batch, candidate + 1 :]
            if remaining.numel() == 0:
                continue
            upper_left = torch.maximum(candidate_box[:2], remaining[:, :2])
            lower_right = torch.minimum(candidate_box[2:], remaining[:, 2:])
            intersection = (lower_right - upper_left).clamp_min(0.0).prod(dim=1)
            candidate_area = (candidate_box[2:] - candidate_box[:2]).prod()
            remaining_area = (remaining[:, 2:] - remaining[:, :2]).prod(dim=1)
            overlap = intersection / (candidate_area + remaining_area - intersection)
            suppressed[candidate + 1 :] |= overlap > threshold
    return keep.cuda()


def _run_nms(compiler: str, target: Target, target_name: str) -> None:
    upper_left = torch.rand(
        (NMS_BATCH, BOXES, 2), device="cuda", dtype=torch.float32
    ) * 0.8
    size = torch.rand_like(upper_left) * 0.2 + 0.01
    boxes = torch.cat((upper_left, upper_left + size), dim=2)
    threshold = 0.5
    _run_generated(
        definition=greedy_nms,
        arguments=(boxes, threshold),
        reference=lambda: _nms_reference(boxes, threshold),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="greedy non-maximum suppression",
        tolerance=0.0,
        cuda_graph=False,
    )


def _roi_align_reference(feature: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    batch = rois[:, 0].long()
    start_x = rois[:, 1]
    start_y = rois[:, 2]
    end_x = rois[:, 3]
    end_y = rois[:, 4]
    bin_width = (end_x - start_x).clamp_min(1.0) / POOLED_WIDTH
    bin_height = (end_y - start_y).clamp_min(1.0) / POOLED_HEIGHT
    output = torch.empty(
        (ROIS, CHANNELS, POOLED_HEIGHT, POOLED_WIDTH),
        device="cuda",
        dtype=torch.float32,
    )
    for pooled_y in range(POOLED_HEIGHT):
        sample_y = start_y + (pooled_y + 0.5) * bin_height
        y_low = sample_y.to(torch.int32).clamp(0, HEIGHT - 1).long()
        y_high = (y_low + 1).clamp_max(HEIGHT - 1)
        y_fraction = sample_y - y_low.float()
        for pooled_x in range(POOLED_WIDTH):
            sample_x = start_x + (pooled_x + 0.5) * bin_width
            x_low = sample_x.to(torch.int32).clamp(0, WIDTH - 1).long()
            x_high = (x_low + 1).clamp_max(WIDTH - 1)
            x_fraction = sample_x - x_low.float()
            top = (
                feature[batch, :, y_low, x_low] * (1.0 - x_fraction[:, None])
                + feature[batch, :, y_low, x_high] * x_fraction[:, None]
            )
            bottom = (
                feature[batch, :, y_high, x_low] * (1.0 - x_fraction[:, None])
                + feature[batch, :, y_high, x_high] * x_fraction[:, None]
            )
            output[:, :, pooled_y, pooled_x] = (
                top * (1.0 - y_fraction[:, None])
                + bottom * y_fraction[:, None]
            )
    return output


def _run_roi_align(compiler: str, target: Target, target_name: str) -> None:
    feature = torch.randn(
        (ROI_BATCH, CHANNELS, HEIGHT, WIDTH),
        device="cuda",
        dtype=torch.float32,
    )
    batch = torch.randint(0, ROI_BATCH, (ROIS, 1), device="cuda").float()
    upper_left = torch.rand((ROIS, 2), device="cuda") * (HEIGHT - 16)
    size = torch.rand((ROIS, 2), device="cuda") * 48 + 8
    lower_right = torch.minimum(
        upper_left + size,
        torch.tensor([WIDTH - 1, HEIGHT - 1], device="cuda"),
    )
    rois = torch.cat((batch, upper_left, lower_right), dim=1)
    _run_generated(
        definition=roi_align_center_sample,
        arguments=(feature, rois),
        reference=lambda: _roi_align_reference(feature, rois),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="ROI Align center sampling",
        tolerance=1.0e-4,
        cuda_graph=False,
    )


def _counter_uniform(seed: int, counters: torch.Tensor) -> torch.Tensor:
    bit_mask = (1 << 32) - 1
    random_bits = (counters ^ seed ^ 1831565813) & bit_mask
    random_bits = (random_bits ^ (random_bits << 13)) & bit_mask
    random_bits = (random_bits ^ (random_bits >> 17)) & bit_mask
    random_bits = (random_bits ^ (random_bits << 5)) & bit_mask
    return (random_bits >> 8).float() * 5.960464477539063e-08


def _run_monte_carlo(compiler: str, target: Target, target_name: str) -> None:
    seed = 17
    initial_price = 100.0
    strike = 95.0
    barrier = 150.0
    drift = 0.0002
    volatility = 0.01

    def reference() -> torch.Tensor:
        path = torch.arange(PATHS, device="cuda", dtype=torch.int64)
        price = torch.full((PATHS,), initial_price, device="cuda")
        knocked_out = torch.zeros((PATHS,), device="cuda", dtype=torch.bool)
        for step in range(STEPS):
            counters = path * STEPS + step
            uniform = _counter_uniform(seed, counters)
            direction = torch.where(uniform >= 0.5, volatility, -volatility)
            next_price = price * torch.exp(torch.tensor(drift, device="cuda") + direction)
            price = torch.where(knocked_out, price, next_price)
            knocked_out |= price >= barrier
        return torch.where(
            knocked_out,
            torch.zeros_like(price),
            torch.clamp_min(price - strike, 0.0),
        )

    _run_generated(
        definition=barrier_option_paths,
        arguments=(seed, initial_price, strike, barrier, drift, volatility),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="barrier-option Monte Carlo paths",
        tolerance=3.0e-4,
        cuda_graph=False,
    )


def _run_causal_conv_update(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (CAUSAL_UPDATE_BATCH, CAUSAL_UPDATE_CHANNELS),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    state = torch.randn(
        (CAUSAL_UPDATE_BATCH, CAUSAL_UPDATE_CHANNELS, CAUSAL_CONV_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    weight = torch.randn(
        (CAUSAL_UPDATE_CHANNELS, CAUSAL_CONV_WIDTH),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    bias = torch.randn(
        (CAUSAL_UPDATE_CHANNELS,), device="cuda", dtype=torch.float16
    ) * 0.1
    initial_state = state.clone()
    expected_state = torch.cat((initial_state[:, :, 1:], x[:, :, None]), dim=2)
    expected_output = (
        (expected_state.float() * weight.float()[None, :, :]).sum(dim=2)
        + bias.float()[None, :]
    )
    expected_output = (expected_output * torch.sigmoid(expected_output)).half()
    artifact = intent.compile(
        causal_depthwise_conv1d_update,
        target=target,
        compiler=compiler,
        constexprs={"SILU": True},
    )
    output = artifact.run(x, state, weight, bias)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(state, output),
        expected=(expected_state, expected_output),
        tolerance=(0.0, 2.0e-3),
        target_name=target_name,
        kernel_name="state-cache causal convolution update",
    )
    launch = prepare_kernel_call(artifact, (x, state, weight, bias), output)
    _report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=errors,
        target_name=target_name,
        kernel_name="state-cache causal convolution update",
        prepare=lambda: state.copy_(initial_state),
        performance_scope="kernel-only",
    )


def _run_qk_rope(
    compiler: str,
    target: Target,
    target_name: str,
    rotary_dimension: int,
    inverse: bool,
    upstream: Callable[[tuple[object, ...]], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
) -> None:
    query = torch.randn(
        (ROPE_BATCH, ROPE_QUERY_HEADS, ROPE_SEQUENCE, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    key = torch.randn(
        (ROPE_BATCH, ROPE_KEY_HEADS, ROPE_SEQUENCE, HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    ) * 0.1
    half = rotary_dimension // 2
    angles = torch.randn(
        (1, ROPE_SEQUENCE, half), device="cuda", dtype=torch.float32
    )
    cosine_half = torch.cos(angles).half()
    sine_half = torch.sin(angles).half()
    if inverse:
        sine_half = -sine_half
    cosine = torch.cat((cosine_half, cosine_half), dim=2)
    sine = torch.cat((sine_half, sine_half), dim=2)
    initial_query = query.clone()
    initial_key = key.clone()

    def rotate(values: torch.Tensor) -> torch.Tensor:
        result = values.clone()
        first = values[:, :, :, :half].float()
        second = values[:, :, :, half:rotary_dimension].float()
        cos = cosine_half.float()[:, None, :, :]
        sin = sine_half.float()[:, None, :, :]
        result[:, :, :, :half] = (first * cos - second * sin).half()
        result[:, :, :, half:rotary_dimension] = (
            second * cos + first * sin
        ).half()
        return result

    expected_query = rotate(initial_query)
    expected_key = rotate(initial_key)
    definition = (
        rotary_qk_inplace
        if rotary_dimension == HEAD_DIMENSION
        else rotary_qk_partial_inplace
    )
    artifact = intent.compile(definition, target=target, compiler=compiler)
    artifact.run(query, key, cosine, sine)
    torch.cuda.synchronize()
    label = (
        "inverse Q/K rotary embedding"
        if inverse
        else (
            "full Q/K rotary embedding"
            if rotary_dimension == HEAD_DIMENSION
            else "partial Q/K rotary embedding"
        )
    )
    errors = _require_close(
        actual=(query, key),
        expected=(expected_query, expected_key),
        tolerance=(2.0e-3, 2.0e-3),
        target_name=target_name,
        kernel_name=label,
    )
    launch = prepare_kernel_call(
        artifact,
        (query, key, cosine, sine),
        (),
    )

    def restore_inputs() -> None:
        query.copy_(initial_query)
        key.copy_(initial_key)

    generated_p50, generated_p95 = benchmark(
        launch,
        warmup=3,
        repetitions=100,
        prepare=restore_inputs,
    )
    upstream_errors = None
    upstream_p50 = None
    upstream_p95 = None
    if upstream is not None:
        upstream_query = initial_query.clone()
        upstream_key = initial_key.clone()
        upstream_arguments = (upstream_query, upstream_key, cosine, sine)
        upstream_output = upstream(upstream_arguments)
        torch.cuda.synchronize()
        upstream_errors = _require_close(
            actual=upstream_output,
            expected=(expected_query, expected_key),
            tolerance=(2.0e-3, 2.0e-3),
            target_name=target_name,
            kernel_name=f"{label} upstream",
        )

        def restore_upstream_inputs() -> None:
            upstream_query.copy_(initial_query)
            upstream_key.copy_(initial_key)

        upstream_p50, upstream_p95 = benchmark(
            lambda: upstream(upstream_arguments),
            warmup=3,
            repetitions=100,
            prepare=restore_upstream_inputs,
        )

    print_artifact(artifact, target_name)
    print(
        f"{target_name} {label} numerical comparison: PASS "
        f"(generated/reference={errors})"
    )
    print(
        f"{target_name} {label} kernel-only performance (CUDA Event): "
        f"p50={generated_p50:.4f} ms, p95={generated_p95:.4f} ms"
    )
    if upstream_p50 is None:
        print(f"{target_name} {label} upstream baseline: unavailable")
    else:
        print(
            f"{target_name} {label} upstream comparison: PASS "
            f"(upstream/reference={upstream_errors}, "
            f"upstream_p50={upstream_p50:.4f} ms, "
            f"upstream_p95={upstream_p95:.4f} ms, "
            f"generated/upstream_p50={generated_p50 / upstream_p50:.4f}x)"
        )


def _run_qk_rope_full(compiler: str, target: Target, target_name: str) -> None:
    _run_qk_rope(compiler, target, target_name, HEAD_DIMENSION, False)


def _run_qk_rope_partial(compiler: str, target: Target, target_name: str) -> None:
    _run_qk_rope(compiler, target, target_name, HEAD_DIMENSION // 2, False)


def _run_qk_rope_inverse(compiler: str, target: Target, target_name: str) -> None:
    _run_qk_rope(compiler, target, target_name, HEAD_DIMENSION, True)


def _run_qk_rope_full_upstream(
    compiler: str, target: Target, target_name: str, upstream
) -> None:
    _run_qk_rope(
        compiler, target, target_name, HEAD_DIMENSION, False, upstream=upstream
    )


def _run_qk_rope_partial_upstream(
    compiler: str, target: Target, target_name: str, upstream
) -> None:
    _run_qk_rope(
        compiler, target, target_name, HEAD_DIMENSION // 2, False, upstream=upstream
    )


def _run_qk_rope_inverse_upstream(
    compiler: str, target: Target, target_name: str, upstream
) -> None:
    _run_qk_rope(
        compiler, target, target_name, HEAD_DIMENSION, True, upstream=upstream
    )


def _run_variant_gemm(compiler: str, target: Target, target_name: str) -> None:
    a = torch.randn((GEMM_M, GEMM_K), device="cuda", dtype=torch.float16) * 0.1
    b = torch.randn((GEMM_K, GEMM_N), device="cuda", dtype=torch.float16) * 0.1
    _run_variant(
        original=gemm,
        variant=gemm_loop_interchange,
        arguments=(a, b),
        reference=lambda: a @ b,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="GEMM loop interchange",
        tolerance=3.0e-2,
        constexprs={"ACTIVATION": Activation.NONE},
    )


def _run_variant_softmax(compiler: str, target: Target, target_name: str) -> None:
    x = torch.randn(
        (SOFTMAX_ROWS, SOFTMAX_COLUMNS), device="cuda", dtype=torch.float32
    )
    _run_variant(
        original=stable_softmax,
        variant=stable_softmax_online,
        arguments=(x,),
        reference=lambda: torch.softmax(x, dim=1),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="stable softmax reduce versus online stream",
        tolerance=1.0e-5,
    )


def _run_variant_online_softmax(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (ONLINE_ROWS, ONLINE_COLUMNS), device="cuda", dtype=torch.float32
    )
    _run_variant(
        original=streamed_online_softmax,
        variant=streamed_online_softmax_inline,
        arguments=(x,),
        reference=lambda: torch.softmax(x, dim=1),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="online softmax named versus inline SSA",
        tolerance=1.0e-5,
    )


def _attention_arguments() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (2, 8, 1024, HEAD_DIMENSION)
    q = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    k = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    v = torch.randn(shape, device="cuda", dtype=torch.float16) * 0.5
    return q, k, v


def _run_attention_variant(
    compiler: str,
    target: Target,
    target_name: str,
    variant,
    kernel_name: str,
) -> None:
    q, k, v = _attention_arguments()
    _run_variant(
        original=flash_attention_fwd,
        variant=variant,
        arguments=(q, k, v, ATTENTION_SCALE),
        reference=lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=ATTENTION_SCALE
        ),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name=kernel_name,
        tolerance=2.0e-2,
        constexprs={"CAUSAL": True},
    )


def _run_variant_attention_inline(
    compiler: str, target: Target, target_name: str
) -> None:
    _run_attention_variant(
        compiler,
        target,
        target_name,
        flash_attention_inline_fwd,
        "attention helper versus inline body",
    )


def _run_variant_attention_select(
    compiler: str, target: Target, target_name: str
) -> None:
    _run_attention_variant(
        compiler,
        target,
        target_name,
        flash_attention_select_fwd,
        "attention mask versus value select",
    )


def _run_variant_rope(compiler: str, target: Target, target_name: str) -> None:
    heads = 32
    tokens = 2048
    rows = heads * tokens
    values = torch.randn(
        (rows, HEAD_DIMENSION), device="cuda", dtype=torch.float16
    )
    cosine = torch.randn(
        (tokens, HALF_DIMENSION), device="cuda", dtype=torch.float16
    )
    sine = torch.randn_like(cosine)

    def reference() -> torch.Tensor:
        dimensions = torch.arange(HEAD_DIMENSION, device="cuda")
        paired = (dimensions - HALF_DIMENSION) % HEAD_DIMENSION
        phase = dimensions % HALF_DIMENSION
        sign = torch.where(dimensions < HALF_DIMENSION, -1.0, 1.0).half()
        token = torch.arange(rows, device="cuda") // heads
        rotated = (
            values * cosine[token][:, phase]
            + values[:, paired] * sine[token][:, phase] * sign
        )
        return rotated.reshape(rows, 2, HALF_DIMENSION)

    _run_variant(
        original=rotary_embedding_flat,
        variant=rotary_embedding_equivalent_index,
        arguments=(values, cosine, sine),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="RoPE equivalent paired index",
        tolerance=1.0e-2,
        constexprs={"HEADS": heads},
    )


def _run_variant_swiglu(compiler: str, target: Target, target_name: str) -> None:
    shape = (SWIGLU_TOKENS, SWIGLU_FEATURES)
    gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.5
    up = torch.randn_like(gate) * 0.5
    _run_variant(
        original=swiglu_forward,
        variant=swiglu_forward_helper,
        arguments=(gate, up),
        reference=lambda: (F.silu(gate.float()) * up.float()).to(torch.bfloat16),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="SwiGLU inline versus helper",
        tolerance=5.0e-2,
    )


def _run_variant_layer_norm(
    compiler: str, target: Target, target_name: str
) -> None:
    epsilon = 1.0e-5
    x = torch.randn(
        (LAYER_ROWS, LAYER_FEATURES), device="cuda", dtype=torch.float32
    )
    weight = torch.randn((LAYER_FEATURES,), device="cuda", dtype=torch.float32)
    bias = torch.randn_like(weight)
    arguments = (x, weight, bias, 1.0 / LAYER_FEATURES, epsilon)
    _run_variant(
        original=weighted_layer_norm,
        variant=weighted_layer_norm_second_moment,
        arguments=arguments,
        reference=lambda: F.layer_norm(
            x, (LAYER_FEATURES,), weight=weight, bias=bias, eps=epsilon
        ),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="LayerNorm equivalent second-moment expression",
        tolerance=5.0e-5,
    )


def _run_variant_conv2d(compiler: str, target: Target, target_name: str) -> None:
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
    _run_variant(
        original=conv2d_same,
        variant=conv2d_reduce_order,
        arguments=(x, weight),
        reference=lambda: F.conv2d(
            x[:, None],
            weight[None, None],
            padding=(CONV2D_FILTER_HEIGHT // 2, CONV2D_FILTER_WIDTH // 2),
        )[:, 0],
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="Conv2D equivalent reduction order",
        tolerance=3.0e-3,
    )


def _run_variant_transpose(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (TRANSPOSE_ROWS, TRANSPOSE_COLUMNS),
        device="cuda",
        dtype=torch.float16,
    )
    _run_variant(
        original=matrix_transpose,
        variant=matrix_transpose_scalar_domains,
        arguments=(x,),
        reference=lambda: x.transpose(0, 1).contiguous(),
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="transpose partition versus explicit domains",
        tolerance=0.0,
    )


def _run_nonzero_compact(
    compiler: str, target: Target, target_name: str
) -> None:
    values = torch.randn(
        (NONZERO_ROWS, NONZERO_VALUES), device="cuda", dtype=torch.float32
    )
    values[torch.rand_like(values) < 0.7] = 0.0

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        flags = values != 0.0
        prefix = flags.to(torch.int32).cumsum(dim=1, dtype=torch.int32)
        output = torch.full_like(prefix, -1)
        rows, columns = torch.nonzero(flags, as_tuple=True)
        output[rows, prefix[rows, columns].long() - 1] = columns.to(torch.int32)
        return output, flags.sum(dim=1, dtype=torch.int32)

    _run_generated(
        definition=compact_nonzero_rows,
        arguments=(values,),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="row-wise nonzero compact-select",
        tolerance=(0.0, 0.0),
        cuda_graph=False,
    )


def _run_unique_consecutive(
    compiler: str, target: Target, target_name: str
) -> None:
    increments = torch.randint(
        0,
        5,
        (UNIQUE_ROWS, UNIQUE_VALUES),
        device="cuda",
        dtype=torch.int32,
    )
    values = torch.cumsum(increments == 0, dim=1, dtype=torch.int32)
    run_lengths = torch.zeros_like(values)
    artifact = intent.compile(
        unique_consecutive_rows,
        target=target,
        compiler=compiler,
    )
    generated = artifact.run(values, run_lengths)
    generated_call = prepare_kernel_call(
        artifact,
        (values, run_lengths),
        generated,
    )
    run_lengths.zero_()
    generated_call()
    run_starts = torch.ones_like(values, dtype=torch.bool)
    run_starts[:, 1:] = values[:, 1:] != values[:, :-1]
    groups = run_starts.to(torch.int32).cumsum(dim=1, dtype=torch.int32) - 1
    expected_values = torch.zeros_like(values)
    rows, columns = torch.nonzero(run_starts, as_tuple=True)
    expected_values[rows, groups[rows, columns].long()] = values[rows, columns]
    expected_lengths = torch.zeros_like(values)
    expected_lengths.scatter_add_(1, groups.long(), torch.ones_like(values))
    expected_counts = run_starts.sum(dim=1, dtype=torch.int32)
    expected = (expected_values, groups, expected_counts, expected_lengths)
    actual = (*_outputs(generated), run_lengths)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=actual,
        expected=expected,
        tolerance=(0.0, 0.0, 0.0, 0.0),
        target_name=target_name,
        kernel_name="unique-consecutive run-length encoding",
    )
    def clear_and_launch() -> None:
        run_lengths.zero_()
        generated_call()

    _report_pipeline(
        artifacts=(artifact,),
        launch=clear_and_launch,
        errors=errors,
        target_name=target_name,
        kernel_name="unique-consecutive run-length encoding",
    )


def _run_moe_align(compiler: str, target: Target, target_name: str) -> None:
    topk_ids = torch.randint(
        0,
        MOE_ALIGN_EXPERTS,
        (MOE_ALIGN_TOKENS, MOE_ALIGN_TOP_K),
        device="cuda",
        dtype=torch.int32,
    )
    expert_counts = torch.zeros(
        (MOE_ALIGN_EXPERTS,), device="cuda", dtype=torch.int32
    )
    expert_cursors = torch.zeros_like(expert_counts)
    sorted_routes = torch.full(
        (MOE_ALIGN_PADDED_ROUTES,), -1, device="cuda", dtype=torch.int32
    )
    count_artifact = intent.compile(
        moe_count_routes, target=target, compiler=compiler
    )
    prefix_artifact = intent.compile(
        moe_prefix_routes, target=target, compiler=compiler
    )
    scatter_artifact = intent.compile(
        moe_scatter_routes, target=target, compiler=compiler
    )
    blocks_artifact = intent.compile(
        moe_mark_expert_blocks, target=target, compiler=compiler
    )
    count_artifact.run(topk_ids, expert_counts)
    prefix_outputs = prefix_artifact.run(expert_counts)
    expert_offsets, total_padded = _outputs(prefix_outputs)
    scatter_artifact.run(topk_ids, expert_offsets, expert_cursors, sorted_routes)
    expert_blocks = blocks_artifact.run(expert_offsets)

    expected_counts = torch.bincount(
        topk_ids.flatten().long(), minlength=MOE_ALIGN_EXPERTS
    ).to(torch.int32)
    expected_padded = (
        (expected_counts + MOE_ALIGN_BLOCK_SIZE - 1) // MOE_ALIGN_BLOCK_SIZE
    ) * MOE_ALIGN_BLOCK_SIZE
    expected_offsets = torch.zeros_like(expert_offsets)
    expected_offsets[1:] = expected_padded.cumsum(dim=0)
    expected_total = expected_offsets[-1:].clone()
    expected_routes = torch.full_like(sorted_routes, -1)
    actual_routes = sorted_routes.clone()
    valid_block_count = int(expected_total.item()) // MOE_ALIGN_BLOCK_SIZE
    expected_blocks = torch.zeros(
        (valid_block_count,), device="cuda", dtype=torch.int32
    )
    flat_ids = topk_ids.flatten()
    for expert in range(MOE_ALIGN_EXPERTS):
        routes = torch.nonzero(flat_ids == expert, as_tuple=False).flatten().to(torch.int32)
        start = int(expected_offsets[expert].item())
        expected_routes[start : start + routes.numel()] = routes
        actual_routes[start : start + routes.numel()] = torch.sort(
            actual_routes[start : start + routes.numel()]
        ).values
        first_block = start // MOE_ALIGN_BLOCK_SIZE
        last_block = int(expected_offsets[expert + 1].item()) // MOE_ALIGN_BLOCK_SIZE
        expected_blocks[first_block:last_block] = expert
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(
            expert_counts,
            expert_offsets,
            total_padded,
            actual_routes,
            expert_blocks[:valid_block_count],
        ),
        expected=(
            expected_counts,
            expected_offsets,
            expected_total,
            expected_routes,
            expected_blocks,
        ),
        tolerance=(0.0, 0.0, 0.0, 0.0, 0.0),
        target_name=target_name,
        kernel_name="MoE block alignment pipeline",
    )
    count_call = prepare_kernel_call(
        count_artifact, (topk_ids, expert_counts), ()
    )
    prefix_call = prepare_kernel_call(
        prefix_artifact, (expert_counts,), prefix_outputs
    )
    scatter_call = prepare_kernel_call(
        scatter_artifact,
        (topk_ids, expert_offsets, expert_cursors, sorted_routes),
        (),
    )
    blocks_call = prepare_kernel_call(
        blocks_artifact, (expert_offsets,), expert_blocks
    )

    def launch_pipeline() -> None:
        expert_counts.zero_()
        expert_cursors.zero_()
        sorted_routes.fill_(-1)
        count_call()
        prefix_call()
        scatter_call()
        blocks_call()

    _report_pipeline(
        artifacts=(count_artifact, prefix_artifact, scatter_artifact, blocks_artifact),
        launch=launch_pipeline,
        errors=errors,
        target_name=target_name,
        kernel_name="MoE block alignment pipeline",
    )


def _run_nested_ragged_pool(
    compiler: str, target: Target, target_name: str
) -> None:
    document_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(NESTED_DOCUMENTS)],
        device="cuda",
        dtype=torch.int32,
    )
    sentence_lengths = torch.tensor(
        [8 if index % 2 == 0 else 24 for index in range(NESTED_SENTENCES)],
        device="cuda",
        dtype=torch.int32,
    )
    document_offsets = torch.zeros(
        (NESTED_DOCUMENTS + 1,), device="cuda", dtype=torch.int32
    )
    sentence_offsets = torch.zeros(
        (NESTED_SENTENCES + 1,), device="cuda", dtype=torch.int32
    )
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn(
        (NESTED_TOKENS, NESTED_FEATURES),
        device="cuda",
        dtype=torch.float32,
    )

    def reference() -> tuple[torch.Tensor, torch.Tensor]:
        sentence_ids = torch.repeat_interleave(
            torch.arange(NESTED_SENTENCES, device="cuda"), sentence_lengths.long()
        )
        sentence_sums = torch.zeros(
            (NESTED_SENTENCES, NESTED_FEATURES), device="cuda"
        )
        sentence_sums.index_add_(0, sentence_ids, values)
        sentence_means = sentence_sums / sentence_lengths[:, None]
        document_ids = torch.repeat_interleave(
            torch.arange(NESTED_DOCUMENTS, device="cuda"), document_lengths.long()
        )
        document_sums = torch.zeros(
            (NESTED_DOCUMENTS, NESTED_FEATURES), device="cuda"
        )
        document_sums.index_add_(0, document_ids, sentence_sums)
        document_tokens = torch.zeros(
            (NESTED_DOCUMENTS,), device="cuda", dtype=torch.int32
        )
        document_tokens.index_add_(0, document_ids, sentence_lengths)
        return sentence_means, document_sums / document_tokens[:, None]

    _run_generated(
        definition=nested_jagged_mean_pool,
        arguments=(document_offsets, sentence_offsets, values),
        reference=reference,
        compiler=compiler,
        target=target,
        target_name=target_name,
        kernel_name="two-level nested jagged mean pooling",
        tolerance=(2.0e-5, 2.0e-5),
        cuda_graph=False,
    )


def _run_adamw(compiler: str, target: Target, target_name: str) -> None:
    gradient = torch.randn(
        (ADAMW_PARAMETERS,), device="cuda", dtype=torch.float32
    ) * 0.01
    parameter = torch.randn_like(gradient)
    first = torch.randn_like(gradient) * 0.01
    second = torch.rand_like(gradient) * 0.01
    expected_parameter = parameter.clone()
    expected_first = first.clone()
    expected_second = second.clone()
    initial_parameter = parameter.clone()
    initial_first = first.clone()
    initial_second = second.clone()
    learning_rate, beta1, beta2 = 1.0e-3, 0.9, 0.999
    inverse_bias1, inverse_bias2 = 10.0, 1000.0
    epsilon, weight_decay = 1.0e-8, 0.01
    expected_first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    expected_second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
    expected_parameter.mul_(1.0 - learning_rate * weight_decay).add_(
        expected_first
        * inverse_bias1
        * torch.rsqrt(expected_second * inverse_bias2 + epsilon),
        alpha=-learning_rate,
    )
    artifact = intent.compile(adamw_update, target=target, compiler=compiler)
    arguments = (
        gradient,
        parameter,
        first,
        second,
        learning_rate,
        beta1,
        beta2,
        inverse_bias1,
        inverse_bias2,
        epsilon,
        weight_decay,
    )
    artifact.run(*arguments)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(parameter, first, second),
        expected=(expected_parameter, expected_first, expected_second),
        tolerance=(2.0e-6, 2.0e-6, 2.0e-6),
        target_name=target_name,
        kernel_name="fused AdamW update",
    )
    launch = prepare_kernel_call(artifact, arguments, ())

    def restore_state() -> None:
        parameter.copy_(initial_parameter)
        first.copy_(initial_first)
        second.copy_(initial_second)

    _report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=errors,
        target_name=target_name,
        kernel_name="fused AdamW update",
        prepare=restore_state,
        performance_scope="kernel-only",
    )


def _run_adafactor(compiler: str, target: Target, target_name: str) -> None:
    gradient = torch.randn(
        (ADAFACTOR_ROWS, ADAFACTOR_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    ) * 0.01
    parameter = torch.randn_like(gradient)
    row_state = torch.rand((ADAFACTOR_ROWS,), device="cuda") * 0.01
    column_state = torch.rand((ADAFACTOR_COLUMNS,), device="cuda") * 0.01
    row_mean = torch.zeros((1,), device="cuda")
    expected_parameter = parameter.clone()
    expected_row = row_state.clone()
    expected_column = column_state.clone()
    initial_parameter = parameter.clone()
    initial_row = row_state.clone()
    initial_column = column_state.clone()
    decay, learning_rate, epsilon = 0.8, 1.0e-2, 1.0e-8
    expected_row.mul_(decay).add_(
        gradient.square().mean(dim=1), alpha=1.0 - decay
    )
    expected_column.mul_(decay).add_(
        gradient.square().mean(dim=0), alpha=1.0 - decay
    )
    expected_mean = expected_row.mean().reshape(1)
    variance = expected_row[:, None] * expected_column[None, :] / expected_mean
    expected_parameter.add_(
        gradient * torch.rsqrt(variance + epsilon), alpha=-learning_rate
    )
    rows_artifact = intent.compile(
        adafactor_update_rows, target=target, compiler=compiler
    )
    columns_artifact = intent.compile(
        adafactor_update_columns, target=target, compiler=compiler
    )
    apply_artifact = intent.compile(adafactor_apply, target=target, compiler=compiler)
    rows_arguments = (
        gradient,
        row_state,
        row_mean,
        decay,
        1.0 / ADAFACTOR_COLUMNS,
        1.0 / ADAFACTOR_ROWS,
    )
    columns_arguments = (
        gradient,
        column_state,
        decay,
        1.0 / ADAFACTOR_ROWS,
    )
    apply_arguments = (
        gradient,
        row_state,
        column_state,
        row_mean,
        parameter,
        learning_rate,
        epsilon,
    )
    rows_artifact.run(*rows_arguments)
    columns_artifact.run(*columns_arguments)
    apply_artifact.run(*apply_arguments)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(parameter, row_state, column_state, row_mean),
        expected=(expected_parameter, expected_row, expected_column, expected_mean),
        tolerance=(2.0e-5, 2.0e-7, 2.0e-7, 2.0e-7),
        target_name=target_name,
        kernel_name="factored Adafactor update pipeline",
    )
    rows_call = prepare_kernel_call(rows_artifact, rows_arguments, ())
    columns_call = prepare_kernel_call(columns_artifact, columns_arguments, ())
    apply_call = prepare_kernel_call(apply_artifact, apply_arguments, ())

    def launch_pipeline() -> None:
        rows_call()
        columns_call()
        apply_call()

    def restore_state() -> None:
        parameter.copy_(initial_parameter)
        row_state.copy_(initial_row)
        column_state.copy_(initial_column)
        row_mean.zero_()

    _report_pipeline(
        artifacts=(rows_artifact, columns_artifact, apply_artifact),
        launch=launch_pipeline,
        errors=errors,
        target_name=target_name,
        kernel_name="factored Adafactor update pipeline",
        prepare=restore_state,
    )


def _run_reshape_cache(compiler: str, target: Target, target_name: str) -> None:
    key = torch.randn(
        (CACHE_TOKENS, CACHE_HEADS, CACHE_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    slots = torch.randperm(
        CACHE_BLOCKS * CACHE_BLOCK_SIZE, device="cuda", dtype=torch.int64
    )[:CACHE_TOKENS].to(torch.int32)
    key_cache = torch.zeros(
        (CACHE_BLOCKS, CACHE_BLOCK_SIZE, CACHE_HEADS, CACHE_HEAD_DIMENSION),
        device="cuda",
        dtype=torch.float16,
    )
    value_cache = torch.zeros_like(key_cache)
    expected_key = key_cache.clone()
    expected_value = value_cache.clone()
    blocks = slots.long() // CACHE_BLOCK_SIZE
    offsets = slots.long() % CACHE_BLOCK_SIZE
    expected_key[blocks, offsets] = key
    expected_value[blocks, offsets] = value
    artifact = intent.compile(reshape_and_cache, target=target, compiler=compiler)
    arguments = (key, value, slots, key_cache, value_cache)
    artifact.run(*arguments)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(key_cache, value_cache),
        expected=(expected_key, expected_value),
        tolerance=(0.0, 0.0),
        target_name=target_name,
        kernel_name="reshape-and-cache in-place update",
    )
    launch = prepare_kernel_call(artifact, arguments, ())
    _report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=errors,
        target_name=target_name,
        kernel_name="reshape-and-cache in-place update",
        cuda_graph=True,
        performance_scope="kernel-only",
    )


def _run_group_norm_silu_backward(
    compiler: str, target: Target, target_name: str
) -> None:
    x = torch.randn(
        (GROUP_NORM_BATCH, GROUP_NORM_CHANNELS, GROUP_NORM_SPATIAL),
        device="cuda",
        dtype=torch.bfloat16,
    )
    upstream = torch.randn_like(x)
    weight = torch.randn((GROUP_NORM_CHANNELS,), device="cuda")
    bias = torch.randn_like(weight)
    grouped = x.float().reshape(
        GROUP_NORM_BATCH,
        GROUP_NORM_GROUPS,
        CHANNELS_PER_GROUP,
        GROUP_NORM_SPATIAL,
    )
    mean = grouped.mean(dim=(2, 3))
    variance = grouped.var(dim=(2, 3), unbiased=False)
    rstd = torch.rsqrt(variance + 1.0e-5)
    x_ref = x.float().detach().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    bias_ref = bias.detach().clone().requires_grad_(True)
    output = F.silu(
        F.group_norm(
            x_ref,
            GROUP_NORM_GROUPS,
            weight_ref,
            bias_ref,
            eps=1.0e-5,
        )
    )
    output.backward(upstream.float())
    expected_dx = x_ref.grad.to(torch.bfloat16)
    expected_dw = weight_ref.grad
    expected_db = bias_ref.grad
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros_like(bias)
    artifact = intent.compile(
        group_norm_silu_backward, target=target, compiler=compiler
    )
    arguments = (
        x,
        upstream,
        weight,
        bias,
        mean,
        rstd,
        dweight,
        dbias,
        1.0 / (CHANNELS_PER_GROUP * GROUP_NORM_SPATIAL),
    )
    dx = artifact.run(*arguments)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(dx, dweight, dbias),
        expected=(expected_dx, expected_dw, expected_db),
        tolerance=(4.0e-2, 3.0e-2, 3.0e-2),
        target_name=target_name,
        kernel_name="GroupNorm plus SiLU backward",
    )
    launch = prepare_kernel_call(artifact, arguments, dx)

    def clear_and_launch() -> None:
        dweight.zero_()
        dbias.zero_()
        launch()

    _report_pipeline(
        artifacts=(artifact,),
        launch=clear_and_launch,
        errors=errors,
        target_name=target_name,
        kernel_name="GroupNorm plus SiLU backward",
    )


def _run_cholesky(compiler: str, target: Target, target_name: str) -> None:
    seed = torch.randn(
        (CHOLESKY_BATCH, CHOLESKY_SIZE, CHOLESKY_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    matrices = seed @ seed.transpose(1, 2)
    matrices.add_(
        torch.eye(CHOLESKY_SIZE, device="cuda")[None], alpha=CHOLESKY_SIZE
    )
    expected = torch.linalg.cholesky(matrices)
    original_matrices = matrices.clone()
    artifact = intent.compile(
        batched_cholesky_lower, target=target, compiler=compiler
    )
    artifact.run(matrices)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=matrices,
        expected=expected,
        tolerance=2.0e-4,
        target_name=target_name,
        kernel_name="batched in-place Cholesky factorization",
    )
    launch = prepare_kernel_call(artifact, (matrices,), ())
    _report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=errors,
        target_name=target_name,
        kernel_name="batched in-place Cholesky factorization",
        cuda_graph=False,
        prepare=lambda: matrices.copy_(original_matrices),
        performance_scope="kernel-only",
    )


def _run_householder_qr(compiler: str, target: Target, target_name: str) -> None:
    matrices = torch.randn(
        (QR_BATCH, QR_ROWS, QR_COLUMNS),
        device="cuda",
        dtype=torch.float32,
    )
    expected_matrix, expected_tau = torch.geqrf(matrices)
    original_matrices = matrices.clone()
    artifact = intent.compile(
        batched_householder_qr, target=target, compiler=compiler
    )
    tau = artifact.run(matrices)
    torch.cuda.synchronize()
    errors = _require_close(
        actual=(matrices, tau),
        expected=(expected_matrix, expected_tau),
        tolerance=(3.0e-4, 3.0e-4),
        target_name=target_name,
        kernel_name="batched in-place Householder QR",
    )
    launch = prepare_kernel_call(artifact, (matrices,), tau)
    _report_pipeline(
        artifacts=(artifact,),
        launch=launch,
        errors=errors,
        target_name=target_name,
        kernel_name="batched in-place Householder QR",
        cuda_graph=False,
        prepare=lambda: matrices.copy_(original_matrices),
        performance_scope="kernel-only",
    )


UNFAMILIAR_RUNNERS: dict[str, Runner] = {
    "histogram": _run_histogram,
    "csr_spmv": _run_csr_spmv,
    "radix2_fft": _run_fft,
    "bitonic_sort": _run_bitonic_sort,
    "kmeans_assign": _run_kmeans,
    "viterbi_decode": _run_viterbi,
    "smith_waterman": _run_smith_waterman,
    "greedy_nms": _run_nms,
    "roi_align": _run_roi_align,
    "barrier_option": _run_monte_carlo,
    "causal_conv1d_update": _run_causal_conv_update,
    "rope_qk_full": _run_qk_rope_full,
    "rope_qk_partial": _run_qk_rope_partial,
    "rope_qk_inverse": _run_qk_rope_inverse,
    "nonzero_compact": _run_nonzero_compact,
    "unique_consecutive": _run_unique_consecutive,
    "moe_align_block": _run_moe_align,
    "nested_ragged_pool": _run_nested_ragged_pool,
    "adamw_update": _run_adamw,
    "adafactor_update": _run_adafactor,
    "reshape_and_cache": _run_reshape_cache,
    "group_norm_silu_backward": _run_group_norm_silu_backward,
    "batched_cholesky": _run_cholesky,
    "batched_householder_qr": _run_householder_qr,
    **SMALL_OPERATOR_RUNNERS,
    "variant_gemm_loop_interchange": _run_variant_gemm,
    "variant_softmax_online": _run_variant_softmax,
    "variant_online_softmax_inline": _run_variant_online_softmax,
    "variant_attention_inline": _run_variant_attention_inline,
    "variant_attention_select": _run_variant_attention_select,
    "variant_rope_index": _run_variant_rope,
    "variant_swiglu_helper": _run_variant_swiglu,
    "variant_layer_norm_second_moment": _run_variant_layer_norm,
    "variant_conv2d_reduce_order": _run_variant_conv2d,
    "variant_transpose_scalar_domains": _run_variant_transpose,
    **DECOMPOSITION_RUNNERS,
}

UNFAMILIAR_UPSTREAM_RUNNERS = {
    "rope_qk_full": _run_qk_rope_full_upstream,
    "rope_qk_partial": _run_qk_rope_partial_upstream,
    "rope_qk_inverse": _run_qk_rope_inverse_upstream,
}


def run_unfamiliar(
    kernel_name: str,
    compiler: str,
    target: Target,
    target_name: str,
    upstream=None,
) -> None:
    if upstream is not None:
        runner = UNFAMILIAR_UPSTREAM_RUNNERS.get(kernel_name)
        if runner is None:
            raise ValueError(
                f"unfamiliar program '{kernel_name}' has no upstream adapter"
            )
        runner(compiler, target, target_name, upstream)
        return
    UNFAMILIAR_RUNNERS[kernel_name](compiler, target, target_name)
