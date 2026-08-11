from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

import intent
from intent.targets.base import Target
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
from kernels.position.rope import rotary_embedding_flat
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


Runner = Callable[[str, Target, str], None]
TensorOutputs = torch.Tensor | tuple[torch.Tensor, ...]


def _outputs(value: TensorOutputs) -> tuple[torch.Tensor, ...]:
    return value if isinstance(value, tuple) else (value,)


def _errors(actual: TensorOutputs, expected: TensorOutputs) -> tuple[float, ...]:
    actual_values = _outputs(actual)
    expected_values = _outputs(expected)
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
        if generated.dtype == torch.bool:
            result.append(0.0 if torch.equal(generated, wanted) else 1.0)
        else:
            result.append((generated - wanted).abs().max().item())
    return tuple(result)


def _require_close(
    *,
    actual: TensorOutputs,
    expected: TensorOutputs,
    tolerance: float | tuple[float, ...],
    target_name: str,
    kernel_name: str,
) -> tuple[float, ...]:
    errors = _errors(actual, expected)
    tolerances = (
        tolerance
        if isinstance(tolerance, tuple)
        else tuple(tolerance for _ in errors)
    )
    if len(tolerances) != len(errors):
        raise RuntimeError("tolerance count does not match the generated results")
    if any(error > limit for error, limit in zip(errors, tolerances)):
        raise RuntimeError(
            f"{target_name} {kernel_name} numerical comparison failed: {errors}"
        )
    return errors


def _run_generated(
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
    errors = _require_close(
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
        f"(generated/reference={errors})"
    )
    print(
        f"{target_name} {kernel_name} kernel-only performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"p50={p50:.4f} ms, p95={p95:.4f} ms"
    )
    print(f"{target_name} {kernel_name} upstream baseline: unavailable")


def _run_variant(
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
    original_errors = _require_close(
        actual=original_output,
        expected=expected,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} original",
    )
    variant_errors = _require_close(
        actual=variant_output,
        expected=expected,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} variant",
    )
    pair_errors = _require_close(
        actual=variant_output,
        expected=original_output,
        tolerance=tolerance,
        target_name=target_name,
        kernel_name=f"{kernel_name} equivalent pair",
    )
    original_call = prepare_kernel_call(original_artifact, arguments, original_output)
    variant_call = prepare_kernel_call(variant_artifact, arguments, variant_output)
    original_p50, original_p95 = benchmark(
        original_call, warmup=3, repetitions=100, cuda_graph=cuda_graph
    )
    variant_p50, variant_p95 = benchmark(
        variant_call, warmup=3, repetitions=100, cuda_graph=cuda_graph
    )
    print_artifact(variant_artifact, target_name)
    print(
        f"{target_name} {kernel_name} equivalent formulation comparison: PASS "
        f"(original/reference={original_errors}, "
        f"variant/reference={variant_errors}, variant/original={pair_errors})"
    )
    print(
        f"{target_name} {kernel_name} kernel-only performance "
        f"({'CUDA Graph' if cuda_graph else 'CUDA Event'}): "
        f"original_p50={original_p50:.4f} ms, original_p95={original_p95:.4f} ms, "
        f"variant_p50={variant_p50:.4f} ms, variant_p95={variant_p95:.4f} ms, "
        f"variant/original_p50={variant_p50 / original_p50:.4f}x"
    )


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
        tolerance=2.0e-5,
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
}


def run_unfamiliar(
    kernel_name: str,
    compiler: str,
    target: Target,
    target_name: str,
) -> None:
    UNFAMILIAR_RUNNERS[kernel_name](compiler, target, target_name)
