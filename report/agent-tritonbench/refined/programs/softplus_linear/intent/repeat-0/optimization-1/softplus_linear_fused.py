import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    ],
    key=[
        "K",
        "N",
        "stride_input_k",
        "stride_weight_n",
        "stride_weight_k",
        "stride_bias",
        "stride_output_n",
    ],
)
@triton.jit
def _fused_matvec(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    stride_input_k,
    stride_weight_n,
    stride_weight_k,
    stride_bias,
    stride_output_n,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    accumulator = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        input_values = tl.load(
            input_ptr + k_offsets * stride_input_k,
            mask=k_mask,
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptr
            + n_offsets[:, None] * stride_weight_n
            + k_offsets[None, :] * stride_weight_k,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            input_values[None, :],
            tl.permute(weight_values, (1, 0)),
            accumulator,
            input_precision="ieee",
        )

    linear = accumulator[0, :]
    if HAS_BIAS:
        linear = linear + tl.load(bias_ptr + n_offsets * stride_bias, mask=n_mask, other=0.0)

    scaled = linear * BETA
    magnitude = tl.maximum(scaled, -scaled, propagate_nan=tl.PropagateNan.ALL)
    smooth = tl.maximum(scaled, 0.0, propagate_nan=tl.PropagateNan.ALL)
    smooth = smooth + libdevice.log1p(libdevice.exp(-magnitude))
    result = tl.where(scaled > THRESHOLD, linear, smooth / BETA)
    tl.store(output_ptr + n_offsets * stride_output_n, result, mask=n_mask)


def launch(input, weight, bias, output, beta=1, threshold=20):
    K = input.shape[1]
    N = weight.shape[0]
    bias_ptr = input if bias is None else bias
    stride_bias = input.stride(1) if bias is None else bias.stride(0)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    return _fused_matvec[grid](
        input,
        weight,
        bias_ptr,
        output,
        K,
        N,
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        stride_bias,
        output.stride(1),
        BETA=float(beta),
        THRESHOLD=float(threshold),
        HAS_BIAS=bias is not None,
    )


def run(input, weight, bias=None, beta=1, threshold=20):
    output = torch.empty(
        (input.shape[0], weight.shape[0]),
        device=input.device,
        dtype=torch.float32,
    )
    launch(input, weight, bias, output, beta, threshold)
    return output
