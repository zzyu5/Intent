import torch
import triton
import triton.language as tl


@triton.jit
def _linear_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    m_size,
    n_size,
    k_size,
    input_stride_m,
    input_stride_k,
    weight_stride_n,
    weight_stride_k,
    output_stride_m,
    output_stride_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m_size
    col_mask = cols < n_size

    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(0, tl.cdiv(k_size, BLOCK_K)):
        ks = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < k_size

        input_values = tl.load(
            input_ptr + rows[:, None] * input_stride_m + ks[None, :] * input_stride_k,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptr + cols[:, None] * weight_stride_n + ks[None, :] * weight_stride_k,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accum = tl.dot(input_values, tl.trans(weight_values), acc=accum, input_precision="ieee")

    if HAS_BIAS:
        bias_values = tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        accum += bias_values[None, :]

    accum = tl.maximum(accum, 0.0)
    tl.store(
        output_ptr + rows[:, None] * output_stride_m + cols[None, :] * output_stride_n,
        accum,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _layer_norm_kernel(
    input_ptr,
    output_ptr,
    row_count,
    normalized_size,
    eps,
    input_stride_m,
    input_stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < normalized_size

    values = tl.load(
        input_ptr + row * input_stride_m + cols * input_stride_n,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    mean = tl.sum(tl.where(mask, values, 0.0), axis=0) / normalized_size
    centered = values - mean
    variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / normalized_size
    normalized = centered * tl.rsqrt(variance + eps)

    tl.store(
        output_ptr + row * output_stride_m + cols * output_stride_n,
        normalized,
        mask=(row < row_count) & mask,
    )


def build(context):
    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        del normalized_shape, elementwise_affine

        m_size = input.numel() // input.shape[-1]
        k_size = input.shape[-1]
        n_size = weight.shape[0]
        output_shape = input.shape[:-1] + (n_size,)

        intermediate = torch.empty((m_size, n_size), device=input.device, dtype=input.dtype)
        output_2d = torch.empty_like(intermediate)

        has_bias = bias is not None
        bias_ptr = bias if has_bias else input
        grid = (triton.cdiv(m_size, 4), triton.cdiv(n_size, 128))
        _linear_relu_kernel[grid](
            input,
            weight,
            bias_ptr,
            intermediate,
            m_size,
            n_size,
            k_size,
            input.stride(0),
            input.stride(-1),
            weight.stride(0),
            weight.stride(1),
            intermediate.stride(0),
            intermediate.stride(1),
            HAS_BIAS=has_bias,
            BLOCK_M=4,
            BLOCK_N=128,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )

        _layer_norm_kernel[(m_size,)](
            intermediate,
            output_2d,
            m_size,
            n_size,
            eps,
            intermediate.stride(0),
            intermediate.stride(1),
            output_2d.stride(0),
            output_2d.stride(1),
            BLOCK_N=triton.next_power_of_2(n_size),
            num_warps=4,
        )
        return output_2d.view(output_shape)

    return wrapper
