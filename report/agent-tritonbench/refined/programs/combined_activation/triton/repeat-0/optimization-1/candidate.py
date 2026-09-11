import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['m', 'n', 'k'],
)
@triton.jit
def _combined_activation_kernel(
    input_ptr,
    weight1_ptr,
    weight2_ptr,
    bias_ptr,
    output_ptr,
    m,
    n,
    k,
    input_stride_m,
    input_stride_k,
    weight1_stride_k,
    weight1_stride_n,
    weight2_stride_n,
    bias_stride_n,
    output_stride_m,
    output_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < m
    col_mask = cols < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k

        input_offsets = (
            input_ptr
            + rows[:, None] * input_stride_m
            + k_offsets[None, :] * input_stride_k
        )
        weight1_offsets = (
            weight1_ptr
            + k_offsets[:, None] * weight1_stride_k
            + cols[None, :] * weight1_stride_n
        )
        lhs = tl.load(input_offsets, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        rhs = tl.load(weight1_offsets, mask=k_mask[:, None] & col_mask[None, :], other=0.0)
        acc = tl.dot(lhs, rhs, acc=acc, input_precision="ieee")

    sigmoid = 1.0 / (1.0 + tl.exp(-acc))
    exp_two_sigmoid = tl.exp(2.0 * sigmoid)
    activation = 1.0 - 2.0 / (exp_two_sigmoid + 1.0)

    weight2 = tl.load(
        weight2_ptr + cols * weight2_stride_n,
        mask=col_mask,
        other=0.0,
    )
    bias = tl.load(
        bias_ptr + cols * bias_stride_n,
        mask=col_mask,
        other=0.0,
    )
    result = activation * weight2[None, :] + bias[None, :]

    output_offsets = (
        output_ptr
        + rows[:, None] * output_stride_m
        + cols[None, :] * output_stride_n
    )
    tl.store(output_offsets, result, mask=row_mask[:, None] & col_mask[None, :])


def build(context):
    def combined_activation(input, weight1, weight2, bias, *, out=None):
        k = input.shape[-1]
        n = weight1.shape[-1]
        m = input.numel() // k

        if out is None:
            output_shape = input.shape[:-1] + (n,)
            output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        else:
            output = out

        grid = lambda meta: (
            triton.cdiv(m, meta['BLOCK_M']),
            triton.cdiv(n, meta['BLOCK_N']),
        )
        _combined_activation_kernel[grid](
            input,
            weight1,
            weight2,
            bias,
            output,
            m,
            n,
            k,
            input.stride(-2),
            input.stride(-1),
            weight1.stride(-2),
            weight1.stride(-1),
            weight2.stride(-1),
            bias.stride(-1),
            output.stride(-2),
            output.stride(-1),
        )
        return output

    return combined_activation
