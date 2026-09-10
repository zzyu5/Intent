import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_add_32x32_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    other_ptr,
    other_value,
    output_ptr,
    num_m,
    HAS_BIAS: tl.constexpr,
    OTHER_MODE: tl.constexpr,
    ALPHA: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    oc = tl.arange(0, BLOCK_N)

    m_mask = m < num_m
    spatial = m % 1024
    batch = m // 1024
    oh = spatial // 32
    ow = spatial % 32

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 288, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)
        ic = k // 9
        kernel_pos = k % 9
        kh = kernel_pos // 3
        kw = kernel_pos % 3

        ih = oh[:, None] + kh[None, :] - 1
        iw = ow[:, None] + kw[None, :] - 1
        input_offsets = (
            batch[:, None] * 32768
            + ic[None, :] * 1024
            + ih * 32
            + iw
        )
        input_mask = (
            m_mask[:, None]
            & (ih >= 0)
            & (ih < 32)
            & (iw >= 0)
            & (iw < 32)
        )
        lhs = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)

        weight_offsets = oc[None, :] * 288 + k[:, None]
        rhs = tl.load(weight_ptr + weight_offsets)
        acc += tl.dot(lhs, rhs, input_precision="ieee")

    output_offsets = (
        batch[:, None] * 32768
        + oc[None, :] * 1024
        + spatial[:, None]
    )
    if HAS_BIAS:
        acc += tl.load(bias_ptr + oc)[None, :]

    if OTHER_MODE == 1:
        acc += tl.load(other_ptr + output_offsets, mask=m_mask[:, None], other=0.0) * ALPHA
    elif OTHER_MODE == 2:
        acc += other_value * ALPHA

    tl.store(output_ptr + output_offsets, acc, mask=m_mask[:, None])


def build(context):
    def wrapper(
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
        out=None,
    ):
        output = out if out is not None else torch.empty_like(input)
        num_m = input.shape[0] * 1024
        if other is None:
            other_ptr = input
            other_value = 0.0
            other_mode = 0
        elif isinstance(other, torch.Tensor):
            other_ptr = other
            other_value = 0.0
            other_mode = 1
        else:
            other_ptr = input
            other_value = float(other)
            other_mode = 2

        bias_ptr = bias if bias is not None else input
        grid = (triton.cdiv(num_m, 128),)
        _conv2d_add_32x32_kernel[grid](
            input,
            weight,
            bias_ptr,
            other_ptr,
            other_value,
            output,
            num_m,
            HAS_BIAS=bias is not None,
            OTHER_MODE=other_mode,
            ALPHA=float(alpha),
            BLOCK_M=128,
            BLOCK_N=32,
            BLOCK_K=32,
            num_warps=4,
            num_stages=2,
        )
        return output

    return wrapper
