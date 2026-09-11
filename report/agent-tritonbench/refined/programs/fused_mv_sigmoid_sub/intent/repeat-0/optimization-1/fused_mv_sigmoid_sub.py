import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "vec", "output"), (False, False, True))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 128}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_K": 256}, num_warps=8, num_stages=2),
    ],
    key=["M", "N", "stride_am", "stride_ak", "stride_v", "stride_out"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(
    input,
    vec,
    output,
    other,
    alpha,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_v: tl.constexpr,
    stride_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    acc = tl.zeros((BLOCK_M, 1), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        cols = k_start + tl.arange(0, BLOCK_K)
        col_mask = cols < N
        mask = row_mask[:, None] & col_mask[None, :]
        a = tl.load(
            input + rows[:, None] * stride_am + cols[None, :] * stride_ak,
            mask=mask,
            other=0.0,
        )
        v = tl.load(vec + cols * stride_v, mask=col_mask, other=0.0)
        acc = tl.dot(a, v[:, None], acc, input_precision="ieee")

    value = tl.reshape(acc, (BLOCK_M,))
    value = 0.5 + 0.5 * libdevice.tanh(0.5 * value)
    value = value - alpha * other
    tl.store(output + rows * stride_out, value, mask=row_mask)


def launch(input, vec, output, other, alpha):
    M = input.shape[0]
    N = input.shape[1]
    stride_am = input.stride(0)
    stride_ak = input.stride(1)
    stride_v = vec.stride(0)
    stride_out = output.stride(0)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    return _intent_kernel[grid](
        input,
        vec,
        output,
        other,
        alpha,
        M,
        N,
        stride_am,
        stride_ak,
        stride_v,
        stride_out,
    )


def run(input, vec, other, alpha):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, vec, output, other, alpha)
    return output
