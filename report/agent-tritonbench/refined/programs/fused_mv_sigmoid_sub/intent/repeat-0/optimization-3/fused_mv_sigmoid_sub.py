import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "vec", "output"), (False, False, True))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256, "ALGO": 0}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 512, "ALGO": 0}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 1024, "ALGO": 0}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 512, "ALGO": 0}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 1024, "ALGO": 0}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 256, "ALGO": 0}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 512, "ALGO": 0}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "ALGO": 0}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256, "ALGO": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 512, "ALGO": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 1024, "ALGO": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 512, "ALGO": 1}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 1024, "ALGO": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 256, "ALGO": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 512, "ALGO": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256, "ALGO": 1}, num_warps=4, num_stages=2),
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
    alpha: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_v: tl.constexpr,
    stride_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ALGO: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    if ALGO == 0:
        acc = tl.zeros((BLOCK_M, 1), dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, N, BLOCK_K):
        cols = k_start + tl.arange(0, BLOCK_K)
        col_mask = cols < N
        if M % BLOCK_M == 0 and N % BLOCK_K == 0:
            a = tl.load(
                input + rows[:, None] * stride_am + cols[None, :] * stride_ak,
                cache_modifier=".cg",
            )
            v = tl.load(vec + cols * stride_v, cache_modifier=".ca")
        else:
            mask = row_mask[:, None] & col_mask[None, :]
            a = tl.load(
                input + rows[:, None] * stride_am + cols[None, :] * stride_ak,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
            )
            v = tl.load(vec + cols * stride_v, mask=col_mask, other=0.0, cache_modifier=".ca")
        if ALGO == 0:
            acc = tl.dot(a, v[:, None], acc, input_precision="ieee")
        else:
            acc += tl.sum(a * v[None, :], axis=1)

    if ALGO == 0:
        value = tl.reshape(acc, (BLOCK_M,))
    else:
        value = acc
    value = 0.5 + 0.5 * libdevice.tanh(0.5 * value)
    value = value - alpha * other
    if M % BLOCK_M == 0:
        tl.store(output + rows * stride_out, value)
    else:
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
