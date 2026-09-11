import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "vec", "output"), (False, False, True))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 1024}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 1024}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 512}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 512}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 256}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 512}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_K": 1024}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_K": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_K": 256}, num_warps=4, num_stages=2),
    ],
    key=["stride_out"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(
    input,
    vec,
    output,
    other: tl.constexpr,
    alpha: tl.constexpr,
    stride_out: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, 1024, BLOCK_K):
        cols = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            input + rows[:, None] * 1024 + cols[None, :],
            cache_modifier=".cg",
        )
        v = tl.load(vec + cols, cache_modifier=".ca")
        acc += tl.sum(a * v[None, :], axis=1)

    if alpha == 1 and other == 0.5:
        value = 0.5 * libdevice.tanh(0.5 * acc)
    else:
        value = 0.5 + 0.5 * libdevice.tanh(0.5 * acc) - alpha * other
    tl.store(output + rows * stride_out, value)


def launch(input, vec, output, other, alpha):
    stride_out = output.stride(0)
    grid = lambda META: (1024 // META["BLOCK_M"],)
    return _intent_kernel[grid](
        input,
        vec,
        output,
        other,
        alpha,
        stride_out,
    )


def run(input, vec, other, alpha):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, vec, output, other, alpha)
    return output
