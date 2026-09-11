import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks


_intent_tuning_hooks = TuningHooks(("input", "other", "output"), (False, False, True,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 8192}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 16384}, num_warps=8, num_stages=1),
    ],
    key=["n"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _kernel(
    input,
    other,
    output,
    alpha,
    n,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    x = tl.load(input + offsets * S0, mask=mask, other=0).to(tl.float32)
    y = tl.load(other + offsets * S1, mask=mask, other=0).to(tl.float32)
    x = x + alpha * y

    x2 = x * x
    inner = x + 0.044714998453855515 * (x2 * x)
    inner = 0.7978845834732056 * inner
    result = 0.5 * x * (1.0 + libdevice.tanh(inner))
    tl.store(output + offsets * S2, result.to(tl.float16), mask=mask)


def launch(input, other, output, alpha):
    n = input.shape[0]
    grid = lambda META: (triton.cdiv(n, META["BLOCK"]),)
    return _kernel[grid](
        input,
        other,
        output,
        alpha,
        n,
        S0=input.stride(0),
        S1=other.stride(0),
        S2=output.stride(0),
    )


def run(input, other, alpha):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float16)
    launch(input, other, output, alpha)
    return output
