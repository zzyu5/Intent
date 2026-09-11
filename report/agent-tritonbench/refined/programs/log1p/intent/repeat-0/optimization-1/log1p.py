import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks


_intent_tuning_hooks = TuningHooks(("input", "output"), (False, True))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 128}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 128}, num_warps=8, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1, num_ctas=1),
    ],
    key=["N", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(
    input,
    output,
    N,
    S0_0,
    S1_0,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(input + offsets * S0_0, mask=mask, other=0.0)
    result = libdevice.log1p(values)
    tl.store(output + offsets * S1_0, result, mask=mask)


def launch(input, output):
    N = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(N, META["BLOCK"]),)
    return _intent_kernel[grid](input, output, N, S0_0, S1_0)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
