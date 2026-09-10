import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("partials", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(partials, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = (0 + tl.arange(0, 16) * 1)
    v3 = tl.full((16,), 16, tl.int64)
    v4 = (v2 < v3)
    v5 = tl.full((16,), 0.0, tl.float32)
    v6 = tl.load(tl.make_block_ptr(base=(partials + tl.cast(0, tl.int64) * S0_0), shape=((16 - tl.cast(0, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(16,), order=(0,)))
    v7 = tl.sum(v6, axis=0)
    v8 = tl.cast(1048576, tl.float32)
    v9 = tl.fdiv(tl.cast(v7, tl.float32), tl.cast(v8, tl.float32), ieee_rounding=True)
    tl.store((output + (0) * S1_0), v9)

def launch(partials, output):
    D1 = partials.shape[0]
    D2 = output.shape[0]
    S0_0 = partials.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](partials, output, D1, D2, S0_0, S1_0)

def run(partials):
    output = torch.empty((1,), device=partials.device, dtype=torch.float32)
    launch(partials, output)
    return output
