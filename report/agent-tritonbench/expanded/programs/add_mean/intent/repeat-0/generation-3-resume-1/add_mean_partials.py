import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "other", "partials", ), (False, False, True, ))

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
    key=["D1", "D4", "S0_0", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, other, partials, D1: tl.constexpr, D4: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 16)
    v2 = (v1 * 1)
    v3 = (0 + v2)
    v4 = (v3 * 65536)
    v5 = (v4 + 65536)
    v6 = (v4 + tl.arange(0, 65536) * 1)
    v7 = D1
    v8 = tl.full((65536,), v7, tl.int64)
    v9 = tl.full((65536,), 0, tl.int64)
    v10 = (v6 < v8)
    v11 = (v6 >= v9)
    v12 = (v11 & v10)
    v13 = tl.full((65536,), 0.0, tl.float32)
    v14 = (v4 + tl.arange(0, 65536) * 1)
    v15 = D1
    v16 = tl.full((65536,), v15, tl.int64)
    v17 = tl.full((65536,), 0, tl.int64)
    v18 = (v14 < v16)
    v19 = (v14 >= v17)
    v20 = (v19 & v18)
    v21 = tl.full((65536,), 0.0, tl.float32)
    v22 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v4, tl.int64) * S0_0), shape=((D1 - tl.cast(v4, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(65536,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v23 = tl.load(tl.make_block_ptr(base=(other + tl.cast(v4, tl.int64) * S1_0), shape=((D1 - tl.cast(v4, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(65536,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v24 = (v22 + v23)
    v25 = tl.sum(v24, axis=0)
    v26 = (v3 < 16)
    v27 = (v3 >= 0)
    v28 = (v27 & v26)
    tl.store((partials + (v3) * S2_0), v25, mask=v28)

def launch(input, other, partials):
    D1 = input.shape[0]
    D4 = partials.shape[0]
    S0_0 = input.stride(0)
    S1_0 = other.stride(0)
    S2_0 = partials.stride(0)
    grid = lambda META: (16,)
    return _intent_kernel[grid](input, other, partials, D1, D4, S0_0, S1_0, S2_0)

def run(input, other):
    partials = torch.empty((16,), device=input.device, dtype=torch.float32)
    launch(input, other, partials)
    return partials
