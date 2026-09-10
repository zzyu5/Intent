import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1, a2, a3):
    v0 = (a0 > a2)
    v1 = (a0 == a2)
    v2 = (a1 <= a3)
    v3 = (v1 & v2)
    v4 = (v0 | v3)
    v5 = tl.where(v4, a0, a2)
    v6 = tl.where(v4, a1, a3)
    return (v5, v6)

def _intent_cover_FULL_D1(args):
    bound = int(args["D1"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D1")

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

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
@triton.heuristics({
    "FULL_D1": _intent_cover_FULL_D1,
})
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FULL_D1: tl.constexpr):
    v7 = tl.program_id(0)
    v8 = (v7 % 1)
    v9 = (0 + tl.arange(0, FULL_D1) * 1)
    v10 = (D1 - 0)
    v11 = (1 - 1)
    v12 = (v10 + v11)
    v13 = ((v12 // 1) - (((v12 % 1) != 0) & (((v12 % 1) < 0) != (1 < 0))))
    v14 = (v13 * 1)
    v15 = (0 + v14)
    v16 = v15
    v17 = (v9 < v16)
    v18 = D1
    v19 = tl.full((FULL_D1,), v18, tl.int64)
    v20 = (v9 < v19)
    v21 = tl.full((FULL_D1,), 0.0, tl.float16)
    v22 = FULL_D1
    v23 = (0 + tl.arange(0, FULL_D1) * 1)
    v24 = v23
    v25 = tl.cast(v24, tl.int32)
    v26 = (v20 & v17)
    v27 = tl.load(tl.make_block_ptr(base=(input + tl.cast(0, tl.int64) * S0_0), shape=((D1 - tl.cast(0, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FULL_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v28, v29 = tl.reduce((v27, v25), axis=0, combine_fn=_intent_reduce_0)
    v30 = tl.cast(v29, tl.int64)
    tl.store((output + (0) * S1_0), v30)

def launch(input, output):
    D1 = input.shape[0]
    D2 = output.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, output, D1, D2, S0_0, S1_0)

def run(input):
    output = torch.empty((1,), device=input.device, dtype=torch.int64)
    launch(input, output)
    return output
