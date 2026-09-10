import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1):
    v0 = tl.maximum(a0, a1, propagate_nan=tl.PropagateNan.ALL)
    return v0

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
    key=["D1", "D3", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D1": _intent_cover_FULL_D1,
})
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FULL_D1: tl.constexpr):
    v1 = tl.program_id(0)
    v2 = (v1 % 1)
    v3 = (0 + tl.arange(0, FULL_D1) * 1)
    v4 = (D1 - 0)
    v5 = (1 - 1)
    v6 = (v4 + v5)
    v7 = ((v6 // 1) - (((v6 % 1) != 0) & (((v6 % 1) < 0) != (1 < 0))))
    v8 = (v7 * 1)
    v9 = (0 + v8)
    v10 = v9
    v11 = (v3 < v10)
    v12 = D1
    v13 = tl.full((FULL_D1,), v12, tl.int64)
    v14 = (v3 < v13)
    v15 = tl.full((FULL_D1,), 0.0, tl.float32)
    v16 = (v14 & v11)
    v17 = (0 + D1)
    v18 = v17
    v19 = (v3 < v18)
    v20 = -float("inf")
    v21 = tl.load(tl.make_block_ptr(base=(input + tl.cast(0, tl.int64) * S0_0), shape=((D1 - tl.cast(0, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FULL_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v22 = tl.where(v19, v21, v20)
    v23 = tl.reduce(v22, axis=0, combine_fn=_intent_reduce_0)
    v24 = v23
    v25 = 0.0
    v26 = (v21 - v24)
    v27 = libdevice.exp(tl.cast(v26, tl.float32))
    v28 = tl.where(v19, v27, v25)
    v29 = tl.sum(v28, axis=0)
    v30 = libdevice.log(tl.cast(v29, tl.float32))
    v31 = (v30 + v23)
    v32 = v31
    v33 = (0 + tl.arange(0, 1) * 1)
    v34 = tl.full((1,), 1, tl.int64)
    v35 = (v33 < v34)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(0, tl.int64) * S1_0), shape=((1 - tl.cast(0, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(1,), order=(0,)), tl.cast(v32, tl.float32))

def launch(input, output):
    D1 = input.shape[0]
    D3 = output.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, output, D1, D3, S0_0, S1_0)

def run(input):
    output = torch.empty((1,), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
