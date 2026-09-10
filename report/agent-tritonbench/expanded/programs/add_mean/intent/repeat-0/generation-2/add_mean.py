import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

def _intent_cover_FULL_D1(args):
    bound = int(args["D1"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D1")

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

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
    key=["D1", "D2", "S0_0", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D1": _intent_cover_FULL_D1,
})
@triton.jit
def _intent_kernel(input, other, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, FULL_D1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = (0 + tl.arange(0, FULL_D1) * 1)
    v3 = (D1 - 0)
    v4 = (1 - 1)
    v5 = (v3 + v4)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = v8
    v10 = (v2 < v9)
    v11 = D1
    v12 = tl.full((FULL_D1,), v11, tl.int64)
    v13 = (v2 < v12)
    v14 = tl.full((FULL_D1,), 0.0, tl.float32)
    v15 = (0 + tl.arange(0, FULL_D1) * 1)
    v16 = (1 - 1)
    v17 = (v3 + v16)
    v18 = ((v17 // 1) - (((v17 % 1) != 0) & (((v17 % 1) < 0) != (1 < 0))))
    v19 = (v18 * 1)
    v20 = (0 + v19)
    v21 = v20
    v22 = (v15 < v21)
    v23 = D1
    v24 = tl.full((FULL_D1,), v23, tl.int64)
    v25 = (v15 < v24)
    v26 = tl.full((FULL_D1,), 0.0, tl.float32)
    v27 = (v25 & v22)
    v28 = (v13 & v10)
    v29 = (0 + D1)
    v30 = v29
    v31 = (v15 < v30)
    v32 = 0.0
    v33 = tl.load(tl.make_block_ptr(base=(input + tl.cast(0, tl.int64) * S0_0), shape=((D1 - tl.cast(0, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FULL_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v34 = tl.load(tl.make_block_ptr(base=(other + tl.cast(0, tl.int64) * S1_0), shape=((D1 - tl.cast(0, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FULL_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v35 = (v33 + v34)
    v36 = tl.where(v31, v35, v32)
    v37 = tl.sum(v36, axis=0)
    v38 = tl.cast(D1, tl.float32)
    v39 = tl.fdiv(tl.cast(v37, tl.float32), tl.cast(v38, tl.float32), ieee_rounding=True)
    tl.store((output + (0) * S2_0), v39)

def launch(input, other, output):
    D1 = input.shape[0]
    D2 = output.shape[0]
    S0_0 = input.stride(0)
    S1_0 = other.stride(0)
    S2_0 = output.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, other, output, D1, D2, S0_0, S1_0, S2_0)

def run(input, other):
    output = torch.empty((1,), device=input.device, dtype=torch.float32)
    launch(input, other, output)
    return output
