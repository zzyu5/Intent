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

def _intent_cover_FULL_D2(args):
    bound = int(args["D2"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D2")

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
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D2": _intent_cover_FULL_D2,
})
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, FULL_D2: tl.constexpr):
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = tl.program_id(0)
    v5 = (v4 % v3)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (0 + tl.arange(0, FULL_D2) * 1)
    v9 = (D2 - 0)
    v10 = (1 - 1)
    v11 = (v9 + v10)
    v12 = ((v11 // 1) - (((v11 % 1) != 0) & (((v11 % 1) < 0) != (1 < 0))))
    v13 = (v12 * 1)
    v14 = (0 + v13)
    v15 = tl.full((FULL_D2,), v14, tl.int64)
    v16 = (v8 < v15)
    v17 = D1
    v18 = (v7 < v17)
    v19 = (v7 >= 0)
    v20 = (v19 & v18)
    v21 = tl.full((FULL_D2,), v20, tl.int1)
    v22 = D2
    v23 = tl.full((FULL_D2,), v22, tl.int64)
    v24 = (v8 < v23)
    v25 = (v21 & v24)
    v26 = tl.full((FULL_D2,), 0.0, tl.float32)
    v27 = (v25 & v16)
    v28 = (0 + D2)
    v29 = tl.full((FULL_D2,), v28, tl.int64)
    v30 = (v8 < v29)
    v31 = tl.full((FULL_D2,), -float("inf"), tl.float32)
    v32 = tl.load((input + (tl.full((FULL_D2,), v7, tl.int64)) * S0_0 + (tl.broadcast_to(v8, (FULL_D2,))) * S0_1), mask=v27, other=v26)
    v33 = tl.where(v30, v32, v31)
    v34 = tl.reduce(v33, axis=0, combine_fn=_intent_reduce_0)
    v35 = tl.full((FULL_D2,), v34, tl.float32)
    v36 = tl.full((FULL_D2,), 1.4426950216293335, tl.float32)
    v37 = tl.full((FULL_D2,), 0.0, tl.float32)
    v38 = (v32 - v35)
    v39 = (v38 * v36)
    v40 = libdevice.exp2(tl.cast(v39, tl.float32))
    v41 = tl.where(v30, v40, v37)
    v42 = tl.sum(v41, axis=0)
    v43 = tl.full((FULL_D2,), v42, tl.float32)
    v44 = (0 + tl.arange(0, FULL_D2) * 1)
    v45 = (1 - 1)
    v46 = (v9 + v45)
    v47 = ((v46 // 1) - (((v46 % 1) != 0) & (((v46 % 1) < 0) != (1 < 0))))
    v48 = (v47 * 1)
    v49 = (0 + v48)
    v50 = tl.full((FULL_D2,), v49, tl.int64)
    v51 = (v44 < v50)
    v52 = D1
    v53 = (v7 < v52)
    v54 = (v7 >= 0)
    v55 = (v54 & v53)
    v56 = tl.full((FULL_D2,), v55, tl.int1)
    v57 = D2
    v58 = tl.full((FULL_D2,), v57, tl.int64)
    v59 = (v44 < v58)
    v60 = (v56 & v59)
    v61 = (v60 & v51)
    v62 = tl.fdiv(tl.cast(v40, tl.float32), tl.cast(v43, tl.float32), ieee_rounding=True)
    tl.store((output + (tl.full((FULL_D2,), v7, tl.int64)) * S1_0 + (tl.broadcast_to(v44, (FULL_D2,))) * S1_1), v62, mask=v61)

def launch(input, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    grid = lambda META: (D1,)
    return _intent_kernel[grid](input, output, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(input):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
