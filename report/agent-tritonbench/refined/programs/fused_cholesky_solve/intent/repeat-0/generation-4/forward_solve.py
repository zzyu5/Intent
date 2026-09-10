import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("factor", "rhs", "intermediate", ), (False, False, True, ))

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
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(factor, rhs, intermediate, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        for iv3 in range(0, D2, 1):
            v4 = D2
            v5 = (iv3 < v4)
            v6 = (iv3 >= 0)
            v7 = (v6 & v5)
            v8 = D1
            v9 = (iv2 < v8)
            v10 = (iv2 >= 0)
            v11 = (v10 & v9)
            v12 = (v7 & v11)
            v13 = tl.load((rhs + (iv3) * S1_0 + (iv2) * S1_1), mask=v12, other=0.0)
            v14 = v13
            for iv15 in range(0, iv3, 1):
                v16 = D2
                v17 = (iv3 < v16)
                v18 = (iv3 >= 0)
                v19 = (v18 & v17)
                v20 = D2
                v21 = (iv15 < v20)
                v22 = (iv15 >= 0)
                v23 = (v22 & v21)
                v24 = (v19 & v23)
                v25 = D2
                v26 = (iv15 < v25)
                v27 = (iv15 >= 0)
                v28 = (v27 & v26)
                v29 = D1
                v30 = (iv2 < v29)
                v31 = (iv2 >= 0)
                v32 = (v31 & v30)
                v33 = (v28 & v32)
                v34 = tl.load((intermediate + (iv15) * S2_0 + (iv2) * S2_1), mask=v33, other=0.0)
                v35 = tl.load((factor + (iv3) * S0_0 + (iv15) * S0_1), mask=v24, other=0.0)
                v36 = (v35 * v34)
                v37 = (v14 - v36)
                v14 = v37
            v38 = D2
            v39 = (iv3 < v38)
            v40 = (iv3 >= 0)
            v41 = (v40 & v39)
            v42 = D1
            v43 = (iv2 < v42)
            v44 = (iv2 >= 0)
            v45 = (v44 & v43)
            v46 = (v41 & v45)
            tl.store((intermediate + (iv3) * S2_0 + (iv2) * S2_1), v14, mask=v46)

def launch(factor, rhs, intermediate):
    D1 = rhs.shape[1]
    D2 = factor.shape[0]
    S0_0 = factor.stride(0)
    S0_1 = factor.stride(1)
    S1_0 = rhs.stride(0)
    S1_1 = rhs.stride(1)
    S2_0 = intermediate.stride(0)
    S2_1 = intermediate.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](factor, rhs, intermediate, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(factor, rhs, intermediate):
    launch(factor, rhs, intermediate)
    return None
