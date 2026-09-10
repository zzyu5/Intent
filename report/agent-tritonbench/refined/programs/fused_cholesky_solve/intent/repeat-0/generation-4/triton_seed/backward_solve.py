import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("factor", "intermediate", "output", ), (False, False, True, ))

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
def _intent_kernel(factor, intermediate, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        for iv3 in range(0, D2, 1):
            v4 = (D2 - 1)
            v5 = (v4 - iv3)
            v6 = D2
            v7 = (v5 < v6)
            v8 = (v5 >= 0)
            v9 = (v8 & v7)
            v10 = D1
            v11 = (iv2 < v10)
            v12 = (iv2 >= 0)
            v13 = (v12 & v11)
            v14 = (v9 & v13)
            v15 = D2
            v16 = (v5 < v15)
            v17 = (v5 >= 0)
            v18 = (v17 & v16)
            v19 = D2
            v20 = (v5 < v19)
            v21 = (v5 >= 0)
            v22 = (v21 & v20)
            v23 = (v18 & v22)
            v24 = (v5 + 1)
            v25 = tl.load((intermediate + (v5) * S1_0 + (iv2) * S1_1), mask=v14, other=0.0)
            v26 = tl.load((factor + (v5) * S0_0 + (v5) * S0_1), mask=v23, other=0.0)
            v27 = tl.fdiv(tl.cast(v25, tl.float32), tl.cast(v26, tl.float32), ieee_rounding=True)
            v28 = v27
            for iv29 in range(v24, D2, 1):
                v30 = D2
                v31 = (iv29 < v30)
                v32 = (iv29 >= 0)
                v33 = (v32 & v31)
                v34 = D2
                v35 = (v5 < v34)
                v36 = (v5 >= 0)
                v37 = (v36 & v35)
                v38 = (v33 & v37)
                v39 = D2
                v40 = (iv29 < v39)
                v41 = (iv29 >= 0)
                v42 = (v41 & v40)
                v43 = D1
                v44 = (iv2 < v43)
                v45 = (iv2 >= 0)
                v46 = (v45 & v44)
                v47 = (v42 & v46)
                v48 = tl.load((output + (iv29) * S2_0 + (iv2) * S2_1), mask=v47, other=0.0)
                v49 = tl.load((factor + (iv29) * S0_0 + (v5) * S0_1), mask=v38, other=0.0)
                v50 = (v49 * v48)
                v51 = (v28 - v50)
                v28 = v51
            v52 = D2
            v53 = (v5 < v52)
            v54 = (v5 >= 0)
            v55 = (v54 & v53)
            v56 = D1
            v57 = (iv2 < v56)
            v58 = (iv2 >= 0)
            v59 = (v58 & v57)
            v60 = (v55 & v59)
            tl.store((output + (v5) * S2_0 + (iv2) * S2_1), v28, mask=v60)

def launch(factor, intermediate, output):
    D1 = intermediate.shape[1]
    D2 = factor.shape[0]
    S0_0 = factor.stride(0)
    S0_1 = factor.stride(1)
    S1_0 = intermediate.stride(0)
    S1_1 = intermediate.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](factor, intermediate, output, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(factor, intermediate, output):
    launch(factor, intermediate, output)
    return None
