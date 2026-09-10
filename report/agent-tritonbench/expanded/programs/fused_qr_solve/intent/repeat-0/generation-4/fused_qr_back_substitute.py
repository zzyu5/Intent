import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("factor", "transformed_rhs", "solution", ), (False, False, True, ))

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
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(factor, transformed_rhs, solution, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        v3 = (D1 - 1)
        v4 = (v3 - iv2)
        for iv5 in range(0, D3, 1):
            v6 = D2
            v7 = (v4 < v6)
            v8 = (v4 >= 0)
            v9 = (v8 & v7)
            v10 = D3
            v11 = (iv5 < v10)
            v12 = (iv5 >= 0)
            v13 = (v12 & v11)
            v14 = (v9 & v13)
            v15 = (v4 + 1)
            v16 = tl.load((transformed_rhs + (v4) * S1_0 + (iv5) * S1_1), mask=v14, other=0.0)
            v17 = v16
            for iv18 in range(v15, D1, 1):
                v19 = D2
                v20 = (v4 < v19)
                v21 = (v4 >= 0)
                v22 = (v21 & v20)
                v23 = D1
                v24 = (iv18 < v23)
                v25 = (iv18 >= 0)
                v26 = (v25 & v24)
                v27 = (v22 & v26)
                v28 = D1
                v29 = (iv18 < v28)
                v30 = (iv18 >= 0)
                v31 = (v30 & v29)
                v32 = D3
                v33 = (iv5 < v32)
                v34 = (iv5 >= 0)
                v35 = (v34 & v33)
                v36 = (v31 & v35)
                v37 = tl.load((solution + (iv18) * S2_0 + (iv5) * S2_1), mask=v36, other=0.0)
                v38 = tl.load((factor + (v4) * S0_0 + (iv18) * S0_1), mask=v27, other=0.0)
                v39 = (v38 * v37)
                v40 = (v17 - v39)
                v17 = v40
            v41 = D2
            v42 = (v4 < v41)
            v43 = (v4 >= 0)
            v44 = (v43 & v42)
            v45 = D1
            v46 = (v4 < v45)
            v47 = (v4 >= 0)
            v48 = (v47 & v46)
            v49 = (v44 & v48)
            v50 = D1
            v51 = (v4 < v50)
            v52 = (v4 >= 0)
            v53 = (v52 & v51)
            v54 = D3
            v55 = (iv5 < v54)
            v56 = (iv5 >= 0)
            v57 = (v56 & v55)
            v58 = (v53 & v57)
            v59 = tl.load((factor + (v4) * S0_0 + (v4) * S0_1), mask=v49, other=0.0)
            v60 = tl.fdiv(tl.cast(v17, tl.float32), tl.cast(v59, tl.float32), ieee_rounding=True)
            tl.store((solution + (v4) * S2_0 + (iv5) * S2_1), v60, mask=v58)

def launch(factor, transformed_rhs, solution):
    D1 = factor.shape[1]
    D2 = factor.shape[0]
    D3 = transformed_rhs.shape[1]
    S0_0 = factor.stride(0)
    S0_1 = factor.stride(1)
    S1_0 = transformed_rhs.stride(0)
    S1_1 = transformed_rhs.stride(1)
    S2_0 = solution.stride(0)
    S2_1 = solution.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](factor, transformed_rhs, solution, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(factor, transformed_rhs, solution):
    launch(factor, transformed_rhs, solution)
    return None
