import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("matrix", "factor", ), (False, True, ))

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
    key=["D1", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(matrix, factor, D1: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        v3 = (iv2 + 1)
        for iv4 in range(0, v3, 1):
            v5 = D1
            v6 = (iv2 < v5)
            v7 = (iv2 >= 0)
            v8 = (v7 & v6)
            v9 = D1
            v10 = (iv4 < v9)
            v11 = (iv4 >= 0)
            v12 = (v11 & v10)
            v13 = (v8 & v12)
            v14 = tl.load((matrix + (iv2) * S0_0 + (iv4) * S0_1), mask=v13, other=0.0)
            v15 = v14
            for iv16 in range(0, iv4, 1):
                v17 = D1
                v18 = (iv2 < v17)
                v19 = (iv2 >= 0)
                v20 = (v19 & v18)
                v21 = D1
                v22 = (iv16 < v21)
                v23 = (iv16 >= 0)
                v24 = (v23 & v22)
                v25 = (v20 & v24)
                v26 = tl.load((factor + (iv2) * S1_0 + (iv16) * S1_1), mask=v25, other=0.0)
                v27 = (iv16 < v17)
                v28 = (iv16 >= 0)
                v29 = (v28 & v27)
                v30 = (iv16 >= 0)
                v31 = (v30 & v22)
                v32 = (v29 & v31)
                v33 = tl.load((factor + (iv16) * S1_0 + (iv16) * S1_1), mask=v32, other=0.0)
                v34 = (v26 * v33)
                v35 = (iv4 < v17)
                v36 = (iv4 >= 0)
                v37 = (v36 & v35)
                v38 = (iv16 >= 0)
                v39 = (v38 & v22)
                v40 = (v37 & v39)
                v41 = tl.load((factor + (iv4) * S1_0 + (iv16) * S1_1), mask=v40, other=0.0)
                v42 = (v34 * v41)
                v43 = (v15 - v42)
                v15 = v43
            v44 = (iv2 == iv4)
            if v44:
                v45 = D1
                v46 = (iv2 < v45)
                v47 = (iv2 >= 0)
                v48 = (v47 & v46)
                v49 = D1
                v50 = (iv4 < v49)
                v51 = (iv4 >= 0)
                v52 = (v51 & v50)
                v53 = (v48 & v52)
                tl.store((factor + (iv2) * S1_0 + (iv4) * S1_1), v15, mask=v53)
            else:
                v54 = D1
                v55 = (iv4 < v54)
                v56 = (iv4 >= 0)
                v57 = (v56 & v55)
                v58 = D1
                v59 = (iv4 < v58)
                v60 = (iv4 >= 0)
                v61 = (v60 & v59)
                v62 = (v57 & v61)
                v63 = tl.load((factor + (iv4) * S1_0 + (iv4) * S1_1), mask=v62, other=0.0)
                v64 = tl.fdiv(tl.cast(v15, tl.float32), tl.cast(v63, tl.float32), ieee_rounding=True)
                v65 = (iv2 < v54)
                v66 = (iv2 >= 0)
                v67 = (v66 & v65)
                v68 = (iv4 >= 0)
                v69 = (v68 & v59)
                v70 = (v67 & v69)
                tl.store((factor + (iv2) * S1_0 + (iv4) * S1_1), v64, mask=v70)

def launch(matrix, factor):
    D1 = matrix.shape[0]
    S0_0 = matrix.stride(0)
    S0_1 = matrix.stride(1)
    S1_0 = factor.stride(0)
    S1_1 = factor.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](matrix, factor, D1, S0_0, S0_1, S1_0, S1_1)

def run(matrix, factor):
    launch(matrix, factor)
    return None
