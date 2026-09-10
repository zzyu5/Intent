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
                v27 = (iv4 < v17)
                v28 = (iv4 >= 0)
                v29 = (v28 & v27)
                v30 = (iv16 >= 0)
                v31 = (v30 & v22)
                v32 = (v29 & v31)
                v33 = tl.load((factor + (iv4) * S1_0 + (iv16) * S1_1), mask=v32, other=0.0)
                v34 = (v26 * v33)
                v35 = (v15 - v34)
                v15 = v35
            v36 = (iv2 == iv4)
            if v36:
                v37 = libdevice.sqrt(tl.cast(v15, tl.float32))
                v38 = D1
                v39 = (iv2 < v38)
                v40 = (iv2 >= 0)
                v41 = (v40 & v39)
                v42 = D1
                v43 = (iv4 < v42)
                v44 = (iv4 >= 0)
                v45 = (v44 & v43)
                v46 = (v41 & v45)
                tl.store((factor + (iv2) * S1_0 + (iv4) * S1_1), v37, mask=v46)
            else:
                v47 = D1
                v48 = (iv4 < v47)
                v49 = (iv4 >= 0)
                v50 = (v49 & v48)
                v51 = D1
                v52 = (iv4 < v51)
                v53 = (iv4 >= 0)
                v54 = (v53 & v52)
                v55 = (v50 & v54)
                v56 = tl.load((factor + (iv4) * S1_0 + (iv4) * S1_1), mask=v55, other=0.0)
                v57 = tl.fdiv(tl.cast(v15, tl.float32), tl.cast(v56, tl.float32), ieee_rounding=True)
                v58 = (iv2 < v47)
                v59 = (iv2 >= 0)
                v60 = (v59 & v58)
                v61 = (iv4 >= 0)
                v62 = (v61 & v52)
                v63 = (v60 & v62)
                tl.store((factor + (iv2) * S1_0 + (iv4) * S1_1), v57, mask=v63)

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
