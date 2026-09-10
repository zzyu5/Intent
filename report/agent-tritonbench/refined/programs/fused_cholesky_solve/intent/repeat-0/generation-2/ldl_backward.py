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
            v15 = (v5 + 1)
            v16 = tl.load((intermediate + (v5) * S1_0 + (iv2) * S1_1), mask=v14, other=0.0)
            v17 = v16
            for iv18 in range(v15, D2, 1):
                v19 = D2
                v20 = (iv18 < v19)
                v21 = (iv18 >= 0)
                v22 = (v21 & v20)
                v23 = D2
                v24 = (v5 < v23)
                v25 = (v5 >= 0)
                v26 = (v25 & v24)
                v27 = (v22 & v26)
                v28 = D2
                v29 = (iv18 < v28)
                v30 = (iv18 >= 0)
                v31 = (v30 & v29)
                v32 = D1
                v33 = (iv2 < v32)
                v34 = (iv2 >= 0)
                v35 = (v34 & v33)
                v36 = (v31 & v35)
                v37 = tl.load((output + (iv18) * S2_0 + (iv2) * S2_1), mask=v36, other=0.0)
                v38 = tl.load((factor + (iv18) * S0_0 + (v5) * S0_1), mask=v27, other=0.0)
                v39 = (v38 * v37)
                v40 = (v17 - v39)
                v17 = v40
            v41 = D2
            v42 = (v5 < v41)
            v43 = (v5 >= 0)
            v44 = (v43 & v42)
            v45 = D1
            v46 = (iv2 < v45)
            v47 = (iv2 >= 0)
            v48 = (v47 & v46)
            v49 = (v44 & v48)
            tl.store((output + (v5) * S2_0 + (iv2) * S2_1), v17, mask=v49)

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
