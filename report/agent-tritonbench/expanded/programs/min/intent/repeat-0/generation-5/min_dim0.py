import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output_values", "output_indices", ), (False, True, True, ))

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
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output_values, output_indices, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * 1)
    v6 = (0 + v5)
    v7 = D2
    v8 = (0 < v7)
    v9 = D1
    v10 = (v6 < v9)
    v11 = (v6 >= 0)
    v12 = (v11 & v10)
    v13 = (v8 & v12)
    v14 = tl.load((input + (0) * S0_0 + (v6) * S0_1), mask=v13, other=0.0)
    v15 = v14
    v16 = 0
    for iv17 in range(1, D2, 1):
        v18 = D2
        v19 = (iv17 < v18)
        v20 = (iv17 >= 0)
        v21 = (v20 & v19)
        v22 = D1
        v23 = (v6 < v22)
        v24 = (v6 >= 0)
        v25 = (v24 & v23)
        v26 = (v21 & v25)
        v27 = tl.load((input + (iv17) * S0_0 + (v6) * S0_1), mask=v26, other=0.0)
        v28 = (v27 != v27)
        if v28:
            v31 = (v15 == v15)
            if v31:
                v34 = tl.cast(iv17, tl.int64)
                v32 = v27
                v33 = v34
            else:
                v32 = v15
                v33 = v16
            v29 = v32
            v30 = v33
        else:
            v35 = (v15 == v15)
            if v35:
                v38 = (v27 < v15)
                if v38:
                    v41 = tl.cast(iv17, tl.int64)
                    v39 = v27
                    v40 = v41
                else:
                    v39 = v15
                    v40 = v16
                v36 = v39
                v37 = v40
            else:
                v36 = v15
                v37 = v16
            v29 = v36
            v30 = v37
        v15 = v29
        v16 = v30
    v42 = D1
    v43 = (v6 < v42)
    v44 = (v6 >= 0)
    v45 = (v44 & v43)
    tl.store((output_values + (v6) * S1_0), v15, mask=v45)
    v46 = D1
    v47 = (v6 < v46)
    v48 = (v6 >= 0)
    v49 = (v48 & v47)
    tl.store((output_indices + (v6) * S2_0), v16, mask=v49)

def launch(input, output_values, output_indices):
    D1 = input.shape[1]
    D2 = input.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output_values.stride(0)
    S2_0 = output_indices.stride(0)
    grid = lambda META: (D1,)
    return _intent_kernel[grid](input, output_values, output_indices, D1, D2, S0_0, S0_1, S1_0, S2_0)

def run(input):
    output_values = torch.empty((input.shape[1],), device=input.device, dtype=torch.float32)
    output_indices = torch.empty((input.shape[1],), device=input.device, dtype=torch.int64)
    launch(input, output_values, output_indices)
    return (output_values, output_indices)
