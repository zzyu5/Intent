import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("x", "columns", ), (False, True, ))

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
    key=["D1", "D2", "D3", "D4", "D9", "D10", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(x, columns, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D9: tl.constexpr, D10: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D2 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D1 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D3 + 0)
    v8 = (v7 - 2)
    v9 = (v8 - 1)
    v10 = ((v9 // 1) - (((v9 % 1) != 0) & (((v9 % 1) < 0) != (1 < 0))))
    v11 = (v10 + 1)
    v12 = (v11 - 0)
    v13 = (v12 + 0)
    v14 = ((v13 // 1) - (((v13 % 1) != 0) & (((v13 % 1) < 0) != (1 < 0))))
    v15 = (D4 + 0)
    v16 = (v15 - 2)
    v17 = (v16 - 1)
    v18 = ((v17 // 1) - (((v17 % 1) != 0) & (((v17 % 1) < 0) != (1 < 0))))
    v19 = (v18 + 1)
    v20 = (v19 - 0)
    v21 = (v20 + 0)
    v22 = ((v21 // 1) - (((v21 % 1) != 0) & (((v21 % 1) < 0) != (1 < 0))))
    v23 = ((((((v0 // v22) // v14) // 3) // 3) // v6) % v3)
    v24 = (((((v0 // v22) // v14) // 3) // 3) % v6)
    v25 = ((((v0 // v22) // v14) // 3) % 3)
    v26 = (((v0 // v22) // v14) % 3)
    v27 = ((v0 // v22) % v14)
    v28 = (v0 % v22)
    v29 = (v23 * 1)
    v30 = (0 + v29)
    v31 = (v24 * 1)
    v32 = (0 + v31)
    v33 = (v25 * 1)
    v34 = (0 + v33)
    v35 = (v26 * 1)
    v36 = (0 + v35)
    v37 = (v27 * 1)
    v38 = (0 + v37)
    v39 = (v28 * 1)
    v40 = (0 + v39)
    v41 = (v40 * 1)
    v42 = (v41 - 0)
    v43 = (v36 * 1)
    v44 = (v42 + v43)
    v45 = (v38 * v19)
    v46 = (v45 + v40)
    v47 = (v38 * 1)
    v48 = (v47 - 0)
    v49 = (v34 * 1)
    v50 = (v48 + v49)
    v51 = D2
    v52 = (v30 < v51)
    v53 = (v30 >= 0)
    v54 = (v53 & v52)
    v55 = D1
    v56 = (v32 < v55)
    v57 = (v32 >= 0)
    v58 = (v57 & v56)
    v59 = (v54 & v58)
    v60 = (v32 * 9)
    v61 = (v34 * 3)
    v62 = (v60 + v61)
    v63 = (v62 + v36)
    v64 = D2
    v65 = (v30 < v64)
    v66 = (v30 >= 0)
    v67 = (v66 & v65)
    v68 = D9
    v69 = (v63 < v68)
    v70 = (v63 >= 0)
    v71 = (v70 & v69)
    v72 = (v67 & v71)
    v73 = D10
    v74 = (v46 < v73)
    v75 = (v46 >= 0)
    v76 = (v75 & v74)
    v77 = (v72 & v76)
    v78 = tl.load((x + (v30) * S0_0 + (v32) * S0_1 + (v50) * S0_2 + (v44) * S0_3), mask=v59, other=0.0)
    tl.store((columns + (v30) * S1_0 + (v63) * S1_1 + (v46) * S1_2), v78, mask=v77)

def launch(x, columns):
    D1 = x.shape[1]
    D2 = x.shape[0]
    D3 = x.shape[2]
    D4 = x.shape[3]
    D9 = columns.shape[1]
    D10 = columns.shape[2]
    S0_0 = x.stride(0)
    S0_1 = x.stride(1)
    S0_2 = x.stride(2)
    S0_3 = x.stride(3)
    S1_0 = columns.stride(0)
    S1_1 = columns.stride(1)
    S1_2 = columns.stride(2)
    grid = lambda META: ((((((D2 * D1) * 3) * 3) * (((D3 - 2) - 1) + 1)) * (((D4 - 2) - 1) + 1)),)
    return _intent_kernel[grid](x, columns, D1, D2, D3, D4, D9, D10, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2)

def run(x):
    columns = torch.empty((x.shape[0], columns.shape[1], columns.shape[2]), device=x.device, dtype=torch.float32)
    launch(x, columns)
    return columns
