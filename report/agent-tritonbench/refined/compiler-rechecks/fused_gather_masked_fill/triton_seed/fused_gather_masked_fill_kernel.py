import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "index", "mask", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S9_A0": 4096}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, index, mask, output, value, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr, FRAGMENT_S9_A0: tl.constexpr):
    v0 = triton.cdiv(max(((D1 * D2) - 0), 0), 1)
    v1 = (FRAGMENT_S9_A0 - 1)
    v2 = (v0 + v1)
    v3 = ((v2 // FRAGMENT_S9_A0) - (((v2 % FRAGMENT_S9_A0) != 0) & (((v2 % FRAGMENT_S9_A0) < 0) != (FRAGMENT_S9_A0 < 0))))
    v4 = tl.program_id(0)
    v5 = (v4 % v3)
    v6 = (D1 * D2)
    v7 = (v5 * FRAGMENT_S9_A0)
    v8 = (v7 * 1)
    v9 = (0 + v8)
    v10 = (v9 + tl.arange(0, FRAGMENT_S9_A0) * 1)
    v11 = tl.full((FRAGMENT_S9_A0,), v6, tl.int64)
    v12 = (v10 < v11)
    v13 = tl.full((FRAGMENT_S9_A0,), D2, tl.int64)
    v14 = ((v10 // v13) - (((v10 % v13) != 0) & (((v10 % v13) < 0) != (v13 < 0))))
    v15 = tl.full((FRAGMENT_S9_A0,), D2, tl.int64)
    v16 = ((v10 % v15) + (((v10 % v15) != 0) & (((v10 % v15) < 0) != (v15 < 0))) * v15)
    v17 = D1
    v18 = tl.full((FRAGMENT_S9_A0,), v17, tl.int64)
    v19 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v20 = (v14 < v18)
    v21 = (v14 >= v19)
    v22 = (v21 & v20)
    v23 = D2
    v24 = tl.full((FRAGMENT_S9_A0,), v23, tl.int64)
    v25 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v26 = (v16 < v24)
    v27 = (v16 >= v25)
    v28 = (v27 & v26)
    v29 = (v22 & v28)
    v30 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v31 = D1
    v32 = tl.full((FRAGMENT_S9_A0,), v31, tl.int64)
    v33 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v34 = (v29 & v12)
    v35 = (v34 & v12)
    v36 = tl.load((index + (tl.broadcast_to(v14, (FRAGMENT_S9_A0,))) * S1_0 + (tl.broadcast_to(v16, (FRAGMENT_S9_A0,))) * S1_1), mask=v35, other=v30)
    v37 = tl.cast(v36, tl.int64)
    v38 = (v37 < v32)
    v39 = (v37 >= v33)
    v40 = (v39 & v38)
    v41 = D2
    v42 = tl.full((FRAGMENT_S9_A0,), v41, tl.int64)
    v43 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v44 = (v16 < v42)
    v45 = (v16 >= v43)
    v46 = (v45 & v44)
    v47 = (v40 & v46)
    v48 = tl.full((FRAGMENT_S9_A0,), 0.0, tl.float32)
    v49 = D1
    v50 = tl.full((FRAGMENT_S9_A0,), v49, tl.int64)
    v51 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v52 = (v14 < v50)
    v53 = (v14 >= v51)
    v54 = (v53 & v52)
    v55 = D2
    v56 = tl.full((FRAGMENT_S9_A0,), v55, tl.int64)
    v57 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v58 = (v16 < v56)
    v59 = (v16 >= v57)
    v60 = (v59 & v58)
    v61 = (v54 & v60)
    v62 = tl.full((FRAGMENT_S9_A0,), False, tl.int1)
    v63 = tl.full((FRAGMENT_S9_A0,), value, tl.float32)
    v64 = D1
    v65 = tl.full((FRAGMENT_S9_A0,), v64, tl.int64)
    v66 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v67 = (v14 < v65)
    v68 = (v14 >= v66)
    v69 = (v68 & v67)
    v70 = D2
    v71 = tl.full((FRAGMENT_S9_A0,), v70, tl.int64)
    v72 = tl.full((FRAGMENT_S9_A0,), 0, tl.int64)
    v73 = (v16 < v71)
    v74 = (v16 >= v72)
    v75 = (v74 & v73)
    v76 = (v69 & v75)
    v77 = (v61 & v12)
    v78 = (v77 & v12)
    v79 = (v47 & v12)
    v80 = (v79 & v12)
    v81 = (v76 & v12)
    v82 = (v81 & v12)
    v83 = tl.load((input + (tl.broadcast_to(v37, (FRAGMENT_S9_A0,))) * S0_0 + (tl.broadcast_to(v16, (FRAGMENT_S9_A0,))) * S0_1), mask=v80, other=v48)
    v84 = tl.load((mask + (tl.broadcast_to(v14, (FRAGMENT_S9_A0,))) * S2_0 + (tl.broadcast_to(v16, (FRAGMENT_S9_A0,))) * S2_1), mask=v78, other=v62)
    v85 = tl.where(v84, v63, v83)
    tl.store((output + (tl.broadcast_to(v14, (FRAGMENT_S9_A0,))) * S4_0 + (tl.broadcast_to(v16, (FRAGMENT_S9_A0,))) * S4_1), v85, mask=v82)

def launch(input, index, mask, output, value):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = index.stride(0)
    S1_1 = index.stride(1)
    S2_0 = mask.stride(0)
    S2_1 = mask.stride(1)
    S4_0 = output.stride(0)
    S4_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(triton.cdiv(max(((D1 * D2) - 0), 0), 1), META["FRAGMENT_S9_A0"]),)
    return _intent_kernel[grid](input, index, mask, output, value, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S4_0, S4_1)

def run(input, index, mask, value):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, index, mask, output, value)
    return output
