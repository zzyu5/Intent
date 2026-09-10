import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "index", "mask", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 256}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, index, mask, output, value, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr, FRAGMENT_D2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = (D2 - 0)
    v4 = (v3 + 0)
    v5 = ((v4 // 1) - (((v4 % 1) != 0) & (((v4 % 1) < 0) != (1 < 0))))
    v6 = (FRAGMENT_D1 - 1)
    v7 = (v2 + v6)
    v8 = ((v7 // FRAGMENT_D1) - (((v7 % FRAGMENT_D1) != 0) & (((v7 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v9 = (FRAGMENT_D2 - 1)
    v10 = (v5 + v9)
    v11 = ((v10 // FRAGMENT_D2) - (((v10 % FRAGMENT_D2) != 0) & (((v10 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v12 = tl.program_id(1)
    v13 = tl.program_id(0)
    v14 = (v12 * v11)
    v15 = (v14 + v13)
    v16 = ((v15 // v11) % v8)
    v17 = (v15 % v11)
    v18 = (v16 * FRAGMENT_D1)
    v19 = (v17 * FRAGMENT_D2)
    v20 = (v18 * 1)
    v21 = (0 + v20)
    v22 = (v21 + 1)
    v23 = (v2 - v18)
    v24 = (v23 * 1)
    v25 = (v21 + v24)
    v26 = (v21 + tl.arange(0, FRAGMENT_D1) * 1)
    v27 = tl.full((FRAGMENT_D1,), v25, tl.int64)
    v28 = (v26 < v27)
    v29 = (v19 * 1)
    v30 = (0 + v29)
    v31 = (v30 + 1)
    v32 = (v5 - v19)
    v33 = (v32 * 1)
    v34 = (v30 + v33)
    v35 = (v30 + tl.arange(0, FRAGMENT_D2) * 1)
    v36 = tl.full((FRAGMENT_D2,), v34, tl.int64)
    v37 = (v35 < v36)
    v38 = D1
    v39 = tl.broadcast_to(v26[None, :], (FRAGMENT_D2, FRAGMENT_D1))
    v40 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v38, tl.int64)
    v41 = (v39 < v40)
    v42 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v43 = (v39 >= v42)
    v44 = (v43 & v41)
    v45 = D2
    v46 = tl.broadcast_to(v35[:, None], (FRAGMENT_D2, FRAGMENT_D1))
    v47 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v45, tl.int64)
    v48 = (v46 < v47)
    v49 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v50 = (v46 >= v49)
    v51 = (v50 & v48)
    v52 = (v44 & v51)
    v53 = D1
    v54 = tl.broadcast_to(v28[None, :], (FRAGMENT_D2, FRAGMENT_D1))
    v55 = (v52 & v54)
    v56 = tl.broadcast_to(v37[:, None], (FRAGMENT_D2, FRAGMENT_D1))
    v57 = (v55 & v56)
    v58 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v59 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v53, tl.int64)
    v60 = tl.load(tl.make_block_ptr(base=(index + tl.cast(v21, tl.int64) * S1_0 + tl.cast(v30, tl.int64) * S1_1), shape=((D2 - tl.cast(v30, tl.int64)), (D1 - tl.cast(v21, tl.int64))), strides=(S1_1, S1_0), offsets=(0, 0), block_shape=(FRAGMENT_D2, FRAGMENT_D1), order=(0, 1)), boundary_check=(0, 1), padding_option="zero")
    v61 = tl.cast(v60, tl.int64)
    v62 = (v61 < v59)
    v63 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v64 = (v61 >= v63)
    v65 = (v64 & v62)
    v66 = D2
    v67 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v66, tl.int64)
    v68 = (v46 < v67)
    v69 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v70 = (v46 >= v69)
    v71 = (v70 & v68)
    v72 = (v65 & v71)
    v73 = D1
    v74 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v73, tl.int64)
    v75 = (v39 < v74)
    v76 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v77 = (v39 >= v76)
    v78 = (v77 & v75)
    v79 = D2
    v80 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v79, tl.int64)
    v81 = (v46 < v80)
    v82 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v83 = (v46 >= v82)
    v84 = (v83 & v81)
    v85 = (v78 & v84)
    v86 = D1
    v87 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v86, tl.int64)
    v88 = (v39 < v87)
    v89 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v90 = (v39 >= v89)
    v91 = (v90 & v88)
    v92 = D2
    v93 = tl.full((FRAGMENT_D2, FRAGMENT_D1), v92, tl.int64)
    v94 = (v46 < v93)
    v95 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v96 = (v46 >= v95)
    v97 = (v96 & v94)
    v98 = (v91 & v97)
    v99 = (v85 & v54)
    v100 = (v99 & v56)
    v101 = tl.full((FRAGMENT_D2, FRAGMENT_D1), False, tl.int1)
    v102 = (v72 & v56)
    v103 = (v102 & v54)
    v104 = (v103 & v56)
    v105 = tl.full((FRAGMENT_D2, FRAGMENT_D1), 0.0, tl.float32)
    v106 = tl.full((FRAGMENT_D2, FRAGMENT_D1), value, tl.float32)
    v107 = (v98 & v54)
    v108 = (v107 & v56)
    v109 = tl.load((input + (tl.broadcast_to(v61, (FRAGMENT_D2, FRAGMENT_D1))) * S0_0 + (tl.broadcast_to(v35[:, None], (FRAGMENT_D2, FRAGMENT_D1))) * S0_1), mask=v104, other=v105)
    v110 = tl.cast(tl.load(tl.make_block_ptr(base=(mask + tl.cast(v21, tl.int64) * S2_0 + tl.cast(v30, tl.int64) * S2_1), shape=((D2 - tl.cast(v30, tl.int64)), (D1 - tl.cast(v21, tl.int64))), strides=(S2_1, S2_0), offsets=(0, 0), block_shape=(FRAGMENT_D2, FRAGMENT_D1), order=(0, 1)), boundary_check=(0, 1), padding_option="zero"), tl.int1)
    v111 = tl.where(v110, v106, v109)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v21, tl.int64) * S4_0 + tl.cast(v30, tl.int64) * S4_1), shape=((D2 - tl.cast(v30, tl.int64)), (D1 - tl.cast(v21, tl.int64))), strides=(S4_1, S4_0), offsets=(0, 0), block_shape=(FRAGMENT_D2, FRAGMENT_D1), order=(0, 1)), tl.cast(v111, tl.float32), boundary_check=(0, 1))

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
    grid = lambda META: (triton.cdiv(D2, META["FRAGMENT_D2"]), triton.cdiv(D1, META["FRAGMENT_D1"]))
    return _intent_kernel[grid](input, index, mask, output, value, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S4_0, S4_1)

def run(input, index, mask, value):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, index, mask, output, value)
    return output
