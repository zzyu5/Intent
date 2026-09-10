import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 256, "FRAGMENT_D2": 1, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256, "FRAGMENT_D2": 1, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256, "FRAGMENT_D2": 1, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256, "FRAGMENT_D2": 1, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1024, "FRAGMENT_D2": 4, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1024, "FRAGMENT_D2": 4, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1024, "FRAGMENT_D2": 4, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1024, "FRAGMENT_D2": 4, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 2, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 2, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 2, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 2, "FRAGMENT_D2": 8, "FRAGMENT_D3": 1, "FRAGMENT_D4": 1}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, FRAGMENT_D4: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D2 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D3 - 0)
    v8 = (v7 + 0)
    v9 = ((v8 // 1) - (((v8 % 1) != 0) & (((v8 % 1) < 0) != (1 < 0))))
    v10 = (D4 - 0)
    v11 = (v10 + 0)
    v12 = ((v11 // 1) - (((v11 % 1) != 0) & (((v11 % 1) < 0) != (1 < 0))))
    v13 = (FRAGMENT_D1 - 1)
    v14 = (v3 + v13)
    v15 = ((v14 // FRAGMENT_D1) - (((v14 % FRAGMENT_D1) != 0) & (((v14 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v16 = (FRAGMENT_D2 - 1)
    v17 = (v6 + v16)
    v18 = ((v17 // FRAGMENT_D2) - (((v17 % FRAGMENT_D2) != 0) & (((v17 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v19 = (FRAGMENT_D3 - 1)
    v20 = (v9 + v19)
    v21 = ((v20 // FRAGMENT_D3) - (((v20 % FRAGMENT_D3) != 0) & (((v20 % FRAGMENT_D3) < 0) != (FRAGMENT_D3 < 0))))
    v22 = (FRAGMENT_D4 - 1)
    v23 = (v12 + v22)
    v24 = ((v23 // FRAGMENT_D4) - (((v23 % FRAGMENT_D4) != 0) & (((v23 % FRAGMENT_D4) < 0) != (FRAGMENT_D4 < 0))))
    v25 = ((((v0 // v24) // v21) // v18) % v15)
    v26 = (((v0 // v24) // v21) % v18)
    v27 = ((v0 // v24) % v21)
    v28 = (v0 % v24)
    v29 = (v25 * FRAGMENT_D1)
    v30 = (v26 * FRAGMENT_D2)
    v31 = (v27 * FRAGMENT_D3)
    v32 = (v28 * FRAGMENT_D4)
    v33 = (v29 * 1)
    v34 = (0 + v33)
    v35 = (v34 + 1)
    v36 = (v3 - v29)
    v37 = (v36 * 1)
    v38 = (v34 + v37)
    v39 = (v34 + tl.arange(0, FRAGMENT_D1) * 1)
    v40 = tl.full((FRAGMENT_D1,), v38, tl.int64)
    v41 = (v39 < v40)
    v42 = (v30 * 1)
    v43 = (0 + v42)
    v44 = (v43 + 1)
    v45 = (v6 - v30)
    v46 = (v45 * 1)
    v47 = (v43 + v46)
    v48 = (v43 + tl.arange(0, FRAGMENT_D2) * 1)
    v49 = tl.full((FRAGMENT_D2,), v47, tl.int64)
    v50 = (v48 < v49)
    v51 = (v31 * 1)
    v52 = (0 + v51)
    v53 = (v52 + 1)
    v54 = (v9 - v31)
    v55 = (v54 * 1)
    v56 = (v52 + v55)
    v57 = (v52 + tl.arange(0, FRAGMENT_D3) * 1)
    v58 = tl.full((FRAGMENT_D3,), v56, tl.int64)
    v59 = (v57 < v58)
    v60 = (v32 * 1)
    v61 = (0 + v60)
    v62 = (v61 + 1)
    v63 = (v12 - v32)
    v64 = (v63 * 1)
    v65 = (v61 + v64)
    v66 = (v61 + tl.arange(0, FRAGMENT_D4) * 1)
    v67 = tl.full((FRAGMENT_D4,), v65, tl.int64)
    v68 = (v66 < v67)
    v69 = D1
    v70 = tl.broadcast_to(v39[None, None, None, :], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v71 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v69, tl.int64)
    v72 = (v70 < v71)
    v73 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v74 = (v70 >= v73)
    v75 = (v74 & v72)
    v76 = D2
    v77 = tl.broadcast_to(v48[None, None, :, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v78 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v76, tl.int64)
    v79 = (v77 < v78)
    v80 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v81 = (v77 >= v80)
    v82 = (v81 & v79)
    v83 = (v75 & v82)
    v84 = D3
    v85 = tl.broadcast_to(v57[None, :, None, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v86 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v84, tl.int64)
    v87 = (v85 < v86)
    v88 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v89 = (v85 >= v88)
    v90 = (v89 & v87)
    v91 = (v83 & v90)
    v92 = D4
    v93 = tl.broadcast_to(v66[:, None, None, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v94 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v92, tl.int64)
    v95 = (v93 < v94)
    v96 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v97 = (v93 >= v96)
    v98 = (v97 & v95)
    v99 = (v91 & v98)
    v100 = D4
    v101 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v100, tl.int64)
    v102 = (v93 < v101)
    v103 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v104 = (v93 >= v103)
    v105 = (v104 & v102)
    v106 = D3
    v107 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v106, tl.int64)
    v108 = (v85 < v107)
    v109 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v110 = (v85 >= v109)
    v111 = (v110 & v108)
    v112 = (v105 & v111)
    v113 = D2
    v114 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v113, tl.int64)
    v115 = (v77 < v114)
    v116 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v117 = (v77 >= v116)
    v118 = (v117 & v115)
    v119 = (v112 & v118)
    v120 = D1
    v121 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), v120, tl.int64)
    v122 = (v70 < v121)
    v123 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0, tl.int64)
    v124 = (v70 >= v123)
    v125 = (v124 & v122)
    v126 = (v119 & v125)
    v127 = tl.broadcast_to(v41[None, None, None, :], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v128 = (v99 & v127)
    v129 = tl.broadcast_to(v50[None, None, :, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v130 = (v128 & v129)
    v131 = tl.broadcast_to(v59[None, :, None, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v132 = (v130 & v131)
    v133 = tl.broadcast_to(v68[:, None, None, None], (FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1))
    v134 = (v132 & v133)
    v135 = tl.full((FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), 0.0, tl.float32)
    v136 = (v126 & v133)
    v137 = (v136 & v131)
    v138 = (v137 & v129)
    v139 = (v138 & v127)
    v140 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v34, tl.int64) * S0_0 + tl.cast(v43, tl.int64) * S0_1 + tl.cast(v52, tl.int64) * S0_2 + tl.cast(v61, tl.int64) * S0_3), shape=((D4 - tl.cast(v61, tl.int64)), (D3 - tl.cast(v52, tl.int64)), (D2 - tl.cast(v43, tl.int64)), (D1 - tl.cast(v34, tl.int64))), strides=(S0_3, S0_2, S0_1, S0_0), offsets=(0, 0, 0, 0), block_shape=(FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), order=(0, 1, 2, 3)), boundary_check=(0, 1, 2, 3), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v61, tl.int64) * S1_0 + tl.cast(v52, tl.int64) * S1_1 + tl.cast(v43, tl.int64) * S1_2 + tl.cast(v34, tl.int64) * S1_3), shape=((D4 - tl.cast(v61, tl.int64)), (D3 - tl.cast(v52, tl.int64)), (D2 - tl.cast(v43, tl.int64)), (D1 - tl.cast(v34, tl.int64))), strides=(S1_0, S1_1, S1_2, S1_3), offsets=(0, 0, 0, 0), block_shape=(FRAGMENT_D4, FRAGMENT_D3, FRAGMENT_D2, FRAGMENT_D1), order=(3, 2, 1, 0)), tl.cast(v140, tl.float32), boundary_check=(0, 1, 2, 3))

def launch(input, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = input.shape[2]
    D4 = input.shape[3]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    S1_2 = output.stride(2)
    S1_3 = output.stride(3)
    grid = lambda META: ((((triton.cdiv(D1, META["FRAGMENT_D1"]) * triton.cdiv(D2, META["FRAGMENT_D2"])) * triton.cdiv(D3, META["FRAGMENT_D3"])) * triton.cdiv(D4, META["FRAGMENT_D4"])),)
    return _intent_kernel[grid](input, output, D1, D2, D3, D4, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3)

def run(input):
    output = torch.empty((input.shape[3], input.shape[2], input.shape[1], input.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
