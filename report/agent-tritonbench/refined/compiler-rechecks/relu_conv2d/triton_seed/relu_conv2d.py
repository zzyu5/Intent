import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("x", "weight", "output", ), (False, False, True, ))

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
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(x, weight, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D2 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D5 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D1 - 0)
    v8 = (v7 + 0)
    v9 = ((v8 // 1) - (((v8 % 1) != 0) & (((v8 % 1) < 0) != (1 < 0))))
    v10 = (D4 - 0)
    v11 = (v10 + 0)
    v12 = ((v11 // 1) - (((v11 % 1) != 0) & (((v11 % 1) < 0) != (1 < 0))))
    v13 = ((((v0 // v12) // v9) // v6) % v3)
    v14 = (((v0 // v12) // v9) % v6)
    v15 = ((v0 // v12) % v9)
    v16 = (v0 % v12)
    v17 = (v13 * 1)
    v18 = (0 + v17)
    v19 = (v14 * 1)
    v20 = (0 + v19)
    v21 = (v15 * 1)
    v22 = (0 + v21)
    v23 = (v16 * 1)
    v24 = (0 + v23)
    v25 = (D1 - 1)
    v26 = (D4 - 1)
    v27 = D2
    v28 = (v18 < v27)
    v29 = (v18 >= 0)
    v30 = (v29 & v28)
    v31 = D3
    v32 = D1
    v33 = D4
    v34 = D5
    v35 = (v20 < v34)
    v36 = (v20 >= 0)
    v37 = (v36 & v35)
    v38 = D3
    v39 = D6
    v40 = D7
    v41 = tl.full((32,), v22, tl.int64)
    v42 = (0 + tl.arange(0, 32) * 1)
    v43 = (27 * 1)
    v44 = (0 + v43)
    v45 = tl.full((32,), v44, tl.int64)
    v46 = (v42 < v45)
    v47 = tl.full((32,), 3, tl.int64)
    v48 = ((v42 // v47) - (((v42 % v47) != 0) & (((v42 % v47) < 0) != (v47 < 0))))
    v49 = tl.full((32,), 3, tl.int64)
    v50 = ((v48 % v49) + (((v48 % v49) != 0) & (((v48 % v49) < 0) != (v49 < 0))) * v49)
    v51 = (v41 + v50)
    v52 = tl.full((32,), 1, tl.int64)
    v53 = (v51 - v52)
    v54 = tl.full((32,), 0, tl.int64)
    v55 = (v53 >= v54)
    v56 = tl.full((32,), D1, tl.int64)
    v57 = (v53 < v56)
    v58 = (v55 & v57)
    v59 = tl.full((32,), v24, tl.int64)
    v60 = tl.full((32,), 3, tl.int64)
    v61 = ((v42 % v60) + (((v42 % v60) != 0) & (((v42 % v60) < 0) != (v60 < 0))) * v60)
    v62 = (v59 + v61)
    v63 = tl.full((32,), 1, tl.int64)
    v64 = (v62 - v63)
    v65 = tl.full((32,), 0, tl.int64)
    v66 = (v64 >= v65)
    v67 = (v58 & v66)
    v68 = tl.full((32,), D4, tl.int64)
    v69 = (v64 < v68)
    v70 = (v67 & v69)
    v71 = tl.full((32,), 9, tl.int64)
    v72 = ((v42 // v71) - (((v42 % v71) != 0) & (((v42 % v71) < 0) != (v71 < 0))))
    v73 = tl.full((32,), 0, tl.int64)
    v74 = tl.maximum(v53, v73, propagate_nan=tl.PropagateNan.ALL)
    v75 = tl.full((32,), v25, tl.int64)
    v76 = tl.minimum(v74, v75, propagate_nan=tl.PropagateNan.ALL)
    v77 = tl.full((32,), 0, tl.int64)
    v78 = tl.maximum(v64, v77, propagate_nan=tl.PropagateNan.ALL)
    v79 = tl.full((32,), v26, tl.int64)
    v80 = tl.minimum(v78, v79, propagate_nan=tl.PropagateNan.ALL)
    v81 = tl.full((32,), v30, tl.int1)
    v82 = tl.full((32,), 0, tl.int64)
    v83 = (v72 >= v82)
    v84 = tl.full((32,), v31, tl.int64)
    v85 = (v72 < v84)
    v86 = (v83 & v85)
    v87 = (v81 & v86)
    v88 = tl.full((32,), 0, tl.int64)
    v89 = (v76 >= v88)
    v90 = tl.full((32,), v32, tl.int64)
    v91 = (v76 < v90)
    v92 = (v89 & v91)
    v93 = (v87 & v92)
    v94 = tl.full((32,), 0, tl.int64)
    v95 = (v80 >= v94)
    v96 = tl.full((32,), v33, tl.int64)
    v97 = (v80 < v96)
    v98 = (v95 & v97)
    v99 = (v93 & v98)
    v100 = tl.full((32,), 0.0, tl.float32)
    v101 = (v99 & v46)
    v102 = tl.full((32,), 0.0, tl.float32)
    v103 = tl.full((32,), v37, tl.int1)
    v104 = tl.full((32,), 0, tl.int64)
    v105 = (v72 >= v104)
    v106 = tl.full((32,), v38, tl.int64)
    v107 = (v72 < v106)
    v108 = (v105 & v107)
    v109 = (v103 & v108)
    v110 = tl.full((32,), 0, tl.int64)
    v111 = (v50 >= v110)
    v112 = tl.full((32,), v39, tl.int64)
    v113 = (v50 < v112)
    v114 = (v111 & v113)
    v115 = (v109 & v114)
    v116 = tl.full((32,), 0, tl.int64)
    v117 = (v61 >= v116)
    v118 = tl.full((32,), v40, tl.int64)
    v119 = (v61 < v118)
    v120 = (v117 & v119)
    v121 = (v115 & v120)
    v122 = tl.full((32,), 0.0, tl.float32)
    v123 = (v121 & v46)
    v124 = tl.full((32,), 0.0, tl.float32)
    v125 = tl.load((weight + (tl.full((32,), v20, tl.int64)) * S1_0 + (tl.broadcast_to(v72, (32,))) * S1_1 + (tl.broadcast_to(v50, (32,))) * S1_2 + (tl.broadcast_to(v61, (32,))) * S1_3), mask=v123, other=v122)
    v126 = tl.load((x + (tl.full((32,), v18, tl.int64)) * S0_0 + (tl.broadcast_to(v72, (32,))) * S0_1 + (tl.broadcast_to(v76, (32,))) * S0_2 + (tl.broadcast_to(v80, (32,))) * S0_3), mask=v101, other=v100)
    v127 = tl.where(v70, v126, v102)
    v128 = (v127 * v125)
    v129 = tl.where(v46, v128, v124)
    v130 = tl.sum(v129, axis=0)
    v131 = tl.maximum(v130, 0.0, propagate_nan=tl.PropagateNan.ALL)
    v132 = D2
    v133 = (v18 < v132)
    v134 = (v18 >= 0)
    v135 = (v134 & v133)
    v136 = D5
    v137 = (v20 < v136)
    v138 = (v20 >= 0)
    v139 = (v138 & v137)
    v140 = (v135 & v139)
    v141 = D1
    v142 = (v22 < v141)
    v143 = (v22 >= 0)
    v144 = (v143 & v142)
    v145 = (v140 & v144)
    v146 = D4
    v147 = (v24 < v146)
    v148 = (v24 >= 0)
    v149 = (v148 & v147)
    v150 = (v145 & v149)
    tl.store((output + (v18) * S2_0 + (v20) * S2_1 + (v22) * S2_2 + (v24) * S2_3), v131, mask=v150)

def launch(x, weight, output):
    D1 = x.shape[2]
    D2 = x.shape[0]
    D3 = x.shape[1]
    D4 = x.shape[3]
    D5 = weight.shape[0]
    D6 = weight.shape[2]
    D7 = weight.shape[3]
    S0_0 = x.stride(0)
    S0_1 = x.stride(1)
    S0_2 = x.stride(2)
    S0_3 = x.stride(3)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S1_2 = weight.stride(2)
    S1_3 = weight.stride(3)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    S2_2 = output.stride(2)
    S2_3 = output.stride(3)
    grid = lambda META: ((((D2 * D5) * D1) * D4),)
    return _intent_kernel[grid](x, weight, output, D1, D2, D3, D4, D5, D6, D7, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3)

def run(x, weight):
    output = torch.empty((x.shape[0], weight.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.float32)
    launch(x, weight, output)
    return output
