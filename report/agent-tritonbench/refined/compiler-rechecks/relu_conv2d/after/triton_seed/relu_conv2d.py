import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("x", "weight", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D4": 1}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 1}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 4}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 128}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 32}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(x, weight, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, FRAGMENT_D4: tl.constexpr):
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
    v13 = (FRAGMENT_D4 - 1)
    v14 = (v12 + v13)
    v15 = ((v14 // FRAGMENT_D4) - (((v14 % FRAGMENT_D4) != 0) & (((v14 % FRAGMENT_D4) < 0) != (FRAGMENT_D4 < 0))))
    v16 = ((((v0 // v15) // v9) // v6) % v3)
    v17 = (((v0 // v15) // v9) % v6)
    v18 = ((v0 // v15) % v9)
    v19 = (v0 % v15)
    v20 = (v19 * FRAGMENT_D4)
    v21 = (v16 * 1)
    v22 = (0 + v21)
    v23 = (v17 * 1)
    v24 = (0 + v23)
    v25 = (v18 * 1)
    v26 = (0 + v25)
    v27 = (v20 * 1)
    v28 = (0 + v27)
    v29 = (v28 + 1)
    v30 = (v12 - v20)
    v31 = (v30 * 1)
    v32 = (v28 + v31)
    v33 = (v28 + tl.arange(0, FRAGMENT_D4) * 1)
    v34 = tl.full((FRAGMENT_D4,), v32, tl.int64)
    v35 = (v33 < v34)
    v36 = (D1 - 1)
    v37 = (D4 - 1)
    v38 = D2
    v39 = (v22 < v38)
    v40 = (v22 >= 0)
    v41 = (v40 & v39)
    v42 = D3
    v43 = D1
    v44 = D4
    v45 = D5
    v46 = (v24 < v45)
    v47 = (v24 >= 0)
    v48 = (v47 & v46)
    v49 = D3
    v50 = D6
    v51 = D7
    v52 = tl.full((FRAGMENT_D4,), 0.0, tl.float32)
    v53 = tl.full((32,), v26, tl.int64)
    v54 = (0 + tl.arange(0, 32) * 1)
    v55 = (27 * 1)
    v56 = (0 + v55)
    v57 = tl.full((32,), v56, tl.int64)
    v58 = (v54 < v57)
    v59 = tl.full((32,), 3, tl.int64)
    v60 = ((v54 // v59) - (((v54 % v59) != 0) & (((v54 % v59) < 0) != (v59 < 0))))
    v61 = tl.full((32,), 3, tl.int64)
    v62 = ((v60 % v61) + (((v60 % v61) != 0) & (((v60 % v61) < 0) != (v61 < 0))) * v61)
    v63 = (v53 + v62)
    v64 = tl.full((32,), 1, tl.int64)
    v65 = (v63 - v64)
    v66 = tl.full((32,), 0, tl.int64)
    v67 = (v65 >= v66)
    v68 = tl.full((32,), D1, tl.int64)
    v69 = (v65 < v68)
    v70 = (v67 & v69)
    v71 = tl.broadcast_to(v70[None, :], (FRAGMENT_D4, 32))
    v72 = tl.broadcast_to(v33[:, None], (FRAGMENT_D4, 32))
    v73 = tl.full((32,), 3, tl.int64)
    v74 = ((v54 % v73) + (((v54 % v73) != 0) & (((v54 % v73) < 0) != (v73 < 0))) * v73)
    v75 = tl.broadcast_to(v74[None, :], (FRAGMENT_D4, 32))
    v76 = (v72 + v75)
    v77 = tl.full((FRAGMENT_D4, 32), 1, tl.int64)
    v78 = (v76 - v77)
    v79 = tl.full((FRAGMENT_D4, 32), 0, tl.int64)
    v80 = (v78 >= v79)
    v81 = (v71 & v80)
    v82 = tl.full((FRAGMENT_D4, 32), D4, tl.int64)
    v83 = (v78 < v82)
    v84 = (v81 & v83)
    v85 = tl.full((32,), 9, tl.int64)
    v86 = ((v54 // v85) - (((v54 % v85) != 0) & (((v54 % v85) < 0) != (v85 < 0))))
    v87 = tl.full((32,), 0, tl.int64)
    v88 = tl.maximum(v65, v87, propagate_nan=tl.PropagateNan.ALL)
    v89 = tl.full((32,), v36, tl.int64)
    v90 = tl.minimum(v88, v89, propagate_nan=tl.PropagateNan.ALL)
    v91 = tl.full((FRAGMENT_D4, 32), 0, tl.int64)
    v92 = tl.maximum(v78, v91, propagate_nan=tl.PropagateNan.ALL)
    v93 = tl.full((FRAGMENT_D4, 32), v37, tl.int64)
    v94 = tl.minimum(v92, v93, propagate_nan=tl.PropagateNan.ALL)
    v95 = tl.full((32,), v41, tl.int1)
    v96 = tl.full((32,), 0, tl.int64)
    v97 = (v86 >= v96)
    v98 = tl.full((32,), v42, tl.int64)
    v99 = (v86 < v98)
    v100 = (v97 & v99)
    v101 = (v95 & v100)
    v102 = tl.full((32,), 0, tl.int64)
    v103 = (v90 >= v102)
    v104 = tl.full((32,), v43, tl.int64)
    v105 = (v90 < v104)
    v106 = (v103 & v105)
    v107 = (v101 & v106)
    v108 = tl.broadcast_to(v107[None, :], (FRAGMENT_D4, 32))
    v109 = tl.full((FRAGMENT_D4, 32), 0, tl.int64)
    v110 = (v94 >= v109)
    v111 = tl.full((FRAGMENT_D4, 32), v44, tl.int64)
    v112 = (v94 < v111)
    v113 = (v110 & v112)
    v114 = (v108 & v113)
    v115 = tl.broadcast_to(v35[:, None], (FRAGMENT_D4, 32))
    v116 = (v114 & v115)
    v117 = tl.full((FRAGMENT_D4, 32), 0.0, tl.float32)
    v118 = tl.reshape(v58, (1, 32), can_reorder=False)
    v119 = tl.broadcast_to(v118, (FRAGMENT_D4, 32))
    v120 = (v116 & v119)
    v121 = tl.full((FRAGMENT_D4, 32), 0.0, tl.float32)
    v122 = tl.full((32,), v48, tl.int1)
    v123 = tl.full((32,), 0, tl.int64)
    v124 = (v86 >= v123)
    v125 = tl.full((32,), v49, tl.int64)
    v126 = (v86 < v125)
    v127 = (v124 & v126)
    v128 = (v122 & v127)
    v129 = tl.full((32,), 0, tl.int64)
    v130 = (v62 >= v129)
    v131 = tl.full((32,), v50, tl.int64)
    v132 = (v62 < v131)
    v133 = (v130 & v132)
    v134 = (v128 & v133)
    v135 = tl.full((32,), 0, tl.int64)
    v136 = (v74 >= v135)
    v137 = tl.full((32,), v51, tl.int64)
    v138 = (v74 < v137)
    v139 = (v136 & v138)
    v140 = (v134 & v139)
    v141 = tl.full((32,), 0.0, tl.float32)
    v142 = (v140 & v58)
    v143 = tl.broadcast_to(v52[:, None], (FRAGMENT_D4, 32))
    v144 = tl.load((weight + (tl.full((32,), v24, tl.int64)) * S1_0 + (tl.broadcast_to(v86, (32,))) * S1_1 + (tl.broadcast_to(v62, (32,))) * S1_2 + (tl.broadcast_to(v74, (32,))) * S1_3), mask=v142, other=v141)
    v145 = tl.broadcast_to(v144[None, :], (FRAGMENT_D4, 32))
    v146 = tl.load((x + (tl.full((FRAGMENT_D4, 32), v22, tl.int64)) * S0_0 + (tl.broadcast_to(v86[None, :], (FRAGMENT_D4, 32))) * S0_1 + (tl.broadcast_to(v90[None, :], (FRAGMENT_D4, 32))) * S0_2 + (tl.broadcast_to(v94, (FRAGMENT_D4, 32))) * S0_3), mask=v120, other=v117)
    v147 = tl.where(v84, v146, v121)
    v148 = (v147 * v145)
    v149 = tl.where(v119, v148, v143)
    v150 = tl.sum(v149, axis=1)
    v151 = tl.full((FRAGMENT_D4,), 0.0, tl.float32)
    v152 = tl.maximum(v150, v151, propagate_nan=tl.PropagateNan.ALL)
    v153 = D2
    v154 = (v22 < v153)
    v155 = (v22 >= 0)
    v156 = (v155 & v154)
    v157 = D5
    v158 = (v24 < v157)
    v159 = (v24 >= 0)
    v160 = (v159 & v158)
    v161 = (v156 & v160)
    v162 = D1
    v163 = (v26 < v162)
    v164 = (v26 >= 0)
    v165 = (v164 & v163)
    v166 = (v161 & v165)
    v167 = D4
    v168 = tl.full((FRAGMENT_D4,), v167, tl.int64)
    v169 = (v33 < v168)
    v170 = tl.full((FRAGMENT_D4,), 0, tl.int64)
    v171 = (v33 >= v170)
    v172 = (v171 & v169)
    v173 = tl.full((FRAGMENT_D4,), v166, tl.int1)
    v174 = (v173 & v172)
    v175 = (v174 & v35)
    tl.store((output + (tl.full((FRAGMENT_D4,), v22, tl.int64)) * S2_0 + (tl.full((FRAGMENT_D4,), v24, tl.int64)) * S2_1 + (tl.full((FRAGMENT_D4,), v26, tl.int64)) * S2_2 + (tl.broadcast_to(v33, (FRAGMENT_D4,))) * S2_3), v152, mask=v175)

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
    grid = lambda META: ((((D2 * D5) * D1) * triton.cdiv(D4, META["FRAGMENT_D4"])),)
    return _intent_kernel[grid](x, weight, output, D1, D2, D3, D4, D5, D6, D7, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3)

def run(x, weight):
    output = torch.empty((x.shape[0], weight.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=torch.float32)
    launch(x, weight, output)
    return output
