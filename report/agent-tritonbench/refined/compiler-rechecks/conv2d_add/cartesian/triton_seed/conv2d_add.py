import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "other", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D8": 1, "FRAGMENT_D9": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 1, "FRAGMENT_D9": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 1, "FRAGMENT_D9": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 1, "FRAGMENT_D9": 256}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 64, "FRAGMENT_D9": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 64, "FRAGMENT_D9": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 64, "FRAGMENT_D9": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 64, "FRAGMENT_D9": 64}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 4, "FRAGMENT_D9": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 4, "FRAGMENT_D9": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 4, "FRAGMENT_D9": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 4, "FRAGMENT_D9": 1024}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 16, "FRAGMENT_D9": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 16, "FRAGMENT_D9": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 16, "FRAGMENT_D9": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 16, "FRAGMENT_D9": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 8}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 2}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 2}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 2}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D8": 8, "FRAGMENT_D9": 2}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, other, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr, FRAGMENT_D9: tl.constexpr, FRAGMENT_D8: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D5 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D8 - 0)
    v8 = (v7 + 0)
    v9 = ((v8 // 1) - (((v8 % 1) != 0) & (((v8 % 1) < 0) != (1 < 0))))
    v10 = (D9 - 0)
    v11 = (v10 + 0)
    v12 = ((v11 // 1) - (((v11 % 1) != 0) & (((v11 % 1) < 0) != (1 < 0))))
    v13 = (FRAGMENT_D8 - 1)
    v14 = (v9 + v13)
    v15 = ((v14 // FRAGMENT_D8) - (((v14 % FRAGMENT_D8) != 0) & (((v14 % FRAGMENT_D8) < 0) != (FRAGMENT_D8 < 0))))
    v16 = (FRAGMENT_D9 - 1)
    v17 = (v12 + v16)
    v18 = ((v17 // FRAGMENT_D9) - (((v17 % FRAGMENT_D9) != 0) & (((v17 % FRAGMENT_D9) < 0) != (FRAGMENT_D9 < 0))))
    v19 = ((((v0 // v18) // v15) // v6) % v3)
    v20 = (((v0 // v18) // v15) % v6)
    v21 = ((v0 // v18) % v15)
    v22 = (v0 % v18)
    v23 = (v21 * FRAGMENT_D8)
    v24 = (v22 * FRAGMENT_D9)
    v25 = (v19 * 1)
    v26 = (0 + v25)
    v27 = (v20 * 1)
    v28 = (0 + v27)
    v29 = (v23 * 1)
    v30 = (0 + v29)
    v31 = (v30 + 1)
    v32 = (v9 - v23)
    v33 = (v32 * 1)
    v34 = (v30 + v33)
    v35 = (v30 + tl.arange(0, FRAGMENT_D8) * 1)
    v36 = tl.full((FRAGMENT_D8,), v34, tl.int64)
    v37 = (v35 < v36)
    v38 = (v24 * 1)
    v39 = (0 + v38)
    v40 = (v39 + 1)
    v41 = (v12 - v24)
    v42 = (v41 * 1)
    v43 = (v39 + v42)
    v44 = (v39 + tl.arange(0, FRAGMENT_D9) * 1)
    v45 = tl.full((FRAGMENT_D9,), v43, tl.int64)
    v46 = (v44 < v45)
    v47 = tl.cast(0, tl.float32)
    v48 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v47, tl.float32)
    v49 = v48
    for iv50 in range(0, D2, 1):
        v51 = v49
        for iv52 in range(0, D6, 1):
            v53 = v51
            for iv54 in range(0, D7, 1):
                v55 = tl.broadcast_to(v35[None, :], (FRAGMENT_D9, FRAGMENT_D8))
                v56 = tl.full((FRAGMENT_D9, FRAGMENT_D8), iv52, tl.int64)
                v57 = (v55 + v56)
                v58 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 1, tl.int64)
                v59 = (v57 - v58)
                v60 = tl.broadcast_to(v44[:, None], (FRAGMENT_D9, FRAGMENT_D8))
                v61 = tl.full((FRAGMENT_D9, FRAGMENT_D8), iv54, tl.int64)
                v62 = (v60 + v61)
                v63 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 1, tl.int64)
                v64 = (v62 - v63)
                v65 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
                v66 = (v59 >= v65)
                v67 = tl.full((FRAGMENT_D9, FRAGMENT_D8), D3, tl.int64)
                v68 = (v59 < v67)
                v69 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
                v70 = (v64 >= v69)
                v71 = tl.full((FRAGMENT_D9, FRAGMENT_D8), D4, tl.int64)
                v72 = (v64 < v71)
                v73 = D1
                v74 = (v26 < v73)
                v75 = (v26 >= 0)
                v76 = (v75 & v74)
                v77 = D2
                v78 = (iv50 < v77)
                v79 = (iv50 >= 0)
                v80 = (v79 & v78)
                v81 = (v76 & v80)
                v82 = D3
                v83 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v82, tl.int64)
                v84 = (v59 < v83)
                v85 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
                v86 = (v59 >= v85)
                v87 = (v86 & v84)
                v88 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v81, tl.int1)
                v89 = (v88 & v87)
                v90 = D4
                v91 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v90, tl.int64)
                v92 = (v64 < v91)
                v93 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
                v94 = (v64 >= v93)
                v95 = (v94 & v92)
                v96 = (v89 & v95)
                v97 = D5
                v98 = (v28 < v97)
                v99 = (v28 >= 0)
                v100 = (v99 & v98)
                v101 = D2
                v102 = (iv50 < v101)
                v103 = (iv50 >= 0)
                v104 = (v103 & v102)
                v105 = (v100 & v104)
                v106 = D6
                v107 = (iv52 < v106)
                v108 = (iv52 >= 0)
                v109 = (v108 & v107)
                v110 = (v105 & v109)
                v111 = D7
                v112 = (iv54 < v111)
                v113 = (iv54 >= 0)
                v114 = (v113 & v112)
                v115 = (v110 & v114)
                v116 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v115, tl.int1)
                v117 = (v72 & v116)
                v118 = (v70 & v117)
                v119 = (v68 & v118)
                v120 = (v66 & v119)
                v121 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0.0, tl.float32)
                v122 = (v72 & v96)
                v123 = (v70 & v122)
                v124 = (v68 & v123)
                v125 = (v66 & v124)
                v126 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0.0, tl.float32)
                v127 = tl.load((input + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v26, tl.int64)) * S0_0 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), iv50, tl.int64)) * S0_1 + (tl.broadcast_to(v59, (FRAGMENT_D9, FRAGMENT_D8))) * S0_2 + (tl.broadcast_to(v64, (FRAGMENT_D9, FRAGMENT_D8))) * S0_3), mask=v125, other=v126)
                v128 = tl.load((weight + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v28, tl.int64)) * S1_0 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), iv50, tl.int64)) * S1_1 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), iv52, tl.int64)) * S1_2 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), iv54, tl.int64)) * S1_3), mask=v120, other=v121)
                v129 = (v127 * v128)
                v130 = (v53 + v129)
                v131 = tl.where(v72, v130, v53)
                v132 = tl.where(v70, v131, v53)
                v133 = tl.where(v68, v132, v53)
                v134 = tl.where(v66, v133, v53)
                v53 = v134
            v51 = v53
        v49 = v51
    v135 = D1
    v136 = (v26 < v135)
    v137 = (v26 >= 0)
    v138 = (v137 & v136)
    v139 = D5
    v140 = (v28 < v139)
    v141 = (v28 >= 0)
    v142 = (v141 & v140)
    v143 = (v138 & v142)
    v144 = D8
    v145 = tl.broadcast_to(v35[None, :], (FRAGMENT_D9, FRAGMENT_D8))
    v146 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v144, tl.int64)
    v147 = (v145 < v146)
    v148 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
    v149 = (v145 >= v148)
    v150 = (v149 & v147)
    v151 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v143, tl.int1)
    v152 = (v151 & v150)
    v153 = D9
    v154 = tl.broadcast_to(v44[:, None], (FRAGMENT_D9, FRAGMENT_D8))
    v155 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v153, tl.int64)
    v156 = (v154 < v155)
    v157 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
    v158 = (v154 >= v157)
    v159 = (v158 & v156)
    v160 = (v152 & v159)
    v161 = D1
    v162 = (v26 < v161)
    v163 = (v26 >= 0)
    v164 = (v163 & v162)
    v165 = D5
    v166 = (v28 < v165)
    v167 = (v28 >= 0)
    v168 = (v167 & v166)
    v169 = (v164 & v168)
    v170 = D8
    v171 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v170, tl.int64)
    v172 = (v145 < v171)
    v173 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
    v174 = (v145 >= v173)
    v175 = (v174 & v172)
    v176 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v169, tl.int1)
    v177 = (v176 & v175)
    v178 = D9
    v179 = tl.full((FRAGMENT_D9, FRAGMENT_D8), v178, tl.int64)
    v180 = (v154 < v179)
    v181 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0, tl.int64)
    v182 = (v154 >= v181)
    v183 = (v182 & v180)
    v184 = (v177 & v183)
    v185 = tl.broadcast_to(v37[None, :], (FRAGMENT_D9, FRAGMENT_D8))
    v186 = (v160 & v185)
    v187 = tl.broadcast_to(v46[:, None], (FRAGMENT_D9, FRAGMENT_D8))
    v188 = (v186 & v187)
    v189 = tl.full((FRAGMENT_D9, FRAGMENT_D8), 0.0, tl.float32)
    v190 = (v184 & v185)
    v191 = (v190 & v187)
    v192 = tl.load((other + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v26, tl.int64)) * S2_0 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v28, tl.int64)) * S2_1 + (tl.broadcast_to(v35[None, :], (FRAGMENT_D9, FRAGMENT_D8))) * S2_2 + (tl.broadcast_to(v44[:, None], (FRAGMENT_D9, FRAGMENT_D8))) * S2_3), mask=v188, other=v189)
    v193 = (v49 + v192)
    tl.store((output + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v26, tl.int64)) * S3_0 + (tl.full((FRAGMENT_D9, FRAGMENT_D8), v28, tl.int64)) * S3_1 + (tl.broadcast_to(v35[None, :], (FRAGMENT_D9, FRAGMENT_D8))) * S3_2 + (tl.broadcast_to(v44[:, None], (FRAGMENT_D9, FRAGMENT_D8))) * S3_3), v193, mask=v191)

def launch(input, weight, other, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = input.shape[2]
    D4 = input.shape[3]
    D5 = weight.shape[0]
    D6 = weight.shape[2]
    D7 = weight.shape[3]
    D8 = other.shape[2]
    D9 = other.shape[3]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S1_2 = weight.stride(2)
    S1_3 = weight.stride(3)
    S2_0 = other.stride(0)
    S2_1 = other.stride(1)
    S2_2 = other.stride(2)
    S2_3 = other.stride(3)
    S3_0 = output.stride(0)
    S3_1 = output.stride(1)
    S3_2 = output.stride(2)
    S3_3 = output.stride(3)
    grid = lambda META: ((((D1 * D5) * triton.cdiv(D8, META["FRAGMENT_D8"])) * triton.cdiv(D9, META["FRAGMENT_D9"])),)
    return _intent_kernel[grid](input, weight, other, output, D1, D2, D3, D4, D5, D6, D7, D8, D9, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, other):
    output = torch.empty((input.shape[0], weight.shape[0], other.shape[2], other.shape[3]), device=input.device, dtype=torch.float32)
    launch(input, weight, other, output)
    return output
