import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "other", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D9": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D9": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, other, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr, FRAGMENT_D9: tl.constexpr):
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
    v13 = (FRAGMENT_D9 - 1)
    v14 = (v12 + v13)
    v15 = ((v14 // FRAGMENT_D9) - (((v14 % FRAGMENT_D9) != 0) & (((v14 % FRAGMENT_D9) < 0) != (FRAGMENT_D9 < 0))))
    v16 = ((((v0 // v15) // v9) // v6) % v3)
    v17 = (((v0 // v15) // v9) % v6)
    v18 = ((v0 // v15) % v9)
    v19 = (v0 % v15)
    v20 = (v19 * FRAGMENT_D9)
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
    v33 = (v28 + tl.arange(0, FRAGMENT_D9) * 1)
    v34 = tl.full((FRAGMENT_D9,), v32, tl.int64)
    v35 = (v33 < v34)
    v36 = tl.cast(0, tl.float32)
    v37 = tl.full((FRAGMENT_D9,), v36, tl.float32)
    v38 = v37
    for iv39 in range(0, D2, 1):
        v40 = v38
        for iv41 in range(0, D6, 1):
            v42 = v40
            for iv43 in range(0, D7, 1):
                v44 = (v26 + iv41)
                v45 = (v44 - 1)
                v46 = tl.full((FRAGMENT_D9,), iv43, tl.int64)
                v47 = (v33 + v46)
                v48 = tl.full((FRAGMENT_D9,), 1, tl.int64)
                v49 = (v47 - v48)
                v50 = (v45 >= 0)
                v51 = (v45 < D3)
                v52 = tl.full((FRAGMENT_D9,), 0, tl.int64)
                v53 = (v49 >= v52)
                v54 = tl.full((FRAGMENT_D9,), D4, tl.int64)
                v55 = (v49 < v54)
                v56 = D1
                v57 = (v22 < v56)
                v58 = (v22 >= 0)
                v59 = (v58 & v57)
                v60 = D2
                v61 = (iv39 < v60)
                v62 = (iv39 >= 0)
                v63 = (v62 & v61)
                v64 = (v59 & v63)
                v65 = D3
                v66 = (v45 < v65)
                v67 = (v45 >= 0)
                v68 = (v67 & v66)
                v69 = (v64 & v68)
                v70 = D4
                v71 = tl.full((FRAGMENT_D9,), v70, tl.int64)
                v72 = (v49 < v71)
                v73 = tl.full((FRAGMENT_D9,), 0, tl.int64)
                v74 = (v49 >= v73)
                v75 = (v74 & v72)
                v76 = tl.full((FRAGMENT_D9,), v69, tl.int1)
                v77 = (v76 & v75)
                v78 = D5
                v79 = (v24 < v78)
                v80 = (v24 >= 0)
                v81 = (v80 & v79)
                v82 = D2
                v83 = (iv39 < v82)
                v84 = (iv39 >= 0)
                v85 = (v84 & v83)
                v86 = (v81 & v85)
                v87 = D6
                v88 = (iv41 < v87)
                v89 = (iv41 >= 0)
                v90 = (v89 & v88)
                v91 = (v86 & v90)
                v92 = D7
                v93 = (iv43 < v92)
                v94 = (iv43 >= 0)
                v95 = (v94 & v93)
                v96 = (v91 & v95)
                v97 = tl.full((FRAGMENT_D9,), v96, tl.int1)
                v98 = (v55 & v97)
                v99 = (v53 & v98)
                v100 = tl.full((FRAGMENT_D9,), v51, tl.int1)
                v101 = (v100 & v99)
                v102 = tl.full((FRAGMENT_D9,), v50, tl.int1)
                v103 = (v102 & v101)
                v104 = tl.full((FRAGMENT_D9,), 0.0, tl.float32)
                v105 = (v55 & v77)
                v106 = (v53 & v105)
                v107 = (v100 & v106)
                v108 = (v102 & v107)
                v109 = (v108 & v35)
                v110 = tl.full((FRAGMENT_D9,), 0.0, tl.float32)
                v111 = tl.load((input + (tl.full((FRAGMENT_D9,), v22, tl.int64)) * S0_0 + (tl.full((FRAGMENT_D9,), iv39, tl.int64)) * S0_1 + (tl.full((FRAGMENT_D9,), v45, tl.int64)) * S0_2 + (tl.broadcast_to(v49, (FRAGMENT_D9,))) * S0_3), mask=v109, other=v110)
                v112 = tl.load((weight + (tl.full((FRAGMENT_D9,), v24, tl.int64)) * S1_0 + (tl.full((FRAGMENT_D9,), iv39, tl.int64)) * S1_1 + (tl.full((FRAGMENT_D9,), iv41, tl.int64)) * S1_2 + (tl.full((FRAGMENT_D9,), iv43, tl.int64)) * S1_3), mask=v103, other=v104)
                v113 = (v111 * v112)
                v114 = (v42 + v113)
                v115 = tl.where(v55, v114, v42)
                v116 = tl.where(v53, v115, v42)
                v117 = tl.where(v100, v116, v42)
                v118 = tl.where(v102, v117, v42)
                v42 = v118
            v40 = v42
        v38 = v40
    v119 = D1
    v120 = (v22 < v119)
    v121 = (v22 >= 0)
    v122 = (v121 & v120)
    v123 = D5
    v124 = (v24 < v123)
    v125 = (v24 >= 0)
    v126 = (v125 & v124)
    v127 = (v122 & v126)
    v128 = D8
    v129 = (v26 < v128)
    v130 = (v26 >= 0)
    v131 = (v130 & v129)
    v132 = (v127 & v131)
    v133 = D9
    v134 = tl.full((FRAGMENT_D9,), v133, tl.int64)
    v135 = (v33 < v134)
    v136 = tl.full((FRAGMENT_D9,), 0, tl.int64)
    v137 = (v33 >= v136)
    v138 = (v137 & v135)
    v139 = tl.full((FRAGMENT_D9,), v132, tl.int1)
    v140 = (v139 & v138)
    v141 = D1
    v142 = (v22 < v141)
    v143 = (v22 >= 0)
    v144 = (v143 & v142)
    v145 = D5
    v146 = (v24 < v145)
    v147 = (v24 >= 0)
    v148 = (v147 & v146)
    v149 = (v144 & v148)
    v150 = D8
    v151 = (v26 < v150)
    v152 = (v26 >= 0)
    v153 = (v152 & v151)
    v154 = (v149 & v153)
    v155 = D9
    v156 = tl.full((FRAGMENT_D9,), v155, tl.int64)
    v157 = (v33 < v156)
    v158 = tl.full((FRAGMENT_D9,), 0, tl.int64)
    v159 = (v33 >= v158)
    v160 = (v159 & v157)
    v161 = tl.full((FRAGMENT_D9,), v154, tl.int1)
    v162 = (v161 & v160)
    v163 = (v140 & v35)
    v164 = tl.full((FRAGMENT_D9,), 0.0, tl.float32)
    v165 = (v162 & v35)
    v166 = tl.load((other + (tl.full((FRAGMENT_D9,), v22, tl.int64)) * S2_0 + (tl.full((FRAGMENT_D9,), v24, tl.int64)) * S2_1 + (tl.full((FRAGMENT_D9,), v26, tl.int64)) * S2_2 + (tl.broadcast_to(v33, (FRAGMENT_D9,))) * S2_3), mask=v163, other=v164)
    v167 = (v38 + v166)
    tl.store((output + (tl.full((FRAGMENT_D9,), v22, tl.int64)) * S3_0 + (tl.full((FRAGMENT_D9,), v24, tl.int64)) * S3_1 + (tl.full((FRAGMENT_D9,), v26, tl.int64)) * S3_2 + (tl.broadcast_to(v33, (FRAGMENT_D9,))) * S3_3), v167, mask=v165)

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
    grid = lambda META: ((((D1 * D5) * D8) * triton.cdiv(D9, META["FRAGMENT_D9"])),)
    return _intent_kernel[grid](input, weight, other, output, D1, D2, D3, D4, D5, D6, D7, D8, D9, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, other):
    output = torch.empty((input.shape[0], weight.shape[0], other.shape[2], other.shape[3]), device=input.device, dtype=torch.float32)
    launch(input, weight, other, output)
    return output
