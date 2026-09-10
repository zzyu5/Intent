import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "grid", "output", ), (False, False, True, ))

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
def _intent_kernel(input, grid, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D2 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D5 - 0)
    v8 = (v7 + 0)
    v9 = ((v8 // 1) - (((v8 % 1) != 0) & (((v8 % 1) < 0) != (1 < 0))))
    v10 = (D6 - 0)
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
    v25 = D1
    v26 = (v18 < v25)
    v27 = (v18 >= 0)
    v28 = (v27 & v26)
    v29 = D5
    v30 = (v22 < v29)
    v31 = (v22 >= 0)
    v32 = (v31 & v30)
    v33 = (v28 & v32)
    v34 = D6
    v35 = (v24 < v34)
    v36 = (v24 >= 0)
    v37 = (v36 & v35)
    v38 = (v33 & v37)
    v39 = (v18 >= 0)
    v40 = (v39 & v26)
    v41 = (v22 >= 0)
    v42 = (v41 & v30)
    v43 = (v40 & v42)
    v44 = (v24 >= 0)
    v45 = (v44 & v35)
    v46 = (v43 & v45)
    v47 = tl.load((grid + (v18) * S1_0 + (v22) * S1_1 + (v24) * S1_2 + (0) * S1_3), mask=v38, other=0.0)
    v48 = (v47 != v47)
    if v48:
        v49 = -1.0
    else:
        v49 = v47
    v50 = tl.load((grid + (v18) * S1_0 + (v22) * S1_1 + (v24) * S1_2 + (1) * S1_3), mask=v46, other=0.0)
    v51 = (v50 != v50)
    if v51:
        v52 = -1.0
    else:
        v52 = v50
    v53 = (v49 + 1.0)
    v54 = tl.cast(D4, tl.float32)
    v55 = (v53 * v54)
    v56 = (v55 - 1.0)
    v57 = tl.fdiv(tl.cast(v56, tl.float32), tl.cast(2.0, tl.float32), ieee_rounding=True)
    v58 = (v52 + 1.0)
    v59 = tl.cast(D3, tl.float32)
    v60 = (v58 * v59)
    v61 = (v60 - 1.0)
    v62 = tl.fdiv(tl.cast(v61, tl.float32), tl.cast(2.0, tl.float32), ieee_rounding=True)
    v63 = tl.cast(v57, tl.int64)
    v64 = tl.cast(v62, tl.int64)
    v65 = tl.cast(v63, tl.float32)
    v66 = (v65 > v57)
    if v66:
        v68 = (v63 - 1)
        v67 = v68
    else:
        v67 = v63
    v69 = tl.cast(v64, tl.float32)
    v70 = (v69 > v62)
    if v70:
        v72 = (v64 - 1)
        v71 = v72
    else:
        v71 = v64
    v73 = (v67 + 1)
    v74 = (v71 + 1)
    v75 = (v71 >= 0)
    if v75:
        v78 = (v71 < D3)
        if v78:
            v81 = (v67 >= 0)
            if v81:
                v83 = (v67 < D4)
                if v83:
                    v85 = D1
                    v86 = (v18 < v85)
                    v87 = (v18 >= 0)
                    v88 = (v87 & v86)
                    v89 = D2
                    v90 = (v20 < v89)
                    v91 = (v20 >= 0)
                    v92 = (v91 & v90)
                    v93 = (v88 & v92)
                    v94 = D3
                    v95 = (v71 < v94)
                    v96 = (v71 >= 0)
                    v97 = (v96 & v95)
                    v98 = (v93 & v97)
                    v99 = D4
                    v100 = (v67 < v99)
                    v101 = (v67 >= 0)
                    v102 = (v101 & v100)
                    v103 = (v98 & v102)
                    v104 = tl.load((input + (v18) * S0_0 + (v20) * S0_1 + (v71) * S0_2 + (v67) * S0_3), mask=v103, other=0.0)
                    v84 = v104
                else:
                    v84 = 0.0
                v82 = v84
            else:
                v82 = 0.0
            v105 = (v73 >= 0)
            if v105:
                v107 = (v73 < D4)
                if v107:
                    v109 = D1
                    v110 = (v18 < v109)
                    v111 = (v18 >= 0)
                    v112 = (v111 & v110)
                    v113 = D2
                    v114 = (v20 < v113)
                    v115 = (v20 >= 0)
                    v116 = (v115 & v114)
                    v117 = (v112 & v116)
                    v118 = D3
                    v119 = (v71 < v118)
                    v120 = (v71 >= 0)
                    v121 = (v120 & v119)
                    v122 = (v117 & v121)
                    v123 = D4
                    v124 = (v73 < v123)
                    v125 = (v73 >= 0)
                    v126 = (v125 & v124)
                    v127 = (v122 & v126)
                    v128 = tl.load((input + (v18) * S0_0 + (v20) * S0_1 + (v71) * S0_2 + (v73) * S0_3), mask=v127, other=0.0)
                    v108 = v128
                else:
                    v108 = 0.0
                v106 = v108
            else:
                v106 = 0.0
            v79 = v82
            v80 = v106
        else:
            v79 = 0.0
            v80 = 0.0
        v76 = v79
        v77 = v80
    else:
        v76 = 0.0
        v77 = 0.0
    v129 = (v74 >= 0)
    if v129:
        v132 = (v74 < D3)
        if v132:
            v135 = (v67 >= 0)
            if v135:
                v137 = (v67 < D4)
                if v137:
                    v139 = D1
                    v140 = (v18 < v139)
                    v141 = (v18 >= 0)
                    v142 = (v141 & v140)
                    v143 = D2
                    v144 = (v20 < v143)
                    v145 = (v20 >= 0)
                    v146 = (v145 & v144)
                    v147 = (v142 & v146)
                    v148 = D3
                    v149 = (v74 < v148)
                    v150 = (v74 >= 0)
                    v151 = (v150 & v149)
                    v152 = (v147 & v151)
                    v153 = D4
                    v154 = (v67 < v153)
                    v155 = (v67 >= 0)
                    v156 = (v155 & v154)
                    v157 = (v152 & v156)
                    v158 = tl.load((input + (v18) * S0_0 + (v20) * S0_1 + (v74) * S0_2 + (v67) * S0_3), mask=v157, other=0.0)
                    v138 = v158
                else:
                    v138 = 0.0
                v136 = v138
            else:
                v136 = 0.0
            v159 = (v73 >= 0)
            if v159:
                v161 = (v73 < D4)
                if v161:
                    v163 = D1
                    v164 = (v18 < v163)
                    v165 = (v18 >= 0)
                    v166 = (v165 & v164)
                    v167 = D2
                    v168 = (v20 < v167)
                    v169 = (v20 >= 0)
                    v170 = (v169 & v168)
                    v171 = (v166 & v170)
                    v172 = D3
                    v173 = (v74 < v172)
                    v174 = (v74 >= 0)
                    v175 = (v174 & v173)
                    v176 = (v171 & v175)
                    v177 = D4
                    v178 = (v73 < v177)
                    v179 = (v73 >= 0)
                    v180 = (v179 & v178)
                    v181 = (v176 & v180)
                    v182 = tl.load((input + (v18) * S0_0 + (v20) * S0_1 + (v74) * S0_2 + (v73) * S0_3), mask=v181, other=0.0)
                    v162 = v182
                else:
                    v162 = 0.0
                v160 = v162
            else:
                v160 = 0.0
            v133 = v136
            v134 = v160
        else:
            v133 = 0.0
            v134 = 0.0
        v130 = v133
        v131 = v134
    else:
        v130 = 0.0
        v131 = 0.0
    v183 = tl.cast(v67, tl.float32)
    v184 = (v57 - v183)
    v185 = tl.cast(v71, tl.float32)
    v186 = (v62 - v185)
    v187 = (1.0 - v184)
    v188 = (1.0 - v186)
    v189 = (v76 * v187)
    v190 = (v189 * v188)
    v191 = (v77 * v184)
    v192 = (v191 * v188)
    v193 = (v190 + v192)
    v194 = (v130 * v187)
    v195 = (v194 * v186)
    v196 = (v193 + v195)
    v197 = (v131 * v184)
    v198 = (v197 * v186)
    v199 = (v196 + v198)
    v200 = D1
    v201 = (v18 < v200)
    v202 = (v18 >= 0)
    v203 = (v202 & v201)
    v204 = D2
    v205 = (v20 < v204)
    v206 = (v20 >= 0)
    v207 = (v206 & v205)
    v208 = (v203 & v207)
    v209 = D5
    v210 = (v22 < v209)
    v211 = (v22 >= 0)
    v212 = (v211 & v210)
    v213 = (v208 & v212)
    v214 = D6
    v215 = (v24 < v214)
    v216 = (v24 >= 0)
    v217 = (v216 & v215)
    v218 = (v213 & v217)
    tl.store((output + (v18) * S2_0 + (v20) * S2_1 + (v22) * S2_2 + (v24) * S2_3), v199, mask=v218)

def launch(input, grid, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = input.shape[2]
    D4 = input.shape[3]
    D5 = grid.shape[1]
    D6 = grid.shape[2]
    D7 = grid.shape[3]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = grid.stride(0)
    S1_1 = grid.stride(1)
    S1_2 = grid.stride(2)
    S1_3 = grid.stride(3)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    S2_2 = output.stride(2)
    S2_3 = output.stride(3)
    grid = lambda META: ((((D1 * D2) * D5) * D6),)
    return _intent_kernel[grid](input, grid, output, D1, D2, D3, D4, D5, D6, D7, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3)

def run(input, grid):
    output = torch.empty((input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]), device=input.device, dtype=torch.float32)
    launch(input, grid, output)
    return output
