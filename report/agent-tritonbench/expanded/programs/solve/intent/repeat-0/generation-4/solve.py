import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("matrix", "rhs", "work", "rhs_work", "result", ), (False, False, True, True, True, ))

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
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S3_0", "S3_1", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(matrix, rhs, work, rhs_work, result, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        for iv3 in range(0, D1, 1):
            v4 = D1
            v5 = (iv2 < v4)
            v6 = (iv2 >= 0)
            v7 = (v6 & v5)
            v8 = D1
            v9 = (iv3 < v8)
            v10 = (iv3 >= 0)
            v11 = (v10 & v9)
            v12 = (v7 & v11)
            v13 = D1
            v14 = (iv2 < v13)
            v15 = (iv2 >= 0)
            v16 = (v15 & v14)
            v17 = D1
            v18 = (iv3 < v17)
            v19 = (iv3 >= 0)
            v20 = (v19 & v18)
            v21 = (v16 & v20)
            v22 = tl.load((matrix + (iv2) * S0_0 + (iv3) * S0_1), mask=v12, other=0.0)
            tl.store((work + (iv2) * S2_0 + (iv3) * S2_1), v22, mask=v21)
        for iv23 in range(0, D2, 1):
            v24 = D1
            v25 = (iv2 < v24)
            v26 = (iv2 >= 0)
            v27 = (v26 & v25)
            v28 = D2
            v29 = (iv23 < v28)
            v30 = (iv23 >= 0)
            v31 = (v30 & v29)
            v32 = (v27 & v31)
            v33 = D1
            v34 = (iv2 < v33)
            v35 = (iv2 >= 0)
            v36 = (v35 & v34)
            v37 = D2
            v38 = (iv23 < v37)
            v39 = (iv23 >= 0)
            v40 = (v39 & v38)
            v41 = (v36 & v40)
            v42 = tl.load((rhs + (iv2) * S1_0 + (iv23) * S1_1), mask=v32, other=0.0)
            tl.store((rhs_work + (iv2) * S3_0 + (iv23) * S3_1), v42, mask=v41)
    for iv43 in range(0, D1, 1):
        v44 = 0.0
        v45 = iv43
        for iv46 in range(0, D1, 1):
            v47 = (iv46 >= iv43)
            if v47:
                v50 = D1
                v51 = (iv46 < v50)
                v52 = (iv46 >= 0)
                v53 = (v52 & v51)
                v54 = D1
                v55 = (iv43 < v54)
                v56 = (iv43 >= 0)
                v57 = (v56 & v55)
                v58 = (v53 & v57)
                v59 = tl.load((work + (iv46) * S2_0 + (iv43) * S2_1), mask=v58, other=0.0)
                v60 = (-v59)
                v61 = tl.maximum(v59, v60, propagate_nan=tl.PropagateNan.ALL)
                v62 = (v61 > v44)
                if v62:
                    v63 = v61
                    v64 = iv46
                else:
                    v63 = v44
                    v64 = v45
                v48 = v63
                v49 = v64
            else:
                v48 = v44
                v49 = v45
            v44 = v48
            v45 = v49
        v65 = (v45 != iv43)
        if v65:
            for iv66 in range(0, D1, 1):
                v67 = D1
                v68 = (iv43 < v67)
                v69 = (iv43 >= 0)
                v70 = (v69 & v68)
                v71 = D1
                v72 = (iv66 < v71)
                v73 = (iv66 >= 0)
                v74 = (v73 & v72)
                v75 = (v70 & v74)
                v76 = tl.load((work + (iv43) * S2_0 + (iv66) * S2_1), mask=v75, other=0.0)
                v77 = (v45 < v67)
                v78 = (v45 >= 0)
                v79 = (v78 & v77)
                v80 = (iv66 >= 0)
                v81 = (v80 & v72)
                v82 = (v79 & v81)
                v83 = tl.load((work + (v45) * S2_0 + (iv66) * S2_1), mask=v82, other=0.0)
                v84 = (iv43 >= 0)
                v85 = (v84 & v68)
                v86 = (iv66 >= 0)
                v87 = (v86 & v72)
                v88 = (v85 & v87)
                tl.store((work + (iv43) * S2_0 + (iv66) * S2_1), v83, mask=v88)
                v89 = (v45 >= 0)
                v90 = (v89 & v77)
                v91 = (iv66 >= 0)
                v92 = (v91 & v72)
                v93 = (v90 & v92)
                tl.store((work + (v45) * S2_0 + (iv66) * S2_1), v76, mask=v93)
            for iv94 in range(0, D2, 1):
                v95 = D1
                v96 = (iv43 < v95)
                v97 = (iv43 >= 0)
                v98 = (v97 & v96)
                v99 = D2
                v100 = (iv94 < v99)
                v101 = (iv94 >= 0)
                v102 = (v101 & v100)
                v103 = (v98 & v102)
                v104 = tl.load((rhs_work + (iv43) * S3_0 + (iv94) * S3_1), mask=v103, other=0.0)
                v105 = (v45 < v95)
                v106 = (v45 >= 0)
                v107 = (v106 & v105)
                v108 = (iv94 >= 0)
                v109 = (v108 & v100)
                v110 = (v107 & v109)
                v111 = tl.load((rhs_work + (v45) * S3_0 + (iv94) * S3_1), mask=v110, other=0.0)
                v112 = (iv43 >= 0)
                v113 = (v112 & v96)
                v114 = (iv94 >= 0)
                v115 = (v114 & v100)
                v116 = (v113 & v115)
                tl.store((rhs_work + (iv43) * S3_0 + (iv94) * S3_1), v111, mask=v116)
                v117 = (v45 >= 0)
                v118 = (v117 & v105)
                v119 = (iv94 >= 0)
                v120 = (v119 & v100)
                v121 = (v118 & v120)
                tl.store((rhs_work + (v45) * S3_0 + (iv94) * S3_1), v104, mask=v121)
        else:
        v122 = D1
        v123 = (iv43 < v122)
        v124 = (iv43 >= 0)
        v125 = (v124 & v123)
        v126 = D1
        v127 = (iv43 < v126)
        v128 = (iv43 >= 0)
        v129 = (v128 & v127)
        v130 = (v125 & v129)
        v131 = tl.load((work + (iv43) * S2_0 + (iv43) * S2_1), mask=v130, other=0.0)
        for iv132 in range(0, D1, 1):
            v133 = D1
            v134 = (iv43 < v133)
            v135 = (iv43 >= 0)
            v136 = (v135 & v134)
            v137 = D1
            v138 = (iv132 < v137)
            v139 = (iv132 >= 0)
            v140 = (v139 & v138)
            v141 = (v136 & v140)
            v142 = tl.load((work + (iv43) * S2_0 + (iv132) * S2_1), mask=v141, other=0.0)
            v143 = tl.fdiv(tl.cast(v142, tl.float32), tl.cast(v131, tl.float32), ieee_rounding=True)
            v144 = (iv43 >= 0)
            v145 = (v144 & v134)
            v146 = (iv132 >= 0)
            v147 = (v146 & v138)
            v148 = (v145 & v147)
            tl.store((work + (iv43) * S2_0 + (iv132) * S2_1), v143, mask=v148)
        for iv149 in range(0, D2, 1):
            v150 = D1
            v151 = (iv43 < v150)
            v152 = (iv43 >= 0)
            v153 = (v152 & v151)
            v154 = D2
            v155 = (iv149 < v154)
            v156 = (iv149 >= 0)
            v157 = (v156 & v155)
            v158 = (v153 & v157)
            v159 = tl.load((rhs_work + (iv43) * S3_0 + (iv149) * S3_1), mask=v158, other=0.0)
            v160 = tl.fdiv(tl.cast(v159, tl.float32), tl.cast(v131, tl.float32), ieee_rounding=True)
            v161 = (iv43 >= 0)
            v162 = (v161 & v151)
            v163 = (iv149 >= 0)
            v164 = (v163 & v155)
            v165 = (v162 & v164)
            tl.store((rhs_work + (iv43) * S3_0 + (iv149) * S3_1), v160, mask=v165)
        for iv166 in range(0, D1, 1):
            v167 = (iv166 != iv43)
            if v167:
                v168 = D1
                v169 = (iv166 < v168)
                v170 = (iv166 >= 0)
                v171 = (v170 & v169)
                v172 = D1
                v173 = (iv43 < v172)
                v174 = (iv43 >= 0)
                v175 = (v174 & v173)
                v176 = (v171 & v175)
                v177 = tl.load((work + (iv166) * S2_0 + (iv43) * S2_1), mask=v176, other=0.0)
                for iv178 in range(0, D1, 1):
                    v179 = D1
                    v180 = (iv166 < v179)
                    v181 = (iv166 >= 0)
                    v182 = (v181 & v180)
                    v183 = D1
                    v184 = (iv178 < v183)
                    v185 = (iv178 >= 0)
                    v186 = (v185 & v184)
                    v187 = (v182 & v186)
                    v188 = tl.load((work + (iv166) * S2_0 + (iv178) * S2_1), mask=v187, other=0.0)
                    v189 = (iv43 < v179)
                    v190 = (iv43 >= 0)
                    v191 = (v190 & v189)
                    v192 = (iv178 >= 0)
                    v193 = (v192 & v184)
                    v194 = (v191 & v193)
                    v195 = tl.load((work + (iv43) * S2_0 + (iv178) * S2_1), mask=v194, other=0.0)
                    v196 = (v177 * v195)
                    v197 = (v188 - v196)
                    v198 = (iv166 >= 0)
                    v199 = (v198 & v180)
                    v200 = (iv178 >= 0)
                    v201 = (v200 & v184)
                    v202 = (v199 & v201)
                    tl.store((work + (iv166) * S2_0 + (iv178) * S2_1), v197, mask=v202)
                for iv203 in range(0, D2, 1):
                    v204 = D1
                    v205 = (iv166 < v204)
                    v206 = (iv166 >= 0)
                    v207 = (v206 & v205)
                    v208 = D2
                    v209 = (iv203 < v208)
                    v210 = (iv203 >= 0)
                    v211 = (v210 & v209)
                    v212 = (v207 & v211)
                    v213 = tl.load((rhs_work + (iv166) * S3_0 + (iv203) * S3_1), mask=v212, other=0.0)
                    v214 = (iv43 < v204)
                    v215 = (iv43 >= 0)
                    v216 = (v215 & v214)
                    v217 = (iv203 >= 0)
                    v218 = (v217 & v209)
                    v219 = (v216 & v218)
                    v220 = tl.load((rhs_work + (iv43) * S3_0 + (iv203) * S3_1), mask=v219, other=0.0)
                    v221 = (v177 * v220)
                    v222 = (v213 - v221)
                    v223 = (iv166 >= 0)
                    v224 = (v223 & v205)
                    v225 = (iv203 >= 0)
                    v226 = (v225 & v209)
                    v227 = (v224 & v226)
                    tl.store((rhs_work + (iv166) * S3_0 + (iv203) * S3_1), v222, mask=v227)
            else:
    for iv228 in range(0, D1, 1):
        for iv229 in range(0, D2, 1):
            v230 = D1
            v231 = (iv228 < v230)
            v232 = (iv228 >= 0)
            v233 = (v232 & v231)
            v234 = D2
            v235 = (iv229 < v234)
            v236 = (iv229 >= 0)
            v237 = (v236 & v235)
            v238 = (v233 & v237)
            v239 = tl.load((rhs_work + (iv228) * S3_0 + (iv229) * S3_1), mask=v238, other=0.0)
            v240 = D1
            v241 = (iv228 < v240)
            v242 = (iv228 >= 0)
            v243 = (v242 & v241)
            v244 = D2
            v245 = (iv229 < v244)
            v246 = (iv229 >= 0)
            v247 = (v246 & v245)
            v248 = (v243 & v247)
            tl.store((result + (iv228) * S4_0 + (iv229) * S4_1), v239, mask=v248)

def launch(matrix, rhs, work, rhs_work, result):
    D1 = matrix.shape[0]
    D2 = rhs.shape[1]
    S0_0 = matrix.stride(0)
    S0_1 = matrix.stride(1)
    S1_0 = rhs.stride(0)
    S1_1 = rhs.stride(1)
    S2_0 = work.stride(0)
    S2_1 = work.stride(1)
    S3_0 = rhs_work.stride(0)
    S3_1 = rhs_work.stride(1)
    S4_0 = result.stride(0)
    S4_1 = result.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](matrix, rhs, work, rhs_work, result, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S3_0, S3_1, S4_0, S4_1)

def run(matrix, rhs, work, rhs_work):
    result = torch.empty((matrix.shape[0], rhs.shape[1]), device=matrix.device, dtype=torch.float32)
    launch(matrix, rhs, work, rhs_work, result)
    return result
