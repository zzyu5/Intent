import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("A", "B", "work", "X", ), (False, False, True, True, ))

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
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(A, B, work, X, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = (D1 + 1)
    for iv3 in range(0, D1, 1):
        v4 = (iv3 * v2)
        for iv5 in range(0, D1, 1):
            v6 = D1
            v7 = (iv3 < v6)
            v8 = (iv3 >= 0)
            v9 = (v8 & v7)
            v10 = D1
            v11 = (iv5 < v10)
            v12 = (iv5 >= 0)
            v13 = (v12 & v11)
            v14 = (v9 & v13)
            v15 = (v4 + iv5)
            v16 = D3
            v17 = (v15 < v16)
            v18 = (v15 >= 0)
            v19 = (v18 & v17)
            v20 = tl.load((A + (iv3) * S0_0 + (iv5) * S0_1), mask=v14, other=0.0)
            tl.store((work + (v15) * S2_0), v20, mask=v19)
        for iv21 in range(0, D2, 1):
            v22 = D1
            v23 = (iv3 < v22)
            v24 = (iv3 >= 0)
            v25 = (v24 & v23)
            v26 = D2
            v27 = (iv21 < v26)
            v28 = (iv21 >= 0)
            v29 = (v28 & v27)
            v30 = (v25 & v29)
            v31 = (v4 + D1)
            v32 = (v31 + iv21)
            v33 = D3
            v34 = (v32 < v33)
            v35 = (v32 >= 0)
            v36 = (v35 & v34)
            v37 = tl.load((B + (iv3) * S1_0 + (iv21) * S1_1), mask=v30, other=0.0)
            tl.store((work + (v32) * S2_0), v37, mask=v36)
    for iv38 in range(0, D1, 1):
        v39 = (iv38 * v2)
        v40 = (v39 + iv38)
        v41 = D3
        v42 = (v40 < v41)
        v43 = (v40 >= 0)
        v44 = (v43 & v42)
        v45 = tl.load((work + (v40) * S2_0), mask=v44, other=0.0)
        v46 = (-v45)
        v47 = tl.maximum(v45, v46, propagate_nan=tl.PropagateNan.ALL)
        v48 = (iv38 + 1)
        v49 = v47
        v50 = iv38
        for iv51 in range(v48, D1, 1):
            v52 = (iv51 * v2)
            v53 = (v52 + iv38)
            v54 = D3
            v55 = (v53 < v54)
            v56 = (v53 >= 0)
            v57 = (v56 & v55)
            v58 = tl.load((work + (v53) * S2_0), mask=v57, other=0.0)
            v59 = (-v58)
            v60 = tl.maximum(v58, v59, propagate_nan=tl.PropagateNan.ALL)
            v61 = (v60 > v49)
            v62 = tl.where(v61, v60, v49)
            v63 = tl.where(v61, iv51, v50)
            v49 = v62
            v50 = v63
        v64 = (v50 != iv38)
        if v64:
            v65 = (v50 * v2)
            for iv66 in range(0, D1, 1):
                v67 = (v39 + iv66)
                v68 = D3
                v69 = (v67 < v68)
                v70 = (v67 >= 0)
                v71 = (v70 & v69)
                v72 = tl.load((work + (v67) * S2_0), mask=v71, other=0.0)
                v73 = (v65 + iv66)
                v74 = (v73 < v68)
                v75 = (v73 >= 0)
                v76 = (v75 & v74)
                v77 = tl.load((work + (v73) * S2_0), mask=v76, other=0.0)
                v78 = (v39 + iv66)
                v79 = (v78 < v68)
                v80 = (v78 >= 0)
                v81 = (v80 & v79)
                tl.store((work + (v78) * S2_0), v77, mask=v81)
                v82 = (v65 + iv66)
                v83 = (v82 < v68)
                v84 = (v82 >= 0)
                v85 = (v84 & v83)
                tl.store((work + (v82) * S2_0), v72, mask=v85)
            for iv86 in range(0, D2, 1):
                v87 = (v39 + D1)
                v88 = (v87 + iv86)
                v89 = D3
                v90 = (v88 < v89)
                v91 = (v88 >= 0)
                v92 = (v91 & v90)
                v93 = tl.load((work + (v88) * S2_0), mask=v92, other=0.0)
                v94 = (v65 + D1)
                v95 = (v94 + iv86)
                v96 = (v95 < v89)
                v97 = (v95 >= 0)
                v98 = (v97 & v96)
                v99 = tl.load((work + (v95) * S2_0), mask=v98, other=0.0)
                v100 = (v39 + D1)
                v101 = (v100 + iv86)
                v102 = (v101 < v89)
                v103 = (v101 >= 0)
                v104 = (v103 & v102)
                tl.store((work + (v101) * S2_0), v99, mask=v104)
                v105 = (v65 + D1)
                v106 = (v105 + iv86)
                v107 = (v106 < v89)
                v108 = (v106 >= 0)
                v109 = (v108 & v107)
                tl.store((work + (v106) * S2_0), v93, mask=v109)
        else:
        v110 = (v39 + iv38)
        v111 = (v110 < v41)
        v112 = (v110 >= 0)
        v113 = (v112 & v111)
        v114 = tl.load((work + (v110) * S2_0), mask=v113, other=0.0)
        for iv115 in range(0, D1, 1):
            v116 = (v39 + iv115)
            v117 = D3
            v118 = (v116 < v117)
            v119 = (v116 >= 0)
            v120 = (v119 & v118)
            v121 = tl.load((work + (v116) * S2_0), mask=v120, other=0.0)
            v122 = tl.fdiv(tl.cast(v121, tl.float32), tl.cast(v114, tl.float32), ieee_rounding=True)
            v123 = (v39 + iv115)
            v124 = (v123 < v117)
            v125 = (v123 >= 0)
            v126 = (v125 & v124)
            tl.store((work + (v123) * S2_0), v122, mask=v126)
        for iv127 in range(0, D2, 1):
            v128 = (v39 + D1)
            v129 = (v128 + iv127)
            v130 = D3
            v131 = (v129 < v130)
            v132 = (v129 >= 0)
            v133 = (v132 & v131)
            v134 = tl.load((work + (v129) * S2_0), mask=v133, other=0.0)
            v135 = tl.fdiv(tl.cast(v134, tl.float32), tl.cast(v114, tl.float32), ieee_rounding=True)
            v136 = (v39 + D1)
            v137 = (v136 + iv127)
            v138 = (v137 < v130)
            v139 = (v137 >= 0)
            v140 = (v139 & v138)
            tl.store((work + (v137) * S2_0), v135, mask=v140)
        for iv141 in range(0, D1, 1):
            v142 = (iv141 != iv38)
            if v142:
                v143 = (iv141 * v2)
                v144 = (v143 + iv38)
                v145 = D3
                v146 = (v144 < v145)
                v147 = (v144 >= 0)
                v148 = (v147 & v146)
                v149 = tl.load((work + (v144) * S2_0), mask=v148, other=0.0)
                for iv150 in range(0, D1, 1):
                    v151 = (v143 + iv150)
                    v152 = D3
                    v153 = (v151 < v152)
                    v154 = (v151 >= 0)
                    v155 = (v154 & v153)
                    v156 = tl.load((work + (v151) * S2_0), mask=v155, other=0.0)
                    v157 = (v39 + iv150)
                    v158 = (v157 < v152)
                    v159 = (v157 >= 0)
                    v160 = (v159 & v158)
                    v161 = tl.load((work + (v157) * S2_0), mask=v160, other=0.0)
                    v162 = (v149 * v161)
                    v163 = (v156 - v162)
                    v164 = (v143 + iv150)
                    v165 = (v164 < v152)
                    v166 = (v164 >= 0)
                    v167 = (v166 & v165)
                    tl.store((work + (v164) * S2_0), v163, mask=v167)
                for iv168 in range(0, D2, 1):
                    v169 = (v143 + D1)
                    v170 = (v169 + iv168)
                    v171 = D3
                    v172 = (v170 < v171)
                    v173 = (v170 >= 0)
                    v174 = (v173 & v172)
                    v175 = tl.load((work + (v170) * S2_0), mask=v174, other=0.0)
                    v176 = (v39 + D1)
                    v177 = (v176 + iv168)
                    v178 = (v177 < v171)
                    v179 = (v177 >= 0)
                    v180 = (v179 & v178)
                    v181 = tl.load((work + (v177) * S2_0), mask=v180, other=0.0)
                    v182 = (v149 * v181)
                    v183 = (v175 - v182)
                    v184 = (v143 + D1)
                    v185 = (v184 + iv168)
                    v186 = (v185 < v171)
                    v187 = (v185 >= 0)
                    v188 = (v187 & v186)
                    tl.store((work + (v185) * S2_0), v183, mask=v188)
            else:
    for iv189 in range(0, D1, 1):
        v190 = (iv189 * v2)
        for iv191 in range(0, D2, 1):
            v192 = (v190 + D1)
            v193 = (v192 + iv191)
            v194 = D3
            v195 = (v193 < v194)
            v196 = (v193 >= 0)
            v197 = (v196 & v195)
            v198 = tl.load((work + (v193) * S2_0), mask=v197, other=0.0)
            v199 = D1
            v200 = (iv189 < v199)
            v201 = (iv189 >= 0)
            v202 = (v201 & v200)
            v203 = D2
            v204 = (iv191 < v203)
            v205 = (iv191 >= 0)
            v206 = (v205 & v204)
            v207 = (v202 & v206)
            tl.store((X + (iv189) * S3_0 + (iv191) * S3_1), v198, mask=v207)

def launch(A, B, work, X):
    D1 = A.shape[0]
    D2 = B.shape[1]
    D3 = work.shape[0]
    S0_0 = A.stride(0)
    S0_1 = A.stride(1)
    S1_0 = B.stride(0)
    S1_1 = B.stride(1)
    S2_0 = work.stride(0)
    S3_0 = X.stride(0)
    S3_1 = X.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](A, B, work, X, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S3_0, S3_1)

def run(A, B, work):
    X = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=torch.float32)
    launch(A, B, work, X)
    return X
