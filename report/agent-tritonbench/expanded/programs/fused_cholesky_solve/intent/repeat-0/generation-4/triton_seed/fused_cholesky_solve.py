import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("A", "b", "L", "y", "x", ), (False, False, True, True, True, ))

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
    key=["D1", "D6", "D8", "D10", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S3_0", "S3_1", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(A, b, L, y, x, D1: tl.constexpr, D6: tl.constexpr, D8: tl.constexpr, D10: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        v3 = D1
        v4 = (iv2 < v3)
        v5 = (iv2 >= 0)
        v6 = (v5 & v4)
        v7 = D1
        v8 = (iv2 < v7)
        v9 = (iv2 >= 0)
        v10 = (v9 & v8)
        v11 = (v6 & v10)
        v12 = tl.load((A + (iv2) * S0_0 + (iv2) * S0_1), mask=v11, other=0.0)
        v13 = v12
        for iv14 in range(0, iv2, 1):
            v15 = D1
            v16 = (iv2 < v15)
            v17 = (iv2 >= 0)
            v18 = (v17 & v16)
            v19 = D1
            v20 = (iv14 < v19)
            v21 = (iv14 >= 0)
            v22 = (v21 & v20)
            v23 = (v18 & v22)
            v24 = tl.load((L + (iv2) * S2_0 + (iv14) * S2_1), mask=v23, other=0.0)
            v25 = (iv2 >= 0)
            v26 = (v25 & v16)
            v27 = (iv14 >= 0)
            v28 = (v27 & v20)
            v29 = (v26 & v28)
            v30 = tl.load((L + (iv2) * S2_0 + (iv14) * S2_1), mask=v29, other=0.0)
            v31 = (v24 * v30)
            v32 = (v13 - v31)
            v13 = v32
        v33 = tl.maximum(v13, 1.0, propagate_nan=tl.PropagateNan.ALL)
        v34 = v33
        for iv35 in range(0, 32, 1):
            v36 = tl.fdiv(tl.cast(v13, tl.float32), tl.cast(v34, tl.float32), ieee_rounding=True)
            v37 = (v34 + v36)
            v38 = tl.fdiv(tl.cast(v37, tl.float32), tl.cast(2.0, tl.float32), ieee_rounding=True)
            v34 = v38
        v39 = D1
        v40 = (iv2 < v39)
        v41 = (iv2 >= 0)
        v42 = (v41 & v40)
        v43 = D1
        v44 = (iv2 < v43)
        v45 = (iv2 >= 0)
        v46 = (v45 & v44)
        v47 = (v42 & v46)
        tl.store((L + (iv2) * S2_0 + (iv2) * S2_1), v34, mask=v47)
        v48 = (iv2 + 1)
        for iv49 in range(v48, D1, 1):
            v50 = D1
            v51 = (iv49 < v50)
            v52 = (iv49 >= 0)
            v53 = (v52 & v51)
            v54 = D1
            v55 = (iv2 < v54)
            v56 = (iv2 >= 0)
            v57 = (v56 & v55)
            v58 = (v53 & v57)
            v59 = tl.load((A + (iv49) * S0_0 + (iv2) * S0_1), mask=v58, other=0.0)
            v60 = v59
            for iv61 in range(0, iv2, 1):
                v62 = D1
                v63 = (iv49 < v62)
                v64 = (iv49 >= 0)
                v65 = (v64 & v63)
                v66 = D1
                v67 = (iv61 < v66)
                v68 = (iv61 >= 0)
                v69 = (v68 & v67)
                v70 = (v65 & v69)
                v71 = tl.load((L + (iv49) * S2_0 + (iv61) * S2_1), mask=v70, other=0.0)
                v72 = (iv2 < v62)
                v73 = (iv2 >= 0)
                v74 = (v73 & v72)
                v75 = (iv61 >= 0)
                v76 = (v75 & v67)
                v77 = (v74 & v76)
                v78 = tl.load((L + (iv2) * S2_0 + (iv61) * S2_1), mask=v77, other=0.0)
                v79 = (v71 * v78)
                v80 = (v60 - v79)
                v60 = v80
            v81 = tl.fdiv(tl.cast(v60, tl.float32), tl.cast(v34, tl.float32), ieee_rounding=True)
            v82 = D1
            v83 = (iv49 < v82)
            v84 = (iv49 >= 0)
            v85 = (v84 & v83)
            v86 = D1
            v87 = (iv2 < v86)
            v88 = (iv2 >= 0)
            v89 = (v88 & v87)
            v90 = (v85 & v89)
            tl.store((L + (iv49) * S2_0 + (iv2) * S2_1), v81, mask=v90)
    for iv91 in range(0, D1, 1):
        v92 = D1
        v93 = (iv91 < v92)
        v94 = (iv91 >= 0)
        v95 = (v94 & v93)
        v96 = tl.load((b + (iv91) * S1_0 + (0) * S1_1), mask=v95, other=0.0)
        v97 = v96
        for iv98 in range(0, iv91, 1):
            v99 = D1
            v100 = (iv91 < v99)
            v101 = (iv91 >= 0)
            v102 = (v101 & v100)
            v103 = D1
            v104 = (iv98 < v103)
            v105 = (iv98 >= 0)
            v106 = (v105 & v104)
            v107 = (v102 & v106)
            v108 = tl.load((L + (iv91) * S2_0 + (iv98) * S2_1), mask=v107, other=0.0)
            v109 = D1
            v110 = (iv98 < v109)
            v111 = (iv98 >= 0)
            v112 = (v111 & v110)
            v113 = tl.load((y + (iv98) * S3_0 + (0) * S3_1), mask=v112, other=0.0)
            v114 = (v108 * v113)
            v115 = (v97 - v114)
            v97 = v115
        v116 = D1
        v117 = (iv91 < v116)
        v118 = (iv91 >= 0)
        v119 = (v118 & v117)
        v120 = D1
        v121 = (iv91 < v120)
        v122 = (iv91 >= 0)
        v123 = (v122 & v121)
        v124 = (v119 & v123)
        v125 = tl.load((L + (iv91) * S2_0 + (iv91) * S2_1), mask=v124, other=0.0)
        v126 = tl.fdiv(tl.cast(v97, tl.float32), tl.cast(v125, tl.float32), ieee_rounding=True)
        v127 = D1
        v128 = (iv91 < v127)
        v129 = (iv91 >= 0)
        v130 = (v129 & v128)
        tl.store((y + (iv91) * S3_0 + (0) * S3_1), v126, mask=v130)
    for iv131 in range(0, D1, 1):
        v132 = (D1 - 1)
        v133 = (v132 - iv131)
        v134 = D1
        v135 = (v133 < v134)
        v136 = (v133 >= 0)
        v137 = (v136 & v135)
        v138 = tl.load((y + (v133) * S3_0 + (0) * S3_1), mask=v137, other=0.0)
        v139 = (v133 + 1)
        v140 = v138
        for iv141 in range(v139, D1, 1):
            v142 = D1
            v143 = (iv141 < v142)
            v144 = (iv141 >= 0)
            v145 = (v144 & v143)
            v146 = D1
            v147 = (v133 < v146)
            v148 = (v133 >= 0)
            v149 = (v148 & v147)
            v150 = (v145 & v149)
            v151 = tl.load((L + (iv141) * S2_0 + (v133) * S2_1), mask=v150, other=0.0)
            v152 = D1
            v153 = (iv141 < v152)
            v154 = (iv141 >= 0)
            v155 = (v154 & v153)
            v156 = tl.load((x + (iv141) * S4_0 + (0) * S4_1), mask=v155, other=0.0)
            v157 = (v151 * v156)
            v158 = (v140 - v157)
            v140 = v158
        v159 = D1
        v160 = (v133 < v159)
        v161 = (v133 >= 0)
        v162 = (v161 & v160)
        v163 = D1
        v164 = (v133 < v163)
        v165 = (v133 >= 0)
        v166 = (v165 & v164)
        v167 = (v162 & v166)
        v168 = tl.load((L + (v133) * S2_0 + (v133) * S2_1), mask=v167, other=0.0)
        v169 = tl.fdiv(tl.cast(v140, tl.float32), tl.cast(v168, tl.float32), ieee_rounding=True)
        v170 = D1
        v171 = (v133 < v170)
        v172 = (v133 >= 0)
        v173 = (v172 & v171)
        tl.store((x + (v133) * S4_0 + (0) * S4_1), v169, mask=v173)

def launch(A, b, L, y, x):
    D1 = A.shape[0]
    D6 = b.shape[1]
    D8 = y.shape[1]
    D10 = x.shape[1]
    S0_0 = A.stride(0)
    S0_1 = A.stride(1)
    S1_0 = b.stride(0)
    S1_1 = b.stride(1)
    S2_0 = L.stride(0)
    S2_1 = L.stride(1)
    S3_0 = y.stride(0)
    S3_1 = y.stride(1)
    S4_0 = x.stride(0)
    S4_1 = x.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](A, b, L, y, x, D1, D6, D8, D10, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S3_0, S3_1, S4_0, S4_1)

def run(A, b, L, y, x):
    launch(A, b, L, y, x)
    return None
