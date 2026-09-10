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
        v46 = (iv38 + 1)
        for iv47 in range(v46, D1, 1):
            v48 = (iv47 * v2)
            v49 = (v48 + iv38)
            v50 = D3
            v51 = (v49 < v50)
            v52 = (v49 >= 0)
            v53 = (v52 & v51)
            v54 = tl.load((work + (v49) * S2_0), mask=v53, other=0.0)
            v55 = tl.fdiv(tl.cast(v54, tl.float32), tl.cast(v45, tl.float32), ieee_rounding=True)
            v56 = (v48 + iv38)
            v57 = (v56 < v50)
            v58 = (v56 >= 0)
            v59 = (v58 & v57)
            tl.store((work + (v56) * S2_0), v55, mask=v59)
            v60 = (iv38 + 1)
            for iv61 in range(v60, D1, 1):
                v62 = (v48 + iv61)
                v63 = D3
                v64 = (v62 < v63)
                v65 = (v62 >= 0)
                v66 = (v65 & v64)
                v67 = tl.load((work + (v62) * S2_0), mask=v66, other=0.0)
                v68 = (v39 + iv61)
                v69 = (v68 < v63)
                v70 = (v68 >= 0)
                v71 = (v70 & v69)
                v72 = tl.load((work + (v68) * S2_0), mask=v71, other=0.0)
                v73 = (v55 * v72)
                v74 = (v67 - v73)
                v75 = (v48 + iv61)
                v76 = (v75 < v63)
                v77 = (v75 >= 0)
                v78 = (v77 & v76)
                tl.store((work + (v75) * S2_0), v74, mask=v78)
            for iv79 in range(0, D2, 1):
                v80 = (v48 + D1)
                v81 = (v80 + iv79)
                v82 = D3
                v83 = (v81 < v82)
                v84 = (v81 >= 0)
                v85 = (v84 & v83)
                v86 = tl.load((work + (v81) * S2_0), mask=v85, other=0.0)
                v87 = (v39 + D1)
                v88 = (v87 + iv79)
                v89 = (v88 < v82)
                v90 = (v88 >= 0)
                v91 = (v90 & v89)
                v92 = tl.load((work + (v88) * S2_0), mask=v91, other=0.0)
                v93 = (v55 * v92)
                v94 = (v86 - v93)
                v95 = (v48 + D1)
                v96 = (v95 + iv79)
                v97 = (v96 < v82)
                v98 = (v96 >= 0)
                v99 = (v98 & v97)
                tl.store((work + (v96) * S2_0), v94, mask=v99)
    for iv100 in range(0, D1, 1):
        v101 = (D1 - 1)
        v102 = (v101 - iv100)
        v103 = (v102 * v2)
        v104 = (v103 + v102)
        v105 = D3
        v106 = (v104 < v105)
        v107 = (v104 >= 0)
        v108 = (v107 & v106)
        v109 = tl.load((work + (v104) * S2_0), mask=v108, other=0.0)
        for iv110 in range(0, D2, 1):
            v111 = (v103 + D1)
            v112 = (v111 + iv110)
            v113 = D3
            v114 = (v112 < v113)
            v115 = (v112 >= 0)
            v116 = (v115 & v114)
            v117 = tl.load((work + (v112) * S2_0), mask=v116, other=0.0)
            v118 = tl.fdiv(tl.cast(v117, tl.float32), tl.cast(v109, tl.float32), ieee_rounding=True)
            v119 = (v103 + D1)
            v120 = (v119 + iv110)
            v121 = (v120 < v113)
            v122 = (v120 >= 0)
            v123 = (v122 & v121)
            tl.store((work + (v120) * S2_0), v118, mask=v123)
        v124 = tl.cast(0, tl.int64)
        for iv125 in range(v124, v102, 1):
            v126 = (iv125 * v2)
            for iv127 in range(0, D2, 1):
                v128 = (v126 + D1)
                v129 = (v128 + iv127)
                v130 = D3
                v131 = (v129 < v130)
                v132 = (v129 >= 0)
                v133 = (v132 & v131)
                v134 = tl.load((work + (v129) * S2_0), mask=v133, other=0.0)
                v135 = (v126 + v102)
                v136 = (v135 < v130)
                v137 = (v135 >= 0)
                v138 = (v137 & v136)
                v139 = tl.load((work + (v135) * S2_0), mask=v138, other=0.0)
                v140 = (v103 + D1)
                v141 = (v140 + iv127)
                v142 = (v141 < v130)
                v143 = (v141 >= 0)
                v144 = (v143 & v142)
                v145 = tl.load((work + (v141) * S2_0), mask=v144, other=0.0)
                v146 = (v139 * v145)
                v147 = (v134 - v146)
                v148 = (v126 + D1)
                v149 = (v148 + iv127)
                v150 = (v149 < v130)
                v151 = (v149 >= 0)
                v152 = (v151 & v150)
                tl.store((work + (v149) * S2_0), v147, mask=v152)
    for iv153 in range(0, D1, 1):
        v154 = (iv153 * v2)
        for iv155 in range(0, D2, 1):
            v156 = (v154 + D1)
            v157 = (v156 + iv155)
            v158 = D3
            v159 = (v157 < v158)
            v160 = (v157 >= 0)
            v161 = (v160 & v159)
            v162 = tl.load((work + (v157) * S2_0), mask=v161, other=0.0)
            v163 = D1
            v164 = (iv153 < v163)
            v165 = (iv153 >= 0)
            v166 = (v165 & v164)
            v167 = D2
            v168 = (iv155 < v167)
            v169 = (iv155 >= 0)
            v170 = (v169 & v168)
            v171 = (v166 & v170)
            tl.store((X + (iv153) * S3_0 + (iv155) * S3_1), v162, mask=v171)

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
