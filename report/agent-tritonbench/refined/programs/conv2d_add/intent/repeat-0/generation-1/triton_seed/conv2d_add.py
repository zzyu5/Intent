import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "other", "output", ), (False, False, False, True, ))

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
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, other, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr):
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
    v25 = tl.cast(0, tl.float32)
    v26 = v25
    for iv27 in range(0, D2, 1):
        v28 = v26
        for iv29 in range(0, D6, 1):
            v30 = v28
            for iv31 in range(0, D7, 1):
                v32 = (v22 + iv29)
                v33 = (v32 - 1)
                v34 = (v24 + iv31)
                v35 = (v34 - 1)
                v36 = (v33 >= 0)
                v37 = (v33 < D3)
                v38 = (v35 >= 0)
                v39 = (v35 < D4)
                v40 = D1
                v41 = (v18 < v40)
                v42 = (v18 >= 0)
                v43 = (v42 & v41)
                v44 = D2
                v45 = (iv27 < v44)
                v46 = (iv27 >= 0)
                v47 = (v46 & v45)
                v48 = (v43 & v47)
                v49 = D3
                v50 = (v33 < v49)
                v51 = (v33 >= 0)
                v52 = (v51 & v50)
                v53 = (v48 & v52)
                v54 = D4
                v55 = (v35 < v54)
                v56 = (v35 >= 0)
                v57 = (v56 & v55)
                v58 = (v53 & v57)
                v59 = D5
                v60 = (v20 < v59)
                v61 = (v20 >= 0)
                v62 = (v61 & v60)
                v63 = D2
                v64 = (iv27 < v63)
                v65 = (iv27 >= 0)
                v66 = (v65 & v64)
                v67 = (v62 & v66)
                v68 = D6
                v69 = (iv29 < v68)
                v70 = (iv29 >= 0)
                v71 = (v70 & v69)
                v72 = (v67 & v71)
                v73 = D7
                v74 = (iv31 < v73)
                v75 = (iv31 >= 0)
                v76 = (v75 & v74)
                v77 = (v72 & v76)
                v78 = (v39 & v77)
                v79 = (v38 & v78)
                v80 = (v37 & v79)
                v81 = (v36 & v80)
                v82 = (v39 & v58)
                v83 = (v38 & v82)
                v84 = (v37 & v83)
                v85 = (v36 & v84)
                v86 = tl.load((input + (v18) * S0_0 + (iv27) * S0_1 + (v33) * S0_2 + (v35) * S0_3), mask=v85, other=0.0)
                v87 = tl.load((weight + (v20) * S1_0 + (iv27) * S1_1 + (iv29) * S1_2 + (iv31) * S1_3), mask=v81, other=0.0)
                v88 = (v86 * v87)
                v89 = (v30 + v88)
                v90 = tl.where(v39, v89, v30)
                v91 = tl.where(v38, v90, v30)
                v92 = tl.where(v37, v91, v30)
                v93 = tl.where(v36, v92, v30)
                v30 = v93
            v28 = v30
        v26 = v28
    v94 = D1
    v95 = (v18 < v94)
    v96 = (v18 >= 0)
    v97 = (v96 & v95)
    v98 = D5
    v99 = (v20 < v98)
    v100 = (v20 >= 0)
    v101 = (v100 & v99)
    v102 = (v97 & v101)
    v103 = D8
    v104 = (v22 < v103)
    v105 = (v22 >= 0)
    v106 = (v105 & v104)
    v107 = (v102 & v106)
    v108 = D9
    v109 = (v24 < v108)
    v110 = (v24 >= 0)
    v111 = (v110 & v109)
    v112 = (v107 & v111)
    v113 = D1
    v114 = (v18 < v113)
    v115 = (v18 >= 0)
    v116 = (v115 & v114)
    v117 = D5
    v118 = (v20 < v117)
    v119 = (v20 >= 0)
    v120 = (v119 & v118)
    v121 = (v116 & v120)
    v122 = D8
    v123 = (v22 < v122)
    v124 = (v22 >= 0)
    v125 = (v124 & v123)
    v126 = (v121 & v125)
    v127 = D9
    v128 = (v24 < v127)
    v129 = (v24 >= 0)
    v130 = (v129 & v128)
    v131 = (v126 & v130)
    v132 = tl.load((other + (v18) * S2_0 + (v20) * S2_1 + (v22) * S2_2 + (v24) * S2_3), mask=v112, other=0.0)
    v133 = (v26 + v132)
    tl.store((output + (v18) * S3_0 + (v20) * S3_1 + (v22) * S3_2 + (v24) * S3_3), v133, mask=v131)

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
    grid = lambda META: ((((D1 * D5) * D8) * D9),)
    return _intent_kernel[grid](input, weight, other, output, D1, D2, D3, D4, D5, D6, D7, D8, D9, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, other):
    output = torch.empty((input.shape[0], weight.shape[0], other.shape[2], other.shape[3]), device=input.device, dtype=torch.float32)
    launch(input, weight, other, output)
    return output
