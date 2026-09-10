import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "output", ), (False, False, True, ))

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
def _intent_kernel(input, weight, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D5 - 0)
    v5 = (v4 + 0)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (D3 - 0)
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
    v25 = 0.0
    for iv26 in range(0, D2, 1):
        v27 = v25
        for iv28 in range(0, D6, 1):
            v29 = (v22 + iv28)
            v30 = (v29 - 1)
            v31 = (v30 >= 0)
            if v31:
                v33 = (v30 < D3)
                if v33:
                    v35 = v27
                    for iv36 in range(0, D7, 1):
                        v37 = (v24 + iv36)
                        v38 = (v37 - 1)
                        v39 = (v38 >= 0)
                        if v39:
                            v41 = (v38 < D4)
                            if v41:
                                v43 = D1
                                v44 = (v18 < v43)
                                v45 = (v18 >= 0)
                                v46 = (v45 & v44)
                                v47 = D2
                                v48 = (iv26 < v47)
                                v49 = (iv26 >= 0)
                                v50 = (v49 & v48)
                                v51 = (v46 & v50)
                                v52 = D3
                                v53 = (v30 < v52)
                                v54 = (v30 >= 0)
                                v55 = (v54 & v53)
                                v56 = (v51 & v55)
                                v57 = D4
                                v58 = (v38 < v57)
                                v59 = (v38 >= 0)
                                v60 = (v59 & v58)
                                v61 = (v56 & v60)
                                v62 = D5
                                v63 = (v20 < v62)
                                v64 = (v20 >= 0)
                                v65 = (v64 & v63)
                                v66 = D2
                                v67 = (iv26 < v66)
                                v68 = (iv26 >= 0)
                                v69 = (v68 & v67)
                                v70 = (v65 & v69)
                                v71 = D6
                                v72 = (iv28 < v71)
                                v73 = (iv28 >= 0)
                                v74 = (v73 & v72)
                                v75 = (v70 & v74)
                                v76 = D7
                                v77 = (iv36 < v76)
                                v78 = (iv36 >= 0)
                                v79 = (v78 & v77)
                                v80 = (v75 & v79)
                                v81 = tl.load((input + (v18) * S0_0 + (iv26) * S0_1 + (v30) * S0_2 + (v38) * S0_3), mask=v61, other=0.0)
                                v82 = tl.load((weight + (v20) * S1_0 + (iv26) * S1_1 + (iv28) * S1_2 + (iv36) * S1_3), mask=v80, other=0.0)
                                v83 = (v81 * v82)
                                v84 = (v35 + v83)
                                v42 = v84
                            else:
                                v42 = v35
                            v40 = v42
                        else:
                            v40 = v35
                        v35 = v40
                    v34 = v35
                else:
                    v34 = v27
                v32 = v34
            else:
                v32 = v27
            v27 = v32
        v25 = v27
    v85 = (v25 < 0.0)
    if v85:
        v87 = (-v25)
        v88 = (v87 * 0.70710676908493042)
        v86 = v88
    else:
        v89 = (v25 * 0.70710676908493042)
        v86 = v89
    v90 = (0.32759109139442444 * v86)
    v91 = (1.0 + v90)
    v92 = tl.fdiv(tl.cast(1.0, tl.float32), tl.cast(v91, tl.float32), ieee_rounding=True)
    v93 = (1.0614054203033447 * v92)
    v94 = (v93 - 1.453152060508728)
    v95 = (v94 * v92)
    v96 = (v95 + 1.421413779258728)
    v97 = (v96 * v92)
    v98 = (v97 - 0.2844967246055603)
    v99 = (v98 * v92)
    v100 = (v99 + 0.25482958555221558)
    v101 = (v100 * v92)
    v102 = (-v86)
    v103 = (v102 * v86)
    v104 = (v103 * 1.4426950216293335)
    v105 = libdevice.exp2(tl.cast(v104, tl.float32))
    v106 = (v101 * v105)
    v107 = (1.0 - v106)
    v108 = (v25 < 0.0)
    if v108:
        v110 = (-v107)
        v109 = v110
    else:
        v109 = v107
    v111 = (0.5 * v25)
    v112 = (1.0 + v109)
    v113 = (v111 * v112)
    v114 = D1
    v115 = (v18 < v114)
    v116 = (v18 >= 0)
    v117 = (v116 & v115)
    v118 = D5
    v119 = (v20 < v118)
    v120 = (v20 >= 0)
    v121 = (v120 & v119)
    v122 = (v117 & v121)
    v123 = D3
    v124 = (v22 < v123)
    v125 = (v22 >= 0)
    v126 = (v125 & v124)
    v127 = (v122 & v126)
    v128 = D4
    v129 = (v24 < v128)
    v130 = (v24 >= 0)
    v131 = (v130 & v129)
    v132 = (v127 & v131)
    tl.store((output + (v18) * S2_0 + (v20) * S2_1 + (v22) * S2_2 + (v24) * S2_3), v113, mask=v132)

def launch(input, weight, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = input.shape[2]
    D4 = input.shape[3]
    D5 = weight.shape[0]
    D6 = weight.shape[2]
    D7 = weight.shape[3]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S1_2 = weight.stride(2)
    S1_3 = weight.stride(3)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    S2_2 = output.stride(2)
    S2_3 = output.stride(3)
    grid = lambda META: ((((D1 * D5) * D3) * D4),)
    return _intent_kernel[grid](input, weight, output, D1, D2, D3, D4, D5, D6, D7, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3)

def run(input, weight):
    output = torch.empty((input.shape[0], weight.shape[0], input.shape[2], input.shape[3]), device=input.device, dtype=torch.float32)
    launch(input, weight, output)
    return output
