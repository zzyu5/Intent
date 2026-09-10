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
    key=["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3", "S4_0", "S4_1", "S4_2", "S4_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, other, output, alpha, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr, S4_2: tl.constexpr, S4_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (D2 - 0)
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
    for iv26 in range(0, D5, 1):
        v27 = v25
        for iv28 in range(0, D6, 1):
            v29 = (v22 + iv28)
            v30 = (v29 - 1)
            v31 = (v30 >= 0)
            if v31:
                v33 = (v30 < D8)
                if v33:
                    v35 = v27
                    for iv36 in range(0, D7, 1):
                        v37 = (v24 + iv36)
                        v38 = (v37 - 1)
                        v39 = (v38 >= 0)
                        if v39:
                            v41 = (v38 < D9)
                            if v41:
                                v43 = D1
                                v44 = (v18 < v43)
                                v45 = (v18 >= 0)
                                v46 = (v45 & v44)
                                v47 = D5
                                v48 = (iv26 < v47)
                                v49 = (iv26 >= 0)
                                v50 = (v49 & v48)
                                v51 = (v46 & v50)
                                v52 = D8
                                v53 = (v30 < v52)
                                v54 = (v30 >= 0)
                                v55 = (v54 & v53)
                                v56 = (v51 & v55)
                                v57 = D9
                                v58 = (v38 < v57)
                                v59 = (v38 >= 0)
                                v60 = (v59 & v58)
                                v61 = (v56 & v60)
                                v62 = D2
                                v63 = (v20 < v62)
                                v64 = (v20 >= 0)
                                v65 = (v64 & v63)
                                v66 = D5
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
    v95 = (v22 < v94)
    v96 = (v22 >= 0)
    v97 = (v96 & v95)
    v98 = (v93 & v97)
    v99 = D4
    v100 = (v24 < v99)
    v101 = (v24 >= 0)
    v102 = (v101 & v100)
    v103 = (v98 & v102)
    v104 = D1
    v105 = (v18 < v104)
    v106 = (v18 >= 0)
    v107 = (v106 & v105)
    v108 = D2
    v109 = (v20 < v108)
    v110 = (v20 >= 0)
    v111 = (v110 & v109)
    v112 = (v107 & v111)
    v113 = D3
    v114 = (v22 < v113)
    v115 = (v22 >= 0)
    v116 = (v115 & v114)
    v117 = (v112 & v116)
    v118 = D4
    v119 = (v24 < v118)
    v120 = (v24 >= 0)
    v121 = (v120 & v119)
    v122 = (v117 & v121)
    v123 = tl.load((other + (v18) * S2_0 + (v20) * S2_1 + (v22) * S2_2 + (v24) * S2_3), mask=v103, other=0.0)
    v124 = (alpha * v123)
    v125 = (v25 + v124)
    tl.store((output + (v18) * S4_0 + (v20) * S4_1 + (v22) * S4_2 + (v24) * S4_3), v125, mask=v122)

def launch(input, weight, other, output, alpha):
    D1 = input.shape[0]
    D2 = weight.shape[0]
    D3 = other.shape[2]
    D4 = other.shape[3]
    D5 = input.shape[1]
    D6 = weight.shape[2]
    D7 = weight.shape[3]
    D8 = input.shape[2]
    D9 = input.shape[3]
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
    S4_0 = output.stride(0)
    S4_1 = output.stride(1)
    S4_2 = output.stride(2)
    S4_3 = output.stride(3)
    grid = lambda META: ((((D1 * D2) * D3) * D4),)
    return _intent_kernel[grid](input, weight, other, output, alpha, D1, D2, D3, D4, D5, D6, D7, D8, D9, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3, S4_0, S4_1, S4_2, S4_3)

def run(input, weight, other, alpha):
    output = torch.empty((input.shape[0], weight.shape[0], other.shape[2], other.shape[3]), device=input.device, dtype=torch.float32)
    launch(input, weight, other, output, alpha)
    return output
