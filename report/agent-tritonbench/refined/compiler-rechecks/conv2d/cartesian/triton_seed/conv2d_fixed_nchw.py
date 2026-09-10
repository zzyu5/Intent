import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "bias", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D3": 1, "FRAGMENT_D4": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 1, "FRAGMENT_D4": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 1, "FRAGMENT_D4": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 1, "FRAGMENT_D4": 256}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 64, "FRAGMENT_D4": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 64, "FRAGMENT_D4": 64}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 4, "FRAGMENT_D4": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 4, "FRAGMENT_D4": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 4, "FRAGMENT_D4": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 4, "FRAGMENT_D4": 1024}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 16, "FRAGMENT_D4": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 16, "FRAGMENT_D4": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 16, "FRAGMENT_D4": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 16, "FRAGMENT_D4": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 8}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 2}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 2}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 2}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D3": 8, "FRAGMENT_D4": 2}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, bias, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, D10: tl.constexpr, D11: tl.constexpr, D12: tl.constexpr, D13: tl.constexpr, D14: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr, FRAGMENT_D4: tl.constexpr, FRAGMENT_D3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (FRAGMENT_D3 - 1)
    v2 = (14 + v1)
    v3 = ((v2 // FRAGMENT_D3) - (((v2 % FRAGMENT_D3) != 0) & (((v2 % FRAGMENT_D3) < 0) != (FRAGMENT_D3 < 0))))
    v4 = (FRAGMENT_D4 - 1)
    v5 = (14 + v4)
    v6 = ((v5 // FRAGMENT_D4) - (((v5 % FRAGMENT_D4) != 0) & (((v5 % FRAGMENT_D4) < 0) != (FRAGMENT_D4 < 0))))
    v7 = ((((v0 // v6) // v3) // 128) % 8)
    v8 = (((v0 // v6) // v3) % 128)
    v9 = ((v0 // v6) % v3)
    v10 = (v0 % v6)
    v11 = (v9 * FRAGMENT_D3)
    v12 = (v10 * FRAGMENT_D4)
    v13 = (v7 * 1)
    v14 = (0 + v13)
    v15 = (v8 * 1)
    v16 = (0 + v15)
    v17 = (v11 * 1)
    v18 = (0 + v17)
    v19 = (v18 + 1)
    v20 = (14 - v11)
    v21 = (v20 * 1)
    v22 = (v18 + v21)
    v23 = (v18 + tl.arange(0, FRAGMENT_D3) * 1)
    v24 = tl.full((FRAGMENT_D3,), v22, tl.int64)
    v25 = (v23 < v24)
    v26 = (v12 * 1)
    v27 = (0 + v26)
    v28 = (v27 + 1)
    v29 = (14 - v12)
    v30 = (v29 * 1)
    v31 = (v27 + v30)
    v32 = (v27 + tl.arange(0, FRAGMENT_D4) * 1)
    v33 = tl.full((FRAGMENT_D4,), v31, tl.int64)
    v34 = (v32 < v33)
    v35 = tl.cast(0, tl.float32)
    v36 = ((64 // 1) - (((64 % 1) != 0) & (((64 % 1) < 0) != (1 < 0))))
    v37 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v35, tl.float32)
    v38 = v37
    for iv39 in range(0, v36, 1):
        v40 = v38
        for iv41 in range(0, 3, 1):
            v42 = v40
            for iv43 in range(0, 3, 1):
                v44 = tl.broadcast_to(v23[None, :], (FRAGMENT_D4, FRAGMENT_D3))
                v45 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 1, tl.int64)
                v46 = (v44 * v45)
                v47 = (iv41 * 1)
                v48 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v47, tl.int64)
                v49 = (v46 + v48)
                v50 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
                v51 = (v49 - v50)
                v52 = tl.broadcast_to(v32[:, None], (FRAGMENT_D4, FRAGMENT_D3))
                v53 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 1, tl.int64)
                v54 = (v52 * v53)
                v55 = (iv43 * 1)
                v56 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v55, tl.int64)
                v57 = (v54 + v56)
                v58 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
                v59 = (v57 - v58)
                v60 = (v14 < 8)
                v61 = (v14 >= 0)
                v62 = (v61 & v60)
                v63 = (iv39 < 64)
                v64 = (iv39 >= 0)
                v65 = (v64 & v63)
                v66 = (v62 & v65)
                v67 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 16, tl.int64)
                v68 = (v51 < v67)
                v69 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
                v70 = (v51 >= v69)
                v71 = (v70 & v68)
                v72 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v66, tl.int1)
                v73 = (v72 & v71)
                v74 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 16, tl.int64)
                v75 = (v59 < v74)
                v76 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
                v77 = (v59 >= v76)
                v78 = (v77 & v75)
                v79 = (v73 & v78)
                v80 = (v16 < 128)
                v81 = (v16 >= 0)
                v82 = (v81 & v80)
                v83 = (iv39 < 64)
                v84 = (iv39 >= 0)
                v85 = (v84 & v83)
                v86 = (v82 & v85)
                v87 = (iv41 < 3)
                v88 = (iv41 >= 0)
                v89 = (v88 & v87)
                v90 = (v86 & v89)
                v91 = (iv43 < 3)
                v92 = (iv43 >= 0)
                v93 = (v92 & v91)
                v94 = (v90 & v93)
                v95 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0.0, tl.float32)
                v96 = tl.load((weight + (v16) * S1_0 + (iv39) * S1_1 + (iv41) * S1_2 + (iv43) * S1_3), mask=v94, other=0.0)
                v97 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v96, tl.float32)
                v98 = tl.load((input + (tl.full((FRAGMENT_D4, FRAGMENT_D3), v14, tl.int64)) * S0_0 + (tl.full((FRAGMENT_D4, FRAGMENT_D3), iv39, tl.int64)) * S0_1 + (tl.broadcast_to(v51, (FRAGMENT_D4, FRAGMENT_D3))) * S0_2 + (tl.broadcast_to(v59, (FRAGMENT_D4, FRAGMENT_D3))) * S0_3), mask=v79, other=v95)
                v99 = (v98 * v97)
                v100 = (v42 + v99)
                v42 = v100
            v40 = v42
        v38 = v40
    v101 = (v16 < 128)
    v102 = (v16 >= 0)
    v103 = (v102 & v101)
    v104 = (v14 < 8)
    v105 = (v14 >= 0)
    v106 = (v105 & v104)
    v107 = (v16 < 128)
    v108 = (v16 >= 0)
    v109 = (v108 & v107)
    v110 = (v106 & v109)
    v111 = tl.broadcast_to(v23[None, :], (FRAGMENT_D4, FRAGMENT_D3))
    v112 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 14, tl.int64)
    v113 = (v111 < v112)
    v114 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
    v115 = (v111 >= v114)
    v116 = (v115 & v113)
    v117 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v110, tl.int1)
    v118 = (v117 & v116)
    v119 = tl.broadcast_to(v32[:, None], (FRAGMENT_D4, FRAGMENT_D3))
    v120 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 14, tl.int64)
    v121 = (v119 < v120)
    v122 = tl.full((FRAGMENT_D4, FRAGMENT_D3), 0, tl.int64)
    v123 = (v119 >= v122)
    v124 = (v123 & v121)
    v125 = (v118 & v124)
    v126 = tl.load((bias + (v16) * S2_0), mask=v103, other=0.0)
    v127 = tl.full((FRAGMENT_D4, FRAGMENT_D3), v126, tl.float32)
    v128 = (v38 + v127)
    v129 = tl.broadcast_to(v25[None, :], (FRAGMENT_D4, FRAGMENT_D3))
    v130 = (v125 & v129)
    v131 = tl.broadcast_to(v34[:, None], (FRAGMENT_D4, FRAGMENT_D3))
    v132 = (v130 & v131)
    tl.store((output + (tl.full((FRAGMENT_D4, FRAGMENT_D3), v14, tl.int64)) * S3_0 + (tl.full((FRAGMENT_D4, FRAGMENT_D3), v16, tl.int64)) * S3_1 + (tl.broadcast_to(v23[None, :], (FRAGMENT_D4, FRAGMENT_D3))) * S3_2 + (tl.broadcast_to(v32[:, None], (FRAGMENT_D4, FRAGMENT_D3))) * S3_3), v128, mask=v132)

def launch(input, weight, bias, output):
    D1 = input.shape[0]
    D2 = weight.shape[0]
    D3 = output.shape[2]
    D4 = output.shape[3]
    D6 = weight.shape[2]
    D7 = weight.shape[3]
    D8 = input.shape[1]
    D9 = input.shape[2]
    D10 = input.shape[3]
    D11 = weight.shape[1]
    D12 = bias.shape[0]
    D13 = output.shape[0]
    D14 = output.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S1_2 = weight.stride(2)
    S1_3 = weight.stride(3)
    S2_0 = bias.stride(0)
    S3_0 = output.stride(0)
    S3_1 = output.stride(1)
    S3_2 = output.stride(2)
    S3_3 = output.stride(3)
    grid = lambda META: ((((8 * 128) * triton.cdiv(14, META["FRAGMENT_D3"])) * triton.cdiv(14, META["FRAGMENT_D4"])),)
    return _intent_kernel[grid](input, weight, bias, output, D1, D2, D3, D4, D6, D7, D8, D9, D10, D11, D12, D13, D14, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, bias):
    output = torch.empty((8, 128, 14, 14), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output)
    return output
