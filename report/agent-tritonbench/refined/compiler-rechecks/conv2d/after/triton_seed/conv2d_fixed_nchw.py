import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "bias", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D4": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D4": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, bias, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, D10: tl.constexpr, D11: tl.constexpr, D12: tl.constexpr, D13: tl.constexpr, D14: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr, FRAGMENT_D4: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (FRAGMENT_D4 - 1)
    v2 = (14 + v1)
    v3 = ((v2 // FRAGMENT_D4) - (((v2 % FRAGMENT_D4) != 0) & (((v2 % FRAGMENT_D4) < 0) != (FRAGMENT_D4 < 0))))
    v4 = ((((v0 // v3) // 14) // 128) % 8)
    v5 = (((v0 // v3) // 14) % 128)
    v6 = ((v0 // v3) % 14)
    v7 = (v0 % v3)
    v8 = (v7 * FRAGMENT_D4)
    v9 = (v4 * 1)
    v10 = (0 + v9)
    v11 = (v5 * 1)
    v12 = (0 + v11)
    v13 = (v6 * 1)
    v14 = (0 + v13)
    v15 = (v8 * 1)
    v16 = (0 + v15)
    v17 = (v16 + 1)
    v18 = (14 - v8)
    v19 = (v18 * 1)
    v20 = (v16 + v19)
    v21 = (v16 + tl.arange(0, FRAGMENT_D4) * 1)
    v22 = tl.full((FRAGMENT_D4,), v20, tl.int64)
    v23 = (v21 < v22)
    v24 = tl.cast(0, tl.float32)
    v25 = ((64 // 1) - (((64 % 1) != 0) & (((64 % 1) < 0) != (1 < 0))))
    v26 = tl.full((FRAGMENT_D4,), v24, tl.float32)
    v27 = v26
    for iv28 in range(0, v25, 1):
        v29 = v27
        for iv30 in range(0, 3, 1):
            v31 = v29
            for iv32 in range(0, 3, 1):
                v33 = (v14 * 1)
                v34 = (iv30 * 1)
                v35 = (v33 + v34)
                v36 = (v35 - 0)
                v37 = tl.full((FRAGMENT_D4,), 1, tl.int64)
                v38 = (v21 * v37)
                v39 = (iv32 * 1)
                v40 = tl.full((FRAGMENT_D4,), v39, tl.int64)
                v41 = (v38 + v40)
                v42 = tl.full((FRAGMENT_D4,), 0, tl.int64)
                v43 = (v41 - v42)
                v44 = (v10 < 8)
                v45 = (v10 >= 0)
                v46 = (v45 & v44)
                v47 = (iv28 < 64)
                v48 = (iv28 >= 0)
                v49 = (v48 & v47)
                v50 = (v46 & v49)
                v51 = (v36 < 16)
                v52 = (v36 >= 0)
                v53 = (v52 & v51)
                v54 = (v50 & v53)
                v55 = tl.full((FRAGMENT_D4,), 16, tl.int64)
                v56 = (v43 < v55)
                v57 = tl.full((FRAGMENT_D4,), 0, tl.int64)
                v58 = (v43 >= v57)
                v59 = (v58 & v56)
                v60 = tl.full((FRAGMENT_D4,), v54, tl.int1)
                v61 = (v60 & v59)
                v62 = (v12 < 128)
                v63 = (v12 >= 0)
                v64 = (v63 & v62)
                v65 = (iv28 < 64)
                v66 = (iv28 >= 0)
                v67 = (v66 & v65)
                v68 = (v64 & v67)
                v69 = (iv30 < 3)
                v70 = (iv30 >= 0)
                v71 = (v70 & v69)
                v72 = (v68 & v71)
                v73 = (iv32 < 3)
                v74 = (iv32 >= 0)
                v75 = (v74 & v73)
                v76 = (v72 & v75)
                v77 = (v61 & v23)
                v78 = tl.full((FRAGMENT_D4,), 0.0, tl.float32)
                v79 = tl.load((weight + (v12) * S1_0 + (iv28) * S1_1 + (iv30) * S1_2 + (iv32) * S1_3), mask=v76, other=0.0)
                v80 = tl.full((FRAGMENT_D4,), v79, tl.float32)
                v81 = tl.load((input + (tl.full((FRAGMENT_D4,), v10, tl.int64)) * S0_0 + (tl.full((FRAGMENT_D4,), iv28, tl.int64)) * S0_1 + (tl.full((FRAGMENT_D4,), v36, tl.int64)) * S0_2 + (tl.broadcast_to(v43, (FRAGMENT_D4,))) * S0_3), mask=v77, other=v78)
                v82 = (v81 * v80)
                v83 = (v31 + v82)
                v31 = v83
            v29 = v31
        v27 = v29
    v84 = (v12 < 128)
    v85 = (v12 >= 0)
    v86 = (v85 & v84)
    v87 = (v10 < 8)
    v88 = (v10 >= 0)
    v89 = (v88 & v87)
    v90 = (v12 < 128)
    v91 = (v12 >= 0)
    v92 = (v91 & v90)
    v93 = (v89 & v92)
    v94 = (v14 < 14)
    v95 = (v14 >= 0)
    v96 = (v95 & v94)
    v97 = (v93 & v96)
    v98 = tl.full((FRAGMENT_D4,), 14, tl.int64)
    v99 = (v21 < v98)
    v100 = tl.full((FRAGMENT_D4,), 0, tl.int64)
    v101 = (v21 >= v100)
    v102 = (v101 & v99)
    v103 = tl.full((FRAGMENT_D4,), v97, tl.int1)
    v104 = (v103 & v102)
    v105 = tl.load((bias + (v12) * S2_0), mask=v86, other=0.0)
    v106 = tl.full((FRAGMENT_D4,), v105, tl.float32)
    v107 = (v27 + v106)
    v108 = (v104 & v23)
    tl.store((output + (tl.full((FRAGMENT_D4,), v10, tl.int64)) * S3_0 + (tl.full((FRAGMENT_D4,), v12, tl.int64)) * S3_1 + (tl.full((FRAGMENT_D4,), v14, tl.int64)) * S3_2 + (tl.broadcast_to(v21, (FRAGMENT_D4,))) * S3_3), v107, mask=v108)

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
    grid = lambda META: ((((8 * 128) * 14) * triton.cdiv(14, META["FRAGMENT_D4"])),)
    return _intent_kernel[grid](input, weight, bias, output, D1, D2, D3, D4, D6, D7, D8, D9, D10, D11, D12, D13, D14, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, bias):
    output = torch.empty((8, 128, 14, 14), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output)
    return output
