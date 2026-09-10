import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "bias", "output", ), (False, False, False, True, ))

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
    key=["D1", "D2", "D3", "D4", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S3_0", "S3_1", "S3_2", "S3_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, bias, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, D10: tl.constexpr, D11: tl.constexpr, D12: tl.constexpr, D13: tl.constexpr, D14: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, S3_2: tl.constexpr, S3_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = ((((v0 // 14) // 14) // 128) % 8)
    v2 = (((v0 // 14) // 14) % 128)
    v3 = ((v0 // 14) % 14)
    v4 = (v0 % 14)
    v5 = (v1 * 1)
    v6 = (0 + v5)
    v7 = (v2 * 1)
    v8 = (0 + v7)
    v9 = (v3 * 1)
    v10 = (0 + v9)
    v11 = (v4 * 1)
    v12 = (0 + v11)
    v13 = tl.cast(0, tl.float32)
    v14 = ((64 // 1) - (((64 % 1) != 0) & (((64 % 1) < 0) != (1 < 0))))
    v15 = v13
    for iv16 in range(0, v14, 1):
        v17 = v15
        for iv18 in range(0, 3, 1):
            v19 = v17
            for iv20 in range(0, 3, 1):
                v21 = (v10 * 1)
                v22 = (iv18 * 1)
                v23 = (v21 + v22)
                v24 = (v23 - 0)
                v25 = (v12 * 1)
                v26 = (iv20 * 1)
                v27 = (v25 + v26)
                v28 = (v27 - 0)
                v29 = (v6 < 8)
                v30 = (v6 >= 0)
                v31 = (v30 & v29)
                v32 = (iv16 < 64)
                v33 = (iv16 >= 0)
                v34 = (v33 & v32)
                v35 = (v31 & v34)
                v36 = (v24 < 16)
                v37 = (v24 >= 0)
                v38 = (v37 & v36)
                v39 = (v35 & v38)
                v40 = (v28 < 16)
                v41 = (v28 >= 0)
                v42 = (v41 & v40)
                v43 = (v39 & v42)
                v44 = (v8 < 128)
                v45 = (v8 >= 0)
                v46 = (v45 & v44)
                v47 = (iv16 < 64)
                v48 = (iv16 >= 0)
                v49 = (v48 & v47)
                v50 = (v46 & v49)
                v51 = (iv18 < 3)
                v52 = (iv18 >= 0)
                v53 = (v52 & v51)
                v54 = (v50 & v53)
                v55 = (iv20 < 3)
                v56 = (iv20 >= 0)
                v57 = (v56 & v55)
                v58 = (v54 & v57)
                v59 = tl.load((input + (v6) * S0_0 + (iv16) * S0_1 + (v24) * S0_2 + (v28) * S0_3), mask=v43, other=0.0)
                v60 = tl.load((weight + (v8) * S1_0 + (iv16) * S1_1 + (iv18) * S1_2 + (iv20) * S1_3), mask=v58, other=0.0)
                v61 = (v59 * v60)
                v62 = (v19 + v61)
                v19 = v62
            v17 = v19
        v15 = v17
    v63 = (v8 < 128)
    v64 = (v8 >= 0)
    v65 = (v64 & v63)
    v66 = (v6 < 8)
    v67 = (v6 >= 0)
    v68 = (v67 & v66)
    v69 = (v8 < 128)
    v70 = (v8 >= 0)
    v71 = (v70 & v69)
    v72 = (v68 & v71)
    v73 = (v10 < 14)
    v74 = (v10 >= 0)
    v75 = (v74 & v73)
    v76 = (v72 & v75)
    v77 = (v12 < 14)
    v78 = (v12 >= 0)
    v79 = (v78 & v77)
    v80 = (v76 & v79)
    v81 = tl.load((bias + (v8) * S2_0), mask=v65, other=0.0)
    v82 = (v15 + v81)
    tl.store((output + (v6) * S3_0 + (v8) * S3_1 + (v10) * S3_2 + (v12) * S3_3), v82, mask=v80)

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
    grid = lambda META: (200704,)
    return _intent_kernel[grid](input, weight, bias, output, D1, D2, D3, D4, D6, D7, D8, D9, D10, D11, D12, D13, D14, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S3_0, S3_1, S3_2, S3_3)

def run(input, weight, bias):
    output = torch.empty((8, 128, 14, 14), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output)
    return output
