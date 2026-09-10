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
    key=["D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3", "S2_0", "S2_1", "S2_2", "S2_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, output, D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr, D8: tl.constexpr, D9: tl.constexpr, D10: tl.constexpr, D11: tl.constexpr, D12: tl.constexpr, D13: tl.constexpr, D14: tl.constexpr, D15: tl.constexpr, D16: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S2_2: tl.constexpr, S2_3: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 4194304)
    v2 = (v1 * 1)
    v3 = (0 + v2)
    v4 = ((v3 % 256) + (((v3 % 256) != 0) & (((v3 % 256) < 0) != (256 < 0))) * 256)
    v5 = ((v3 // 256) - (((v3 % 256) != 0) & (((v3 % 256) < 0) != (256 < 0))))
    v6 = ((v5 % 256) + (((v5 % 256) != 0) & (((v5 % 256) < 0) != (256 < 0))) * 256)
    v7 = ((v5 // 256) - (((v5 % 256) != 0) & (((v5 % 256) < 0) != (256 < 0))))
    v8 = 0.0
    for iv9 in range(0, 3, 1):
        v10 = v8
        for iv11 in range(0, 3, 1):
            v12 = (v6 + iv11)
            v13 = (v12 - 1)
            v14 = (v13 >= 0)
            if v14:
                v16 = (v13 < 256)
                if v16:
                    v18 = v10
                    for iv19 in range(0, 3, 1):
                        v20 = (v4 + iv19)
                        v21 = (v20 - 1)
                        v22 = (v21 >= 0)
                        v23 = (v21 < 256)
                        v24 = (iv9 < 3)
                        v25 = (iv9 >= 0)
                        v26 = (v25 & v24)
                        v27 = (v13 < 256)
                        v28 = (v13 >= 0)
                        v29 = (v28 & v27)
                        v30 = (v26 & v29)
                        v31 = (v21 < 256)
                        v32 = (v21 >= 0)
                        v33 = (v32 & v31)
                        v34 = (v30 & v33)
                        v35 = (v7 < 64)
                        v36 = (v7 >= 0)
                        v37 = (v36 & v35)
                        v38 = (iv9 < 3)
                        v39 = (iv9 >= 0)
                        v40 = (v39 & v38)
                        v41 = (v37 & v40)
                        v42 = (iv11 < 3)
                        v43 = (iv11 >= 0)
                        v44 = (v43 & v42)
                        v45 = (v41 & v44)
                        v46 = (iv19 < 3)
                        v47 = (iv19 >= 0)
                        v48 = (v47 & v46)
                        v49 = (v45 & v48)
                        v50 = (v23 & v49)
                        v51 = (v22 & v50)
                        v52 = (v23 & v34)
                        v53 = (v22 & v52)
                        v54 = tl.load((input + (0) * S0_0 + (iv9) * S0_1 + (v13) * S0_2 + (v21) * S0_3), mask=v53, other=0.0)
                        v55 = tl.load((weight + (v7) * S1_0 + (iv9) * S1_1 + (iv11) * S1_2 + (iv19) * S1_3), mask=v51, other=0.0)
                        v56 = (v54 * v55)
                        v57 = (v18 + v56)
                        v58 = tl.where(v23, v57, v18)
                        v59 = tl.where(v22, v58, v18)
                        v18 = v59
                    v17 = v18
                else:
                    v17 = v10
                v15 = v17
            else:
                v15 = v10
            v10 = v15
        v8 = v10
    v60 = tl.maximum(v8, 0.0, propagate_nan=tl.PropagateNan.ALL)
    v61 = (v7 < 64)
    v62 = (v7 >= 0)
    v63 = (v62 & v61)
    v64 = (v6 < 256)
    v65 = (v6 >= 0)
    v66 = (v65 & v64)
    v67 = (v63 & v66)
    v68 = (v4 < 256)
    v69 = (v4 >= 0)
    v70 = (v69 & v68)
    v71 = (v67 & v70)
    tl.store((output + (0) * S2_0 + (v7) * S2_1 + (v6) * S2_2 + (v4) * S2_3), v60, mask=v71)

def launch(input, weight, output):
    D5 = input.shape[0]
    D6 = input.shape[1]
    D7 = input.shape[2]
    D8 = input.shape[3]
    D9 = weight.shape[0]
    D10 = weight.shape[1]
    D11 = weight.shape[2]
    D12 = weight.shape[3]
    D13 = output.shape[0]
    D14 = output.shape[1]
    D15 = output.shape[2]
    D16 = output.shape[3]
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
    grid = lambda META: (4194304,)
    return _intent_kernel[grid](input, weight, output, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15, D16, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3, S2_0, S2_1, S2_2, S2_3)

def run(input, weight):
    output = torch.empty((1, 64, 256, 256), device=input.device, dtype=torch.float32)
    launch(input, weight, output)
    return output
