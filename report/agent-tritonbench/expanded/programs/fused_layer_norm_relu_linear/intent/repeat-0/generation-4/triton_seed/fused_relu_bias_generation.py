import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "bias", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D2": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, bias, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, FRAGMENT_D2: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = (FRAGMENT_D2 - 1)
    v4 = (D2 + v3)
    v5 = ((v4 // FRAGMENT_D2) - (((v4 % FRAGMENT_D2) != 0) & (((v4 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v6 = tl.program_id(1)
    v7 = tl.program_id(0)
    v8 = (v6 * v5)
    v9 = (v8 + v7)
    v10 = ((v9 // v5) % v2)
    v11 = (v9 % v5)
    v12 = (v10 * 1)
    v13 = (0 + v12)
    v14 = (v11 * FRAGMENT_D2)
    v15 = (v14 * 1)
    v16 = (0 + v15)
    v17 = (v16 + tl.arange(0, FRAGMENT_D2) * 1)
    v18 = D2
    v19 = (v17 < v18)
    v20 = D1
    v21 = (v13 < v20)
    v22 = (v13 >= 0)
    v23 = (v22 & v21)
    v24 = tl.full((FRAGMENT_D2,), v23, tl.int1)
    v25 = D2
    v26 = tl.full((FRAGMENT_D2,), v25, tl.int64)
    v27 = (v17 < v26)
    v28 = (v24 & v27)
    v29 = tl.full((FRAGMENT_D2,), 0.0, tl.float32)
    v30 = (v16 + tl.arange(0, FRAGMENT_D2) * 1)
    v31 = (v30 < v18)
    v32 = D2
    v33 = tl.full((FRAGMENT_D2,), v32, tl.int64)
    v34 = (v30 < v33)
    v35 = tl.full((FRAGMENT_D2,), 0.0, tl.float32)
    v36 = 0.0
    v37 = (v16 + tl.arange(0, FRAGMENT_D2) * 1)
    v38 = (v37 < v18)
    v39 = D1
    v40 = (v13 < v39)
    v41 = (v13 >= 0)
    v42 = (v41 & v40)
    v43 = tl.full((FRAGMENT_D2,), v42, tl.int1)
    v44 = D2
    v45 = tl.full((FRAGMENT_D2,), v44, tl.int64)
    v46 = (v37 < v45)
    v47 = (v43 & v46)
    v48 = (v34 & v31)
    v49 = (v28 & v19)
    v50 = (v47 & v38)
    v51 = tl.load((input + (v13) * S0_0 + (tl.broadcast_to(v17, (FRAGMENT_D2, ))) * S0_1), mask=v49, other=v29)
    v52 = tl.load(tl.make_block_ptr(base=(bias + tl.cast(v16, tl.int64) * S1_0), shape=((D2 - tl.cast(v16, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v53 = (v51 + v52)
    v54 = tl.maximum(v53, v36, propagate_nan=tl.PropagateNan.ALL)
    tl.store((output + (v13) * S2_0 + (tl.broadcast_to(v37, (FRAGMENT_D2, ))) * S2_1), v54, mask=v50)

def launch(input, bias, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = bias.stride(0)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D2, META["FRAGMENT_D2"]), D1)
    return _intent_kernel[grid](input, bias, output, D1, D2, S0_0, S0_1, S1_0, S2_0, S2_1)

def run(input, bias):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, bias, output)
    return output
