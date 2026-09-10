import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0", "S1_1", "S1_2", "S1_3"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S1_3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * FRAGMENT_D1)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v9 = (v5 * 1)
    v10 = (0 + v9)
    v11 = (v10 + tl.arange(0, FRAGMENT_D1) * 1)
    v12 = (v5 * 1)
    v13 = (0 + v12)
    v14 = (v13 + tl.arange(0, FRAGMENT_D1) * 1)
    v15 = (v5 * 1)
    v16 = (0 + v15)
    v17 = (v16 + tl.arange(0, FRAGMENT_D1) * 1)
    v18 = D1
    v19 = tl.full((FRAGMENT_D1,), v18, tl.int64)
    v20 = (v8 < v19)
    v21 = v20[None, None, None, :]
    v22 = D1
    v23 = tl.full((FRAGMENT_D1,), v22, tl.int64)
    v24 = (v11 < v23)
    v25 = v24[None, None, None, :]
    v26 = (v21 & v25)
    v27 = D1
    v28 = tl.full((FRAGMENT_D1,), v27, tl.int64)
    v29 = (v14 < v28)
    v30 = v29[None, None, None, :]
    v31 = (v26 & v30)
    v32 = D1
    v33 = tl.full((FRAGMENT_D1,), v32, tl.int64)
    v34 = (v17 < v33)
    v35 = v34[None, None, None, :]
    v36 = (v31 & v35)
    v37 = tl.full((FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1), 0.0, tl.float32)
    v38 = (v16 + tl.arange(0, FRAGMENT_D1) * 1)
    v39 = (v13 + tl.arange(0, FRAGMENT_D1) * 1)
    v40 = (v10 + tl.arange(0, FRAGMENT_D1) * 1)
    v41 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v42 = D1
    v43 = tl.full((FRAGMENT_D1,), v42, tl.int64)
    v44 = (v38 < v43)
    v45 = v44[None, None, None, :]
    v46 = D1
    v47 = tl.full((FRAGMENT_D1,), v46, tl.int64)
    v48 = (v39 < v47)
    v49 = v48[None, None, None, :]
    v50 = (v45 & v49)
    v51 = D1
    v52 = tl.full((FRAGMENT_D1,), v51, tl.int64)
    v53 = (v40 < v52)
    v54 = v53[None, None, None, :]
    v55 = (v50 & v54)
    v56 = D1
    v57 = tl.full((FRAGMENT_D1,), v56, tl.int64)
    v58 = (v41 < v57)
    v59 = v58[None, None, None, :]
    v60 = (v55 & v59)
    v61 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0 + tl.cast(v10, tl.int64) * S0_1 + tl.cast(v13, tl.int64) * S0_2 + tl.cast(v16, tl.int64) * S0_3), shape=((D1 - tl.cast(v7, tl.int64)), (D1 - tl.cast(v10, tl.int64)), (D1 - tl.cast(v13, tl.int64)), (D1 - tl.cast(v16, tl.int64))), strides=(S0_0, S0_1, S0_2, S0_3), offsets=(0, 0, 0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1), order=(3, 2, 1, 0)), boundary_check=(0, 1, 2, 3), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v16, tl.int64) * S1_0 + tl.cast(v13, tl.int64) * S1_1 + tl.cast(v10, tl.int64) * S1_2 + tl.cast(v7, tl.int64) * S1_3), shape=((D1 - tl.cast(v16, tl.int64)), (D1 - tl.cast(v13, tl.int64)), (D1 - tl.cast(v10, tl.int64)), (D1 - tl.cast(v7, tl.int64))), strides=(S1_0, S1_1, S1_2, S1_3), offsets=(0, 0, 0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1, FRAGMENT_D1), order=(3, 2, 1, 0)), tl.cast(v61, tl.float32), boundary_check=(0, 1, 2, 3))

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S0_2 = input.stride(2)
    S0_3 = input.stride(3)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    S1_2 = output.stride(2)
    S1_3 = output.stride(3)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, S0_0, S0_1, S0_2, S0_3, S1_0, S1_1, S1_2, S1_3)

def run(input):
    output = torch.empty((input.shape[0], input.shape[0], input.shape[0], input.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
