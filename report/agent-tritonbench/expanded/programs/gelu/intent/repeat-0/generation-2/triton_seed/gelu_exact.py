import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * FRAGMENT_D1)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v9 = D1
    v10 = (v8 < v9)
    v11 = D1
    v12 = tl.full((FRAGMENT_D1,), v11, tl.int64)
    v13 = (v8 < v12)
    v14 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v15 = 0.70710676908493042
    v16 = 0.32759109139442444
    v17 = 1.0
    v18 = 1.0
    v19 = 1.0614054203033447
    v20 = 1.453152060508728
    v21 = 1.421413779258728
    v22 = 0.2844967246055603
    v23 = 0.25482958555221558
    v24 = -1.4426950216293335
    v25 = 1.0
    v26 = 0.0
    v27 = (v13 & v10)
    v28 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v29 = (v15 * v28)
    v30 = (v29 < v26)
    v31 = 0.0
    v32 = (v29 == v31)
    v33 = 0.5
    v34 = 1.0
    v35 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v36 = (v35 < v9)
    v37 = D1
    v38 = tl.full((FRAGMENT_D1,), v37, tl.int64)
    v39 = (v35 < v38)
    v40 = (v39 & v36)
    v41 = (v33 * v28)
    v42 = (-v29)
    v43 = tl.maximum(v29, v42, propagate_nan=tl.PropagateNan.ALL)
    v44 = (v16 * v43)
    v45 = (v17 + v44)
    v46 = tl.fdiv(tl.cast(v18, tl.float32), tl.cast(v45, tl.float32), ieee_rounding=True)
    v47 = (v19 * v46)
    v48 = (v47 - v20)
    v49 = (v48 * v46)
    v50 = (v49 + v21)
    v51 = (v50 * v46)
    v52 = (v51 - v22)
    v53 = (v52 * v46)
    v54 = (v53 + v23)
    v55 = (v54 * v46)
    v56 = (v24 * v29)
    v57 = (v56 * v29)
    v58 = libdevice.exp2(tl.cast(v57, tl.float32))
    v59 = (v55 * v58)
    v60 = (v25 - v59)
    v61 = (-v60)
    v62 = tl.where(v30, v61, v60)
    v63 = tl.where(v32, v29, v62)
    v64 = (v34 + v63)
    v65 = (v41 * v64)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v65, tl.float32), boundary_check=(0,))

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, S0_0, S1_0)

def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
