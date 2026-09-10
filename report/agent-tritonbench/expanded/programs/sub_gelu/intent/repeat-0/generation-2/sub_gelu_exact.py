import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

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
    key=["D1", "S0_0", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, other, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v15 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v16 = (v15 < v9)
    v17 = D1
    v18 = tl.full((FRAGMENT_D1,), v17, tl.int64)
    v19 = (v15 < v18)
    v20 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v21 = 0.70710676908493042
    v22 = 0.32759109139442444
    v23 = 1.0
    v24 = 1.0
    v25 = 1.0614054203033447
    v26 = 1.453152060508728
    v27 = 1.421413779258728
    v28 = 0.2844967246055603
    v29 = 0.25482958555221558
    v30 = 1.4426950216293335
    v31 = 1.0
    v32 = 0.0
    v33 = (v19 & v16)
    v34 = (v13 & v10)
    v35 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v36 = tl.load(tl.make_block_ptr(base=(other + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v37 = (v35 - v36)
    v38 = (v37 * v21)
    v39 = (v38 >= v32)
    v40 = tl.cast(v39, tl.float32)
    v41 = 2.0
    v42 = (v40 * v41)
    v43 = 1.0
    v44 = (v42 - v43)
    v45 = 0.5
    v46 = 1.0
    v47 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v48 = (v47 < v9)
    v49 = D1
    v50 = tl.full((FRAGMENT_D1,), v49, tl.int64)
    v51 = (v47 < v50)
    v52 = (v51 & v48)
    v53 = (v45 * v37)
    v54 = (-v38)
    v55 = tl.maximum(v38, v54, propagate_nan=tl.PropagateNan.ALL)
    v56 = (v22 * v55)
    v57 = (v23 + v56)
    v58 = tl.fdiv(tl.cast(v24, tl.float32), tl.cast(v57, tl.float32), ieee_rounding=True)
    v59 = (v25 * v58)
    v60 = (v59 - v26)
    v61 = (v60 * v58)
    v62 = (v61 + v27)
    v63 = (v62 * v58)
    v64 = (v63 - v28)
    v65 = (v64 * v58)
    v66 = (v65 + v29)
    v67 = (v66 * v58)
    v68 = (-v55)
    v69 = (v68 * v55)
    v70 = (v69 * v30)
    v71 = libdevice.exp2(tl.cast(v70, tl.float32))
    v72 = (v67 * v71)
    v73 = (v31 - v72)
    v74 = (v44 * v73)
    v75 = (v46 + v74)
    v76 = (v53 * v75)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v7, tl.int64) * S2_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v76, tl.float32), boundary_check=(0,))

def launch(input, other, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = other.stride(0)
    S2_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, other, output, D1, S0_0, S1_0, S2_0)

def run(input, other):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, other, output)
    return output
