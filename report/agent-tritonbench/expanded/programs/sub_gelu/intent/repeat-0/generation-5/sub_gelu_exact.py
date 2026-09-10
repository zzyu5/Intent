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
    v21 = 0.0
    v22 = (v19 & v16)
    v23 = (v13 & v10)
    v24 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v25 = tl.load(tl.make_block_ptr(base=(other + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v26 = (v24 - v25)
    v27 = (v26 >= v21)
    v28 = tl.cast(v27, tl.float32)
    v29 = 2.0
    v30 = (v28 * v29)
    v31 = 1.0
    v32 = (v30 - v31)
    v33 = 0.23164190351963043
    v34 = 1.0
    v35 = 1.0
    v36 = 1.3302744626998901
    v37 = 1.8212559223175049
    v38 = 1.7814779281616211
    v39 = 0.35656377673149109
    v40 = 0.31938153505325317
    v41 = -0.5
    v42 = 1.4426950216293335
    v43 = 0.39894229173660278
    v44 = 1.0
    v45 = 0.5
    v46 = 0.5
    v47 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v48 = (v47 < v9)
    v49 = D1
    v50 = tl.full((FRAGMENT_D1,), v49, tl.int64)
    v51 = (v47 < v50)
    v52 = (v51 & v48)
    v53 = (v26 * v32)
    v54 = (v33 * v53)
    v55 = (v34 + v54)
    v56 = tl.fdiv(tl.cast(v35, tl.float32), tl.cast(v55, tl.float32), ieee_rounding=True)
    v57 = (v36 * v56)
    v58 = (v57 - v37)
    v59 = (v58 * v56)
    v60 = (v59 + v38)
    v61 = (v60 * v56)
    v62 = (v61 - v39)
    v63 = (v62 * v56)
    v64 = (v63 + v40)
    v65 = (v64 * v56)
    v66 = (v41 * v53)
    v67 = (v66 * v53)
    v68 = (v67 * v42)
    v69 = libdevice.exp2(tl.cast(v68, tl.float32))
    v70 = (v43 * v69)
    v71 = (v70 * v65)
    v72 = (v44 - v71)
    v73 = (v72 - v45)
    v74 = (v32 * v73)
    v75 = (v46 + v74)
    v76 = (v26 * v75)
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
