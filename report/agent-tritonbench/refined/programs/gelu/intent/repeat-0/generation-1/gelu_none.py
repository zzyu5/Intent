import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("x", "output", ), (False, True, ))

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
def _intent_kernel(x, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * FRAGMENT_D1)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v9 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v10 = (v8 < v9)
    v11 = D1
    v12 = tl.full((FRAGMENT_D1,), v11, tl.int64)
    v13 = (v8 < v12)
    v14 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v15 = tl.full((FRAGMENT_D1,), 0.70710676908493042, tl.float32)
    v16 = tl.full((FRAGMENT_D1,), 0.32759109139442444, tl.float32)
    v17 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v18 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v19 = tl.full((FRAGMENT_D1,), 1.0614054203033447, tl.float32)
    v20 = tl.full((FRAGMENT_D1,), 1.453152060508728, tl.float32)
    v21 = tl.full((FRAGMENT_D1,), 1.421413779258728, tl.float32)
    v22 = tl.full((FRAGMENT_D1,), 0.2844967246055603, tl.float32)
    v23 = tl.full((FRAGMENT_D1,), 0.25482958555221558, tl.float32)
    v24 = tl.full((FRAGMENT_D1,), 1.4426950216293335, tl.float32)
    v25 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v26 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v27 = (v13 & v10)
    v28 = tl.load(tl.make_block_ptr(base=(x + tl.cast(v7, tl.int64) * S0_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v29 = (v28 * v15)
    v30 = (v29 >= v26)
    v31 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v32 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v33 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v34 = (v33 < v9)
    v35 = D1
    v36 = tl.full((FRAGMENT_D1,), v35, tl.int64)
    v37 = (v33 < v36)
    v38 = (v37 & v34)
    v39 = (v31 * v28)
    v40 = (-v29)
    v41 = tl.maximum(v29, v40, propagate_nan=tl.PropagateNan.ALL)
    v42 = (v16 * v41)
    v43 = (v17 + v42)
    v44 = tl.fdiv(tl.cast(v18, tl.float32), tl.cast(v43, tl.float32), ieee_rounding=True)
    v45 = (v19 * v44)
    v46 = (v45 - v20)
    v47 = (v46 * v44)
    v48 = (v47 + v21)
    v49 = (v48 * v44)
    v50 = (v49 - v22)
    v51 = (v50 * v44)
    v52 = (v51 + v23)
    v53 = (v52 * v44)
    v54 = (-v41)
    v55 = (v54 * v41)
    v56 = (v55 * v24)
    v57 = libdevice.exp2(tl.cast(v56, tl.float32))
    v58 = (v53 * v57)
    v59 = (v25 - v58)
    v60 = (-v59)
    v61 = tl.where(v30, v59, v60)
    v62 = (v32 + v61)
    v63 = (v39 * v62)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v63, tl.float32), boundary_check=(0,))

def launch(x, output):
    D1 = x.shape[0]
    S0_0 = x.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](x, output, D1, S0_0, S1_0)

def run(x):
    output = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
    launch(x, output)
    return output
