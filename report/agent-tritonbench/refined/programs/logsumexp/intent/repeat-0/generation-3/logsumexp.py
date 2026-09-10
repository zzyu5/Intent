import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1):
    v0 = tl.maximum(a0, a1, propagate_nan=tl.PropagateNan.ALL)
    return v0

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"REDUCE_CHUNK_4_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_4_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "S0_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, S0_0: tl.constexpr, REDUCE_CHUNK_4_A0: tl.constexpr):
    v1 = tl.program_id(0)
    v2 = (v1 % 1)
    v3 = D1
    v4 = -float("inf")
    for iv5 in range(0, D1, REDUCE_CHUNK_4_A0):
        v6 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v7 = tl.full((REDUCE_CHUNK_4_A0,), D1, tl.int64)
        v8 = (v6 < v7)
        v9 = v8
        v10 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v11 = (v10 < v7)
        v12 = v11
        v13 = tl.full((REDUCE_CHUNK_4_A0,), v3, tl.int64)
        v14 = (v6 < v13)
        v15 = (v12 & v14)
        v16 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v17 = tl.full((REDUCE_CHUNK_4_A0,), -float("inf"), tl.float32)
        v18 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv5, tl.int64) * S0_0), shape=((D1 - tl.cast(iv5, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v19 = tl.where(v9, v18, v17)
        v20 = tl.reduce(v19, axis=0, combine_fn=_intent_reduce_0)
        v21 = tl.maximum(v4, v20, propagate_nan=tl.PropagateNan.ALL)
        v4 = v21
    v22 = 0.0
    for iv23 in range(0, D1, REDUCE_CHUNK_4_A0):
        v24 = (iv23 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v25 = tl.full((REDUCE_CHUNK_4_A0,), D1, tl.int64)
        v26 = (v24 < v25)
        v27 = v26
        v28 = (iv23 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v29 = (v28 < v25)
        v30 = v29
        v31 = tl.full((REDUCE_CHUNK_4_A0,), v3, tl.int64)
        v32 = (v24 < v31)
        v33 = (v30 & v32)
        v34 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v35 = tl.full((REDUCE_CHUNK_4_A0,), v4, tl.float32)
        v36 = tl.full((REDUCE_CHUNK_4_A0,), 1.4426950216293335, tl.float32)
        v37 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v38 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv23, tl.int64) * S0_0), shape=((D1 - tl.cast(iv23, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v39 = (v38 - v35)
        v40 = (v39 * v36)
        v41 = libdevice.exp2(tl.cast(v40, tl.float32))
        v42 = tl.where(v27, v41, v37)
        v43 = tl.sum(v42, axis=0)
        v44 = (v22 + v43)
        v22 = v44
    v45 = 0.0
    v46 = v22
    for iv47 in range(0, 20, 1):
        v48 = (v46 >= 2.0)
        v49 = (v46 * 0.5)
        v50 = (v45 + 1.0)
        v51 = tl.where(v48, v50, v45)
        v52 = tl.where(v48, v49, v46)
        v45 = v51
        v46 = v52
    v53 = (v46 - 1.0)
    v54 = (v46 + 1.0)
    v55 = tl.fdiv(tl.cast(v53, tl.float32), tl.cast(v54, tl.float32), ieee_rounding=True)
    v56 = (v55 * v55)
    v57 = (v55 * v56)
    v58 = tl.fdiv(tl.cast(v57, tl.float32), tl.cast(3.0, tl.float32), ieee_rounding=True)
    v59 = (v55 + v58)
    v60 = (v57 * v56)
    v61 = tl.fdiv(tl.cast(v60, tl.float32), tl.cast(5.0, tl.float32), ieee_rounding=True)
    v62 = (v59 + v61)
    v63 = (v60 * v56)
    v64 = tl.fdiv(tl.cast(v63, tl.float32), tl.cast(7.0, tl.float32), ieee_rounding=True)
    v65 = (v62 + v64)
    v66 = (v63 * v56)
    v67 = tl.fdiv(tl.cast(v66, tl.float32), tl.cast(9.0, tl.float32), ieee_rounding=True)
    v68 = (v65 + v67)
    v69 = (v66 * v56)
    v70 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(11.0, tl.float32), ieee_rounding=True)
    v71 = (v68 + v70)
    v72 = (v69 * v56)
    v73 = tl.fdiv(tl.cast(v72, tl.float32), tl.cast(13.0, tl.float32), ieee_rounding=True)
    v74 = (v71 + v73)
    v75 = (v72 * v56)
    v76 = tl.fdiv(tl.cast(v75, tl.float32), tl.cast(15.0, tl.float32), ieee_rounding=True)
    v77 = (v74 + v76)
    v78 = (v75 * v56)
    v79 = tl.fdiv(tl.cast(v78, tl.float32), tl.cast(17.0, tl.float32), ieee_rounding=True)
    v80 = (v77 + v79)
    v81 = (v45 * 0.69314718246459961)
    v82 = (2.0 * v80)
    v83 = (v81 + v82)
    v84 = (v4 + v83)
    tl.store((output), v84)

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, output, D1, S0_0)

def run(input):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
