import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1, a2, a3):
    v0 = (a0 > a2)
    v1 = (a0 == a2)
    v2 = (a1 <= a3)
    v3 = (v1 & v2)
    v4 = (v0 | v3)
    v5 = (a0 != a0)
    v6 = (a2 != a2)
    v7 = tl.where(v6, v2, True)
    v8 = tl.where(v5, v7, v4)
    v9 = tl.where(v8, a0, a2)
    v10 = tl.where(v8, a1, a3)
    return (v9, v10)

_intent_tuning_hooks = TuningHooks(("input", "values", "indices", ), (False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 4096}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 2048}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 2048}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 2048}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D3", "D4", "S0_0", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, values, indices, D1: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr):
    v11 = tl.program_id(0)
    v12 = (v11 % 1)
    v13 = D1
    v14 = -float("inf")
    v15 = 9223372036854775807
    for iv16 in range(0, D1, REDUCE_CHUNK_1_A0):
        v17 = (iv16 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v18 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v19 = (v17 < v18)
        v20 = v19
        v21 = (iv16 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v22 = (v21 < v18)
        v23 = v22
        v24 = tl.full((REDUCE_CHUNK_1_A0,), v13, tl.int64)
        v25 = (v17 < v24)
        v26 = (v23 & v25)
        v27 = tl.full((REDUCE_CHUNK_1_A0,), 0.0, tl.float32)
        v28 = tl.full((REDUCE_CHUNK_1_A0,), -float("inf"), tl.float32)
        v29 = (iv16 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v30 = (v29 < v18)
        v31 = v30
        v32 = tl.full((REDUCE_CHUNK_1_A0,), 0, tl.int64)
        v33 = (v29 - v32)
        v34 = tl.full((REDUCE_CHUNK_1_A0,), 1, tl.int64)
        v35 = ((v33 // v34) - (((v33 % v34) != 0) & (((v33 % v34) < 0) != (v34 < 0))))
        v36 = v35
        v37 = tl.cast(v36, tl.int64)
        v38 = tl.full((REDUCE_CHUNK_1_A0,), 9223372036854775807, tl.int64)
        v39 = tl.where(v31, v37, v38)
        v40 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv16, tl.int64) * S0_0), shape=((D1 - tl.cast(iv16, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_1_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v41 = tl.where(v20, v40, v28)
        v42, v43 = tl.reduce((v41, v39), axis=0, combine_fn=_intent_reduce_0)
        v44 = (v14 > v42)
        v45 = (v14 == v42)
        v46 = (v15 <= v43)
        v47 = (v45 & v46)
        v48 = (v44 | v47)
        v49 = (v14 != v14)
        v50 = (v42 != v42)
        v51 = tl.where(v50, v46, True)
        v52 = tl.where(v49, v51, v48)
        v53 = tl.where(v52, v14, v42)
        v54 = tl.where(v52, v15, v43)
        v14 = v53
        v15 = v54
    v55 = tl.full((1,), v14, tl.float32)
    v56 = (0 + tl.arange(0, 1) * 1)
    v57 = tl.full((1,), 1, tl.int64)
    v58 = (v56 < v57)
    tl.store(tl.make_block_ptr(base=(values + tl.cast(0, tl.int64) * S1_0), shape=((1 - tl.cast(0, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(1,), order=(0,)), tl.cast(v55, tl.float32))
    v59 = tl.cast(v15, tl.int64)
    v60 = tl.full((1,), v59, tl.int64)
    v61 = (0 + tl.arange(0, 1) * 1)
    v62 = tl.full((1,), 1, tl.int64)
    v63 = (v61 < v62)
    tl.store(tl.make_block_ptr(base=(indices + tl.cast(0, tl.int64) * S2_0), shape=((1 - tl.cast(0, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(1,), order=(0,)), tl.cast(v60, tl.int64))

def launch(input, values, indices):
    D1 = input.shape[0]
    D3 = values.shape[0]
    D4 = indices.shape[0]
    S0_0 = input.stride(0)
    S1_0 = values.stride(0)
    S2_0 = indices.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, values, indices, D1, D3, D4, S0_0, S1_0, S2_0)

def run(input):
    values = torch.empty((1,), device=input.device, dtype=torch.float32)
    indices = torch.empty((1,), device=input.device, dtype=torch.int64)
    launch(input, values, indices)
    return (values, indices)
