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
    key=["D1", "D3", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, REDUCE_CHUNK_4_A0: tl.constexpr):
    v1 = tl.program_id(0)
    v2 = (v1 % 1)
    v3 = D1
    v4 = -float("inf")
    for iv5 in range(0, D1, REDUCE_CHUNK_4_A0):
        v6 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v7 = D1
        v8 = (v6 < v7)
        v9 = v8
        v10 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v11 = (v10 < v7)
        v12 = v11
        v13 = tl.full((REDUCE_CHUNK_4_A0,), v3, tl.int64)
        v14 = (v6 < v13)
        v15 = (v12 & v14)
        v16 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v17 = -float("inf")
        v18 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv5, tl.int64) * S0_0), shape=((D1 - tl.cast(iv5, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v19 = tl.where(v9, v18, v17)
        v20 = tl.reduce(v19, axis=0, combine_fn=_intent_reduce_0)
        v21 = tl.maximum(v4, v20, propagate_nan=tl.PropagateNan.ALL)
        v4 = v21
    v22 = 0.0
    for iv23 in range(0, D1, REDUCE_CHUNK_4_A0):
        v24 = (iv23 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v25 = D1
        v26 = (v24 < v25)
        v27 = v26
        v28 = (iv23 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v29 = (v28 < v25)
        v30 = v29
        v31 = tl.full((REDUCE_CHUNK_4_A0,), v3, tl.int64)
        v32 = (v24 < v31)
        v33 = (v30 & v32)
        v34 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v35 = v4
        v36 = 0.0
        v37 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv23, tl.int64) * S0_0), shape=((D1 - tl.cast(iv23, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v38 = (v37 - v35)
        v39 = libdevice.exp(tl.cast(v38, tl.float32))
        v40 = tl.where(v27, v39, v36)
        v41 = tl.sum(v40, axis=0)
        v42 = (v22 + v41)
        v22 = v42
    v43 = libdevice.log(tl.cast(v22, tl.float32))
    v44 = (v43 + v4)
    v45 = v44
    v46 = (0 + tl.arange(0, 1) * 1)
    v47 = tl.full((1,), 1, tl.int64)
    v48 = (v46 < v47)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(0, tl.int64) * S1_0), shape=((1 - tl.cast(0, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(1,), order=(0,)), tl.cast(v45, tl.float32))

def launch(input, output):
    D1 = input.shape[0]
    D3 = output.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, output, D1, D3, S0_0, S1_0)

def run(input):
    output = torch.empty((1,), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
