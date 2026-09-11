import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"REDUCE_CHUNK_5_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_5_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, other, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, REDUCE_CHUNK_5_A0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = D1
    v3 = D1
    v4 = 0.0
    for iv5 in range(0, D1, REDUCE_CHUNK_5_A0):
        v6 = (iv5 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v7 = tl.full((REDUCE_CHUNK_5_A0,), D1, tl.int64)
        v8 = (v6 < v7)
        v9 = v8
        v10 = (iv5 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v11 = (v10 < v7)
        v12 = v11
        v13 = (v9 & v12)
        v14 = (iv5 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v15 = (v14 < v7)
        v16 = v15
        v17 = tl.full((REDUCE_CHUNK_5_A0,), v2, tl.int64)
        v18 = (v6 < v17)
        v19 = (v16 & v18)
        v20 = tl.full((REDUCE_CHUNK_5_A0,), 0.0, tl.float32)
        v21 = (iv5 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v22 = (v21 < v7)
        v23 = v22
        v24 = tl.full((REDUCE_CHUNK_5_A0,), v3, tl.int64)
        v25 = (v10 < v24)
        v26 = (v23 & v25)
        v27 = tl.full((REDUCE_CHUNK_5_A0,), 0.0, tl.float32)
        v28 = tl.full((REDUCE_CHUNK_5_A0,), 0.0, tl.float32)
        v29 = tl.load(tl.make_block_ptr(base=(other + tl.cast(iv5, tl.int64) * S1_0), shape=((D1 - tl.cast(iv5, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_5_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v30 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv5, tl.int64) * S0_0), shape=((D1 - tl.cast(iv5, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_5_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v31 = (v30 + v29)
        v32 = tl.where(v13, v31, v28)
        v33 = tl.sum(v32, axis=0)
        v34 = (v4 + v33)
        v4 = v34
    v35 = tl.cast(D1, tl.float32)
    v36 = tl.fdiv(tl.cast(v4, tl.float32), tl.cast(v35, tl.float32), ieee_rounding=True)
    tl.store((output), v36)

def launch(input, other, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = other.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, other, output, D1, S0_0, S1_0)

def run(input, other):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, other, output)
    return output
