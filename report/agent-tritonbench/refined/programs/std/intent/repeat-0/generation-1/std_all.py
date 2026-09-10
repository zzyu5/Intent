import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

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
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = D1
    v3 = tl.cast(D1, tl.float32)
    v4 = 0.0
    for iv5 in range(0, D1, REDUCE_CHUNK_4_A0):
        v6 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v7 = tl.full((REDUCE_CHUNK_4_A0,), D1, tl.int64)
        v8 = (v6 < v7)
        v9 = v8
        v10 = (iv5 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v11 = (v10 < v7)
        v12 = v11
        v13 = tl.full((REDUCE_CHUNK_4_A0,), v2, tl.int64)
        v14 = (v6 < v13)
        v15 = (v12 & v14)
        v16 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v17 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v18 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv5, tl.int64) * S0_0), shape=((D1 - tl.cast(iv5, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v19 = tl.where(v9, v18, v17)
        v20 = tl.sum(v19, axis=0)
        v21 = (v4 + v20)
        v4 = v21
    v22 = tl.fdiv(tl.cast(v4, tl.float32), tl.cast(v3, tl.float32), ieee_rounding=True)
    v23 = 0.0
    for iv24 in range(0, D1, REDUCE_CHUNK_4_A0):
        v25 = (iv24 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v26 = tl.full((REDUCE_CHUNK_4_A0,), D1, tl.int64)
        v27 = (v25 < v26)
        v28 = v27
        v29 = (iv24 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v30 = (v29 < v26)
        v31 = v30
        v32 = tl.full((REDUCE_CHUNK_4_A0,), v2, tl.int64)
        v33 = (v25 < v32)
        v34 = (v31 & v33)
        v35 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v36 = tl.full((REDUCE_CHUNK_4_A0,), v22, tl.float32)
        v37 = tl.full((REDUCE_CHUNK_4_A0,), 0.0, tl.float32)
        v38 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv24, tl.int64) * S0_0), shape=((D1 - tl.cast(iv24, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v39 = (v38 - v36)
        v40 = (v39 * v39)
        v41 = tl.where(v28, v40, v37)
        v42 = tl.sum(v41, axis=0)
        v43 = (v23 + v42)
        v23 = v43
    v44 = (v3 - 1.0)
    v45 = tl.fdiv(tl.cast(v23, tl.float32), tl.cast(v44, tl.float32), ieee_rounding=True)
    v46 = libdevice.pow(v45, 0.5)
    tl.store((output), v46)

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input, output, D1, S0_0)

def run(input):
    output = torch.empty((), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
