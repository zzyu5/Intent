import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("values", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_1_A1": 512}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_1_A1": 512}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_1_A1": 512}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 16, "REDUCE_CHUNK_1_A1": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 16, "REDUCE_CHUNK_1_A1": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 16, "REDUCE_CHUNK_1_A1": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_1_A1": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_1_A1": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_1_A1": 256}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "S0_0", "S0_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(values, output, D1: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, REDUCE_CHUNK_1_A1: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = D1
    v3 = D1
    v4 = 0.0
    for iv5 in range(0, D1, REDUCE_CHUNK_1_A0):
        v6 = (iv5 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v7 = tl.full((REDUCE_CHUNK_1_A0,), v2, tl.int64)
        v8 = (v6 < v7)
        v9 = (iv5 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v10 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v11 = (v9 < v10)
        v12 = tl.full((REDUCE_CHUNK_1_A0,), 0.0, tl.float32)
        v13 = v12
        for iv14 in range(0, D1, REDUCE_CHUNK_1_A1):
            v15 = (iv14 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
            v16 = tl.full((REDUCE_CHUNK_1_A1,), D1, tl.int64)
            v17 = (v15 < v16)
            v18 = tl.broadcast_to(v17[None, :], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v19 = (iv14 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
            v20 = (v19 < v16)
            v21 = tl.broadcast_to(v20[None, :], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v22 = tl.full((REDUCE_CHUNK_1_A1,), v3, tl.int64)
            v23 = (v15 < v22)
            v24 = tl.broadcast_to(v23[None, :], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v25 = tl.broadcast_to(v8[:, None], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v26 = (v25 & v24)
            v27 = (v21 & v26)
            v28 = tl.full((REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1), 0.0, tl.float32)
            v29 = tl.full((REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1), 0.0, tl.float32)
            v30 = tl.broadcast_to(v11[:, None], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v31 = tl.broadcast_to(v12[:, None], (REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1))
            v32 = tl.load(tl.make_block_ptr(base=(values + tl.cast(iv5, tl.int64) * S0_0 + tl.cast(iv14, tl.int64) * S0_1), shape=((D1 - tl.cast(iv5, tl.int64)), (D1 - tl.cast(iv14, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_1_A0, REDUCE_CHUNK_1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
            v33 = tl.where(v30, v32, v29)
            v34 = tl.where(v18, v33, v31)
            v35 = tl.sum(v34, axis=1)
            v36 = (v13 + v35)
            v13 = v36
        v37 = tl.sum(v13, axis=0)
        v38 = (v4 + v37)
        v4 = v38
    tl.store((output), v4)

def launch(values, output):
    D1 = values.shape[0]
    S0_0 = values.stride(0)
    S0_1 = values.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](values, output, D1, S0_0, S0_1)

def run(values):
    output = torch.empty((), device=values.device, dtype=torch.float32)
    launch(values, output)
    return output
