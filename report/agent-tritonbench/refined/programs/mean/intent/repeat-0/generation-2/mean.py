import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input_tensor", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"REDUCE_CHUNK_1_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"REDUCE_CHUNK_1_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "S0_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input_tensor, output, D1: tl.constexpr, S0_0: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = D1
    v3 = 0.0
    for iv4 in range(0, D1, REDUCE_CHUNK_1_A0):
        v5 = (iv4 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v6 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v7 = (v5 < v6)
        v8 = v7
        v9 = (iv4 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v10 = (v9 < v6)
        v11 = v10
        v12 = tl.full((REDUCE_CHUNK_1_A0,), v2, tl.int64)
        v13 = (v5 < v12)
        v14 = (v11 & v13)
        v15 = tl.full((REDUCE_CHUNK_1_A0,), 0.0, tl.float32)
        v16 = tl.full((REDUCE_CHUNK_1_A0,), 0.0, tl.float32)
        v17 = tl.load(tl.make_block_ptr(base=(input_tensor + tl.cast(iv4, tl.int64) * S0_0), shape=((D1 - tl.cast(iv4, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(REDUCE_CHUNK_1_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v18 = tl.where(v8, v17, v16)
        v19 = tl.sum(v18, axis=0)
        v20 = (v3 + v19)
        v3 = v20
    v21 = tl.cast(D1, tl.float32)
    v22 = tl.fdiv(tl.cast(v3, tl.float32), tl.cast(v21, tl.float32), ieee_rounding=True)
    tl.store((output), v22)

def launch(input_tensor, output):
    D1 = input_tensor.shape[0]
    S0_0 = input_tensor.stride(0)
    grid = lambda META: (1,)
    return _intent_kernel[grid](input_tensor, output, D1, S0_0)

def run(input_tensor):
    output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
    launch(input_tensor, output)
    return output
