import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 1, "REDUCE_CHUNK_1_A1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "REDUCE_CHUNK_1_A1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "REDUCE_CHUNK_1_A1": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "REDUCE_CHUNK_1_A1": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "REDUCE_CHUNK_1_A1": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "REDUCE_CHUNK_1_A1": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "REDUCE_CHUNK_1_A1": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "REDUCE_CHUNK_1_A1": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "REDUCE_CHUNK_1_A1": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "REDUCE_CHUNK_1_A1": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "REDUCE_CHUNK_1_A1": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "REDUCE_CHUNK_1_A1": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, REDUCE_CHUNK_1_A1: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = tl.cast(D2, tl.float32)
    v6 = (v4 * FRAGMENT_D1)
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = (v8 + tl.arange(0, FRAGMENT_D1) * 1)
    v10 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v11 = (v9 < v10)
    v12 = D1
    v13 = tl.full((FRAGMENT_D1,), v12, tl.int64)
    v14 = (v9 < v13)
    v15 = D2
    v16 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v17 = v16
    for iv18 in range(0, D2, REDUCE_CHUNK_1_A1):
        v19 = (iv18 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v20 = tl.full((REDUCE_CHUNK_1_A1,), D2, tl.int64)
        v21 = (v19 < v20)
        v22 = tl.broadcast_to(v21[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v23 = (iv18 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v24 = (v23 < v20)
        v25 = tl.broadcast_to(v24[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v26 = tl.full((REDUCE_CHUNK_1_A1,), v15, tl.int64)
        v27 = (v19 < v26)
        v28 = tl.broadcast_to(v27[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v29 = tl.broadcast_to(v14[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v30 = (v29 & v28)
        v31 = tl.broadcast_to(v11[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v32 = (v30 & v31)
        v33 = (v25 & v32)
        v34 = tl.full((FRAGMENT_D1, REDUCE_CHUNK_1_A1), 0.0, tl.float32)
        v35 = tl.broadcast_to(v16[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v36 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v8, tl.int64) * S0_0 + tl.cast(iv18, tl.int64) * S0_1), shape=((D1 - tl.cast(v8, tl.int64)), (D2 - tl.cast(iv18, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, REDUCE_CHUNK_1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v37 = tl.where(v22, v36, v35)
        v38 = tl.sum(v37, axis=1)
        v39 = (v17 + v38)
        v17 = v39
    v40 = tl.full((FRAGMENT_D1,), v5, tl.float32)
    v41 = tl.fdiv(tl.cast(v17, tl.float32), tl.cast(v40, tl.float32), ieee_rounding=True)
    v42 = (v6 * 1)
    v43 = (0 + v42)
    v44 = (v43 + tl.arange(0, FRAGMENT_D1) * 1)
    v45 = (v44 < v10)
    v46 = D1
    v47 = tl.full((FRAGMENT_D1,), v46, tl.int64)
    v48 = (v44 < v47)
    v49 = (v48 & v45)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v43, tl.int64) * S1_0), shape=((D1 - tl.cast(v43, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v41, tl.float32), boundary_check=(0,))

def launch(input, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, D2, S0_0, S0_1, S1_0)

def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
