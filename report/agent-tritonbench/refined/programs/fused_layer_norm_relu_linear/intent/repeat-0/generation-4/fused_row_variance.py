import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "mean", "output", ), (False, False, True, ))

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
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, mean, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, REDUCE_CHUNK_1_A1: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v15 = (v5 * 1)
    v16 = (0 + v15)
    v17 = (v16 + tl.arange(0, FRAGMENT_D1) * 1)
    v18 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v19 = (v17 < v18)
    v20 = D1
    v21 = tl.full((FRAGMENT_D1,), v20, tl.int64)
    v22 = (v17 < v21)
    v23 = D2
    v24 = tl.cast(D2, tl.float32)
    v25 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v26 = (v13 & v10)
    v27 = tl.load(tl.make_block_ptr(base=(mean + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v28 = tl.reshape(v27, (FRAGMENT_D1, 1), can_reorder=False)
    v29 = v25
    for iv30 in range(0, D2, REDUCE_CHUNK_1_A1):
        v31 = (iv30 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v32 = tl.full((REDUCE_CHUNK_1_A1,), D2, tl.int64)
        v33 = (v31 < v32)
        v34 = tl.broadcast_to(v33[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v35 = (iv30 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v36 = (v35 < v32)
        v37 = tl.broadcast_to(v36[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v38 = tl.full((REDUCE_CHUNK_1_A1,), v23, tl.int64)
        v39 = (v31 < v38)
        v40 = tl.broadcast_to(v39[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v41 = tl.broadcast_to(v22[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v42 = (v41 & v40)
        v43 = tl.broadcast_to(v19[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v44 = (v42 & v43)
        v45 = (v37 & v44)
        v46 = tl.full((FRAGMENT_D1, REDUCE_CHUNK_1_A1), 0.0, tl.float32)
        v47 = tl.broadcast_to(v28, (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v48 = tl.broadcast_to(v25[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v49 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v16, tl.int64) * S0_0 + tl.cast(iv30, tl.int64) * S0_1), shape=((D1 - tl.cast(v16, tl.int64)), (D2 - tl.cast(iv30, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, REDUCE_CHUNK_1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v50 = (v49 - v47)
        v51 = (v50 * v50)
        v52 = tl.where(v34, v51, v48)
        v53 = tl.sum(v52, axis=1)
        v54 = (v29 + v53)
        v29 = v54
    v55 = tl.full((FRAGMENT_D1,), v24, tl.float32)
    v56 = tl.fdiv(tl.cast(v29, tl.float32), tl.cast(v55, tl.float32), ieee_rounding=True)
    v57 = (v5 * 1)
    v58 = (0 + v57)
    v59 = (v58 + tl.arange(0, FRAGMENT_D1) * 1)
    v60 = (v59 < v18)
    v61 = D1
    v62 = tl.full((FRAGMENT_D1,), v61, tl.int64)
    v63 = (v59 < v62)
    v64 = (v63 & v60)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v58, tl.int64) * S2_0), shape=((D1 - tl.cast(v58, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v56, tl.float32), boundary_check=(0,))

def launch(input, mean, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = mean.stride(0)
    S2_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, mean, output, D1, D2, S0_0, S0_1, S1_0, S2_0)

def run(input, mean):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, mean, output)
    return output
