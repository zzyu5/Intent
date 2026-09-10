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
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_4_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_4_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_4_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_4_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_4_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_4_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_4_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_4_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_4_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, REDUCE_CHUNK_4_A0: tl.constexpr, FRAGMENT_D2: tl.constexpr):
    v1 = (FRAGMENT_D2 - 1)
    v2 = (D2 + v1)
    v3 = ((v2 // FRAGMENT_D2) - (((v2 % FRAGMENT_D2) != 0) & (((v2 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v4 = tl.program_id(0)
    v5 = (v4 % v3)
    v6 = (v5 * FRAGMENT_D2)
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = (v8 + tl.arange(0, FRAGMENT_D2) * 1)
    v10 = D2
    v11 = (v9 < v10)
    v12 = D1
    v13 = D2
    v14 = tl.full((FRAGMENT_D2,), v13, tl.int64)
    v15 = (v9 < v14)
    v16 = tl.full((FRAGMENT_D2,), -float("inf"), tl.float32)
    v17 = v16
    for iv18 in range(0, D1, REDUCE_CHUNK_4_A0):
        v19 = (iv18 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v20 = D1
        v21 = (v19 < v20)
        v22 = v21[:, None]
        v23 = (iv18 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v24 = (v23 < v20)
        v25 = v24[:, None]
        v26 = tl.full((REDUCE_CHUNK_4_A0,), v12, tl.int64)
        v27 = (v19 < v26)
        v28 = v27[:, None]
        v29 = v15[None, :]
        v30 = (v28 & v29)
        v31 = v11[None, :]
        v32 = (v30 & v31)
        v33 = (v25 & v32)
        v34 = tl.full((REDUCE_CHUNK_4_A0, FRAGMENT_D2), 0.0, tl.float32)
        v35 = v16[None, :]
        v36 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv18, tl.int64) * S0_0 + tl.cast(v8, tl.int64) * S0_1), shape=((D1 - tl.cast(iv18, tl.int64)), (D2 - tl.cast(v8, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_4_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v37 = tl.where(v22, v36, v35)
        v38 = tl.reduce(v37, axis=0, combine_fn=_intent_reduce_0)
        v39 = tl.maximum(v17, v38, propagate_nan=tl.PropagateNan.ALL)
        v17 = v39
    v40 = tl.full((FRAGMENT_D2,), 0.0, tl.float32)
    v41 = tl.full((REDUCE_CHUNK_4_A0, FRAGMENT_D2), 0.0, tl.float32)
    v42 = v41
    for iv43 in range(0, D1, REDUCE_CHUNK_4_A0):
        v44 = (iv43 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v45 = D1
        v46 = (v44 < v45)
        v47 = v46[:, None]
        v48 = (iv43 + tl.arange(0, REDUCE_CHUNK_4_A0) * 1)
        v49 = (v48 < v45)
        v50 = v49[:, None]
        v51 = tl.full((REDUCE_CHUNK_4_A0,), v12, tl.int64)
        v52 = (v44 < v51)
        v53 = v52[:, None]
        v54 = v15[None, :]
        v55 = (v53 & v54)
        v56 = v11[None, :]
        v57 = (v55 & v56)
        v58 = (v50 & v57)
        v59 = tl.full((REDUCE_CHUNK_4_A0, FRAGMENT_D2), 0.0, tl.float32)
        v60 = v17[None, :]
        v61 = v40[None, :]
        v62 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv43, tl.int64) * S0_0 + tl.cast(v8, tl.int64) * S0_1), shape=((D1 - tl.cast(iv43, tl.int64)), (D2 - tl.cast(v8, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_4_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v63 = (v62 - v60)
        v64 = libdevice.exp(tl.cast(v63, tl.float32))
        v65 = tl.where(v47, v64, v61)
        v66 = (v42 + v65)
        v42 = v66
    v67 = tl.sum(v42, axis=0)
    v68 = libdevice.log(tl.cast(v67, tl.float32))
    v69 = (v68 + v17)
    v70 = (v8 + tl.arange(0, FRAGMENT_D2) * 1)
    v71 = (v70 < v10)
    v72 = D2
    v73 = tl.full((FRAGMENT_D2,), v72, tl.int64)
    v74 = (v70 < v73)
    v75 = (v74 & v71)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v8, tl.int64) * S1_0), shape=((D2 - tl.cast(v8, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), tl.cast(v69, tl.float32), boundary_check=(0,))

def launch(input, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D2, META["FRAGMENT_D2"]),)
    return _intent_kernel[grid](input, output, D1, D2, S0_0, S0_1, S1_0)

def run(input):
    output = torch.empty((input.shape[1],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
