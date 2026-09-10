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
    v7 = True
    v8 = tl.where(v6, v2, v7)
    v9 = tl.where(v5, v8, v4)
    v10 = tl.where(v9, a0, a2)
    v11 = tl.where(v9, a1, a3)
    return (v10, v11)

_intent_tuning_hooks = TuningHooks(("input", "output_values", "output_indices", ), (False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output_values, output_indices, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v12 = (FRAGMENT_D1 - 1)
    v13 = (D1 + v12)
    v14 = ((v13 // FRAGMENT_D1) - (((v13 % FRAGMENT_D1) != 0) & (((v13 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v15 = tl.program_id(0)
    v16 = (v15 % v14)
    v17 = (v16 * FRAGMENT_D1)
    v18 = (v17 * 1)
    v19 = (0 + v18)
    v20 = (v19 + tl.arange(0, FRAGMENT_D1) * 1)
    v21 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v22 = (v20 < v21)
    v23 = D2
    v24 = D1
    v25 = tl.full((FRAGMENT_D1,), v24, tl.int64)
    v26 = (v20 < v25)
    v27 = tl.full((FRAGMENT_D1,), -float("inf"), tl.float32)
    v28 = tl.full((FRAGMENT_D1,), 9223372036854775807, tl.int64)
    v29 = v27
    v30 = v28
    for iv31 in range(0, D2, REDUCE_CHUNK_1_A0):
        v32 = (iv31 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v33 = tl.full((REDUCE_CHUNK_1_A0,), D2, tl.int64)
        v34 = (v32 < v33)
        v35 = tl.broadcast_to(v34[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v36 = (iv31 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v37 = (v36 < v33)
        v38 = tl.broadcast_to(v37[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v39 = tl.full((REDUCE_CHUNK_1_A0,), v23, tl.int64)
        v40 = (v32 < v39)
        v41 = tl.broadcast_to(v40[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v42 = tl.broadcast_to(v26[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v43 = (v41 & v42)
        v44 = tl.broadcast_to(v22[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v45 = (v43 & v44)
        v46 = (v38 & v45)
        v47 = tl.full((REDUCE_CHUNK_1_A0, FRAGMENT_D1), 0.0, tl.float32)
        v48 = tl.broadcast_to(v27[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v49 = (iv31 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v50 = (v49 < v33)
        v51 = tl.broadcast_to(v50[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v52 = tl.full((REDUCE_CHUNK_1_A0,), 0, tl.int64)
        v53 = (v49 - v52)
        v54 = tl.full((REDUCE_CHUNK_1_A0,), 1, tl.int64)
        v55 = ((v53 // v54) - (((v53 % v54) != 0) & (((v53 % v54) < 0) != (v54 < 0))))
        v56 = tl.broadcast_to(v55[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v57 = tl.cast(v56, tl.int64)
        v58 = tl.broadcast_to(v28[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D1))
        v59 = tl.where(v51, v57, v58)
        v60 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv31, tl.int64) * S0_0 + tl.cast(v19, tl.int64) * S0_1), shape=((D2 - tl.cast(iv31, tl.int64)), (D1 - tl.cast(v19, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_1_A0, FRAGMENT_D1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v61 = (-v60)
        v62 = tl.where(v35, v61, v48)
        v63, v64 = tl.reduce((v62, v59), axis=0, combine_fn=_intent_reduce_0)
        v65 = (v29 > v63)
        v66 = (v29 == v63)
        v67 = (v30 <= v64)
        v68 = (v66 & v67)
        v69 = (v65 | v68)
        v70 = (v29 != v29)
        v71 = (v63 != v63)
        v72 = tl.full((FRAGMENT_D1,), True, tl.int1)
        v73 = tl.where(v71, v67, v72)
        v74 = tl.where(v70, v73, v69)
        v75 = tl.where(v74, v29, v63)
        v76 = tl.where(v74, v30, v64)
        v29 = v75
        v30 = v76
    v77 = (-v29)
    v78 = (v17 * 1)
    v79 = (0 + v78)
    v80 = (v79 + tl.arange(0, FRAGMENT_D1) * 1)
    v81 = (v80 < v21)
    v82 = D1
    v83 = tl.full((FRAGMENT_D1,), v82, tl.int64)
    v84 = (v80 < v83)
    v85 = (v84 & v81)
    tl.store(tl.make_block_ptr(base=(output_values + tl.cast(v79, tl.int64) * S1_0), shape=((D1 - tl.cast(v79, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v77, tl.float32), boundary_check=(0,))
    v86 = (v79 + tl.arange(0, FRAGMENT_D1) * 1)
    v87 = (v86 < v21)
    v88 = D1
    v89 = tl.full((FRAGMENT_D1,), v88, tl.int64)
    v90 = (v86 < v89)
    v91 = (v90 & v87)
    tl.store(tl.make_block_ptr(base=(output_indices + tl.cast(v79, tl.int64) * S2_0), shape=((D1 - tl.cast(v79, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v30, tl.int64), boundary_check=(0,))

def launch(input, output_values, output_indices):
    D1 = input.shape[1]
    D2 = input.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output_values.stride(0)
    S2_0 = output_indices.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output_values, output_indices, D1, D2, S0_0, S0_1, S1_0, S2_0)

def run(input):
    output_values = torch.empty((input.shape[1],), device=input.device, dtype=torch.float32)
    output_indices = torch.empty((input.shape[1],), device=input.device, dtype=torch.int64)
    launch(input, output_values, output_indices)
    return (output_values, output_indices)
