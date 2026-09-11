import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1, a2, a3):
    v0 = (a0, a1)
    v1 = (a2, a3)
    v2 = v0[0]
    v3 = v0[1]
    v4 = v1[0]
    v5 = v1[1]
    v6 = (v2 == v2)
    v7 = (v4 != v4)
    v8 = (v4 < v2)
    v9 = (v7 | v8)
    v10 = (v6 & v9)
    v11 = tl.where(v10, v4, v2)
    v12 = tl.where(v10, v5, v3)
    v13 = (v11, v12)
    v14 = v13[0]
    v15 = v13[1]
    return (v14, v15)

_intent_tuning_hooks = TuningHooks(("input", "output", "output_indices", ), (False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 4096}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 2048}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, output_indices, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr, FRAGMENT_D2: tl.constexpr):
    v16 = (FRAGMENT_D2 - 1)
    v17 = (D2 + v16)
    v18 = ((v17 // FRAGMENT_D2) - (((v17 % FRAGMENT_D2) != 0) & (((v17 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v19 = tl.program_id(0)
    v20 = (v19 % v18)
    v21 = tl.full((1,), float("inf"), tl.float32)
    v22 = tl.full((1,), 0, tl.int64)
    v23 = (v20 * FRAGMENT_D2)
    v24 = (v23 * 1)
    v25 = (0 + v24)
    v26 = (v25 + tl.arange(0, FRAGMENT_D2) * 1)
    v27 = tl.full((FRAGMENT_D2,), D2, tl.int64)
    v28 = (v26 < v27)
    v29 = D1
    v30 = D2
    v31 = tl.full((FRAGMENT_D2,), v30, tl.int64)
    v32 = (v26 < v31)
    v33 = (v21, v22)
    v34 = v33[0]
    v35 = v33[1]
    v36 = tl.broadcast_to(v34, (FRAGMENT_D2,))
    v37 = tl.broadcast_to(v35, (FRAGMENT_D2,))
    v38 = v36
    v39 = v37
    for iv40 in range(0, D1, REDUCE_CHUNK_1_A0):
        v41 = (iv40 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v42 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v43 = (v41 < v42)
        v44 = tl.broadcast_to(v43[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v45 = (iv40 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v46 = (v45 < v42)
        v47 = tl.broadcast_to(v46[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v48 = tl.full((REDUCE_CHUNK_1_A0,), v29, tl.int64)
        v49 = (v41 < v48)
        v50 = tl.broadcast_to(v49[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v51 = tl.broadcast_to(v32[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v52 = (v50 & v51)
        v53 = tl.broadcast_to(v28[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v54 = (v52 & v53)
        v55 = (v47 & v54)
        v56 = tl.full((REDUCE_CHUNK_1_A0, FRAGMENT_D2), 0.0, tl.float32)
        v57 = tl.broadcast_to(v36[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v58 = (iv40 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v59 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v60 = (v58 < v59)
        v61 = tl.broadcast_to(v60[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v62 = tl.reshape(v58, (REDUCE_CHUNK_1_A0, 1), can_reorder=False)
        v63 = tl.cast(v62, tl.int64)
        v64 = tl.broadcast_to(v63, (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v65 = tl.full((REDUCE_CHUNK_1_A0, FRAGMENT_D2), 0, tl.int64)
        v66 = (v64 + v65)
        v67 = tl.broadcast_to(v37[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v68 = tl.where(v61, v66, v67)
        v69 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv40, tl.int64) * S0_0 + tl.cast(v25, tl.int64) * S0_1), shape=((D1 - tl.cast(iv40, tl.int64)), (D2 - tl.cast(v25, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_1_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v70 = tl.where(v44, v69, v57)
        v71, v72 = tl.reduce((v70, v68), axis=0, combine_fn=_intent_reduce_0)
        v73 = (v38, v39)
        v74 = (v71, v72)
        v75 = v73[0]
        v76 = v73[1]
        v77 = v74[0]
        v78 = v74[1]
        v79 = (v75 == v75)
        v80 = (v77 != v77)
        v81 = (v77 < v75)
        v82 = (v80 | v81)
        v83 = (v79 & v82)
        v84 = tl.where(v83, v77, v75)
        v85 = tl.broadcast_to(v83, (FRAGMENT_D2,))
        v86 = tl.where(v85, v78, v76)
        v87 = (v84, v86)
        v88 = v87[0]
        v89 = v87[1]
        v38 = v88
        v39 = v89
    v90 = (v38, v39)
    v91 = v90[0]
    v92 = v90[1]
    v93 = (v23 * 1)
    v94 = (0 + v93)
    v95 = (v94 + tl.arange(0, FRAGMENT_D2) * 1)
    v96 = (v95 < v27)
    v97 = D2
    v98 = tl.full((FRAGMENT_D2,), v97, tl.int64)
    v99 = (v95 < v98)
    v100 = (v99 & v96)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v94, tl.int64) * S1_0), shape=((D2 - tl.cast(v94, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), tl.cast(v91, tl.float32), boundary_check=(0,))
    v101 = (v23 * 1)
    v102 = (0 + v101)
    v103 = (v102 + tl.arange(0, FRAGMENT_D2) * 1)
    v104 = tl.full((FRAGMENT_D2,), D2, tl.int64)
    v105 = (v103 < v104)
    v106 = D2
    v107 = tl.full((FRAGMENT_D2,), v106, tl.int64)
    v108 = (v103 < v107)
    v109 = (v108 & v105)
    tl.store(tl.make_block_ptr(base=(output_indices + tl.cast(v102, tl.int64) * S2_0), shape=((D2 - tl.cast(v102, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), tl.cast(v92, tl.int64), boundary_check=(0,))

def launch(input, output, output_indices):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    S2_0 = output_indices.stride(0)
    grid = lambda META: (triton.cdiv(D2, META["FRAGMENT_D2"]),)
    return _intent_kernel[grid](input, output, output_indices, D1, D2, S0_0, S0_1, S1_0, S2_0)

def run(input):
    output = torch.empty((input.shape[1],), device=input.device, dtype=torch.float32)
    output_indices = torch.empty((input.shape[1],), device=input.device, dtype=torch.int64)
    launch(input, output, output_indices)
    return (output, output_indices)
