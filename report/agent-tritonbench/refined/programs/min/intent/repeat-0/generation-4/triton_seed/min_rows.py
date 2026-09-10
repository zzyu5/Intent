import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

@triton.jit
def _intent_reduce_0(a0, a1):
    v0 = tl.minimum(a0, a1, propagate_nan=tl.PropagateNan.ALL)
    return v0

_intent_tuning_hooks = TuningHooks(("input", "output", "output_indices", ), (False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_5_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_5_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 1, "REDUCE_CHUNK_1_A0": 64, "REDUCE_CHUNK_5_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_1_A0": 128, "REDUCE_CHUNK_5_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_1_A0": 128, "REDUCE_CHUNK_5_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 4, "REDUCE_CHUNK_1_A0": 128, "REDUCE_CHUNK_5_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_5_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_5_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 8, "REDUCE_CHUNK_1_A0": 32, "REDUCE_CHUNK_5_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_1_A0": 1024, "REDUCE_CHUNK_5_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_1_A0": 1024, "REDUCE_CHUNK_5_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 128, "REDUCE_CHUNK_1_A0": 1024, "REDUCE_CHUNK_5_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 64, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 32, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D2": 16, "REDUCE_CHUNK_1_A0": 8192, "REDUCE_CHUNK_5_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S2_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, output_indices, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S2_0: tl.constexpr, REDUCE_CHUNK_5_A0: tl.constexpr, REDUCE_CHUNK_1_A0: tl.constexpr, FRAGMENT_D2: tl.constexpr):
    v1 = (FRAGMENT_D2 - 1)
    v2 = (D2 + v1)
    v3 = ((v2 // FRAGMENT_D2) - (((v2 % FRAGMENT_D2) != 0) & (((v2 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
    v4 = tl.program_id(0)
    v5 = (v4 % v3)
    v6 = (v5 * FRAGMENT_D2)
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = (v8 + tl.arange(0, FRAGMENT_D2) * 1)
    v10 = tl.full((FRAGMENT_D2,), D2, tl.int64)
    v11 = (v9 < v10)
    v12 = D1
    v13 = D2
    v14 = tl.full((FRAGMENT_D2,), v13, tl.int64)
    v15 = (v9 < v14)
    v16 = tl.full((FRAGMENT_D2,), float("inf"), tl.float32)
    v17 = v16
    for iv18 in range(0, D1, REDUCE_CHUNK_1_A0):
        v19 = (iv18 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v20 = tl.full((REDUCE_CHUNK_1_A0,), D1, tl.int64)
        v21 = (v19 < v20)
        v22 = tl.broadcast_to(v21[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v23 = (iv18 + tl.arange(0, REDUCE_CHUNK_1_A0) * 1)
        v24 = (v23 < v20)
        v25 = tl.broadcast_to(v24[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v26 = tl.full((REDUCE_CHUNK_1_A0,), v12, tl.int64)
        v27 = (v19 < v26)
        v28 = tl.broadcast_to(v27[:, None], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v29 = tl.broadcast_to(v15[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v30 = (v28 & v29)
        v31 = tl.broadcast_to(v11[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v32 = (v30 & v31)
        v33 = (v25 & v32)
        v34 = tl.full((REDUCE_CHUNK_1_A0, FRAGMENT_D2), 0.0, tl.float32)
        v35 = tl.broadcast_to(v16[None, :], (REDUCE_CHUNK_1_A0, FRAGMENT_D2))
        v36 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv18, tl.int64) * S0_0 + tl.cast(v8, tl.int64) * S0_1), shape=((D1 - tl.cast(iv18, tl.int64)), (D2 - tl.cast(v8, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_1_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v37 = tl.where(v22, v36, v35)
        v38 = tl.reduce(v37, axis=0, combine_fn=_intent_reduce_0)
        v39 = tl.minimum(v17, v38, propagate_nan=tl.PropagateNan.ALL)
        v17 = v39
    v40 = (v17 != v17)
    v41 = (v6 * 1)
    v42 = (0 + v41)
    v43 = (v42 + tl.arange(0, FRAGMENT_D2) * 1)
    v44 = (v43 < v10)
    v45 = (v43 < v14)
    v46 = (v6 * 1)
    v47 = (0 + v46)
    v48 = (v47 + tl.arange(0, FRAGMENT_D2) * 1)
    v49 = (v48 < v10)
    v50 = (v48 < v14)
    v51 = (v6 * 1)
    v52 = (0 + v51)
    v53 = (v52 + tl.arange(0, FRAGMENT_D2) * 1)
    v54 = (v53 < v10)
    v55 = (v53 < v14)
    v56 = tl.full((FRAGMENT_D2,), 1024, tl.int64)
    v57 = v56
    for iv58 in range(0, D1, REDUCE_CHUNK_5_A0):
        v59 = (iv58 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v60 = tl.full((REDUCE_CHUNK_5_A0,), D1, tl.int64)
        v61 = (v59 < v60)
        v62 = tl.broadcast_to(v61[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v63 = (iv58 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v64 = tl.full((REDUCE_CHUNK_5_A0,), D1, tl.int64)
        v65 = (v63 < v64)
        v66 = tl.broadcast_to(v65[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v67 = (v62 & v66)
        v68 = (iv58 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v69 = (v68 < v64)
        v70 = tl.broadcast_to(v69[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v71 = (v67 & v70)
        v72 = (iv58 + tl.arange(0, REDUCE_CHUNK_5_A0) * 1)
        v73 = (v72 < v64)
        v74 = tl.broadcast_to(v73[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v75 = (v71 & v74)
        v76 = tl.full((REDUCE_CHUNK_5_A0,), v12, tl.int64)
        v77 = (v63 < v76)
        v78 = tl.broadcast_to(v77[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v79 = tl.broadcast_to(v45[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v80 = (v78 & v79)
        v81 = tl.broadcast_to(v44[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v82 = (v80 & v81)
        v83 = tl.full((REDUCE_CHUNK_5_A0, FRAGMENT_D2), 0.0, tl.float32)
        v84 = tl.broadcast_to(v17[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v85 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv58, tl.int64) * S0_0 + tl.cast(v42, tl.int64) * S0_1), shape=((D1 - tl.cast(iv58, tl.int64)), (D2 - tl.cast(v42, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_5_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v86 = (v85 == v84)
        v87 = (v68 < v76)
        v88 = tl.broadcast_to(v87[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v89 = tl.broadcast_to(v50[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v90 = (v88 & v89)
        v91 = tl.broadcast_to(v49[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v92 = (v90 & v91)
        v93 = tl.full((REDUCE_CHUNK_5_A0, FRAGMENT_D2), 0.0, tl.float32)
        v94 = (v72 < v76)
        v95 = tl.broadcast_to(v94[:, None], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v96 = tl.broadcast_to(v55[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v97 = (v95 & v96)
        v98 = tl.broadcast_to(v54[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v99 = (v97 & v98)
        v100 = tl.full((REDUCE_CHUNK_5_A0, FRAGMENT_D2), 0.0, tl.float32)
        v101 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv58, tl.int64) * S0_0 + tl.cast(v52, tl.int64) * S0_1), shape=((D1 - tl.cast(iv58, tl.int64)), (D2 - tl.cast(v52, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_5_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v102 = tl.load(tl.make_block_ptr(base=(input + tl.cast(iv58, tl.int64) * S0_0 + tl.cast(v47, tl.int64) * S0_1), shape=((D1 - tl.cast(iv58, tl.int64)), (D2 - tl.cast(v47, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(REDUCE_CHUNK_5_A0, FRAGMENT_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v103 = (v102 != v101)
        v104 = tl.broadcast_to(v103, (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v105 = tl.broadcast_to(v40[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v106 = (v105 & v104)
        v107 = tl.broadcast_to(v106, (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v108 = (v86 | v107)
        v109 = tl.broadcast_to(v108, (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v110 = tl.reshape(v59, (REDUCE_CHUNK_5_A0, 1), can_reorder=False)
        v111 = tl.cast(v110, tl.int64)
        v112 = tl.broadcast_to(v111, (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v113 = tl.full((REDUCE_CHUNK_5_A0, FRAGMENT_D2), 0, tl.int64)
        v114 = (v112 + v113)
        v115 = tl.full((REDUCE_CHUNK_5_A0, FRAGMENT_D2), 1024, tl.int64)
        v116 = tl.where(v109, v114, v115)
        v117 = tl.broadcast_to(v56[None, :], (REDUCE_CHUNK_5_A0, FRAGMENT_D2))
        v118 = tl.where(v75, v116, v117)
        v119 = tl.min(v118, axis=0)
        v120 = tl.minimum(v57, v119, propagate_nan=tl.PropagateNan.ALL)
        v57 = v120
    v121 = (v6 * 1)
    v122 = (0 + v121)
    v123 = (v122 + tl.arange(0, FRAGMENT_D2) * 1)
    v124 = (v123 < v10)
    v125 = D2
    v126 = tl.full((FRAGMENT_D2,), v125, tl.int64)
    v127 = (v123 < v126)
    v128 = (v127 & v124)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v122, tl.int64) * S1_0), shape=((D2 - tl.cast(v122, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), tl.cast(v17, tl.float32), boundary_check=(0,))
    v129 = (v6 * 1)
    v130 = (0 + v129)
    v131 = (v130 + tl.arange(0, FRAGMENT_D2) * 1)
    v132 = tl.full((FRAGMENT_D2,), D2, tl.int64)
    v133 = (v131 < v132)
    v134 = D2
    v135 = tl.full((FRAGMENT_D2,), v134, tl.int64)
    v136 = (v131 < v135)
    v137 = (v136 & v133)
    tl.store(tl.make_block_ptr(base=(output_indices + tl.cast(v130, tl.int64) * S2_0), shape=((D2 - tl.cast(v130, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D2,), order=(0,)), tl.cast(v57, tl.int64), boundary_check=(0,))

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
