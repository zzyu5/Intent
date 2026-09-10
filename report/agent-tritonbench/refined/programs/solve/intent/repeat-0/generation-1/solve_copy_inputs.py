import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

def _intent_cover_FULL_D1(args):
    bound = int(args["D1"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D1")

def _intent_cover_FULL_D2(args):
    bound = int(args["D2"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D2")

_intent_tuning_hooks = TuningHooks(("a", "b", "work", "rhs", ), (False, False, True, True, ))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D1": _intent_cover_FULL_D1,
    "FULL_D2": _intent_cover_FULL_D2,
})
@triton.jit
def _intent_kernel(a, b, work, rhs, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, FULL_D2: tl.constexpr, FULL_D1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    v2 = (0 + tl.arange(0, FULL_D1) * 1)
    v3 = (D1 - 0)
    v4 = (1 - 1)
    v5 = (v3 + v4)
    v6 = ((v5 // 1) - (((v5 % 1) != 0) & (((v5 % 1) < 0) != (1 < 0))))
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = tl.full((FULL_D1,), v8, tl.int64)
    v10 = (v2 < v9)
    v11 = (0 + tl.arange(0, FULL_D1) * 1)
    v12 = (D1 - 0)
    v13 = (1 - 1)
    v14 = (v12 + v13)
    v15 = ((v14 // 1) - (((v14 % 1) != 0) & (((v14 % 1) < 0) != (1 < 0))))
    v16 = (v15 * 1)
    v17 = (0 + v16)
    v18 = tl.full((FULL_D1,), v17, tl.int64)
    v19 = (v11 < v18)
    v20 = D1
    v21 = tl.full((FULL_D1,), v20, tl.int64)
    v22 = (v2 < v21)
    v23 = tl.broadcast_to(v22[None, :], (FULL_D1, FULL_D1))
    v24 = D1
    v25 = tl.full((FULL_D1,), v24, tl.int64)
    v26 = (v11 < v25)
    v27 = tl.broadcast_to(v26[None, :], (FULL_D1, FULL_D1))
    v28 = (v23 & v27)
    v29 = tl.full((FULL_D1, FULL_D1), 0.0, tl.float32)
    v30 = (0 + tl.arange(0, FULL_D1) * 1)
    v31 = (1 - 1)
    v32 = (v3 + v31)
    v33 = ((v32 // 1) - (((v32 % 1) != 0) & (((v32 % 1) < 0) != (1 < 0))))
    v34 = (v33 * 1)
    v35 = (0 + v34)
    v36 = tl.full((FULL_D1,), v35, tl.int64)
    v37 = (v30 < v36)
    v38 = (0 + tl.arange(0, FULL_D1) * 1)
    v39 = (1 - 1)
    v40 = (v12 + v39)
    v41 = ((v40 // 1) - (((v40 % 1) != 0) & (((v40 % 1) < 0) != (1 < 0))))
    v42 = (v41 * 1)
    v43 = (0 + v42)
    v44 = tl.full((FULL_D1,), v43, tl.int64)
    v45 = (v38 < v44)
    v46 = D1
    v47 = tl.full((FULL_D1,), v46, tl.int64)
    v48 = (v30 < v47)
    v49 = tl.broadcast_to(v48[None, :], (FULL_D1, FULL_D1))
    v50 = D1
    v51 = tl.full((FULL_D1,), v50, tl.int64)
    v52 = (v38 < v51)
    v53 = tl.broadcast_to(v52[None, :], (FULL_D1, FULL_D1))
    v54 = (v49 & v53)
    v55 = tl.reshape(v10, (FULL_D1, 1), can_reorder=False)
    v56 = tl.broadcast_to(v55, (FULL_D1, FULL_D1))
    v57 = tl.reshape(v10, (1, FULL_D1), can_reorder=False)
    v58 = tl.broadcast_to(v57, (FULL_D1, FULL_D1))
    v59 = (v56 & v58)
    v60 = tl.reshape(v19, (1, FULL_D1), can_reorder=False)
    v61 = tl.broadcast_to(v60, (FULL_D1, FULL_D1))
    v62 = (v59 & v61)
    v63 = tl.reshape(v19, (FULL_D1, 1), can_reorder=False)
    v64 = tl.broadcast_to(v63, (FULL_D1, FULL_D1))
    v65 = (v62 & v64)
    v66 = (v28 & v65)
    v67 = tl.reshape(v37, (FULL_D1, 1), can_reorder=False)
    v68 = tl.broadcast_to(v67, (FULL_D1, FULL_D1))
    v69 = tl.reshape(v37, (1, FULL_D1), can_reorder=False)
    v70 = tl.broadcast_to(v69, (FULL_D1, FULL_D1))
    v71 = (v68 & v70)
    v72 = tl.reshape(v45, (1, FULL_D1), can_reorder=False)
    v73 = tl.broadcast_to(v72, (FULL_D1, FULL_D1))
    v74 = (v71 & v73)
    v75 = tl.reshape(v45, (FULL_D1, 1), can_reorder=False)
    v76 = tl.broadcast_to(v75, (FULL_D1, FULL_D1))
    v77 = (v74 & v76)
    v78 = (v54 & v77)
    v79 = tl.load(tl.make_block_ptr(base=(a + tl.cast(0, tl.int64) * S0_0 + tl.cast(0, tl.int64) * S0_1), shape=((D1 - tl.cast(0, tl.int64)), (D1 - tl.cast(0, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(work + tl.cast(0, tl.int64) * S2_0 + tl.cast(0, tl.int64) * S2_1), shape=((D1 - tl.cast(0, tl.int64)), (D1 - tl.cast(0, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D1), order=(1, 0)), tl.cast(v79, tl.float32), boundary_check=(0, 1))
    v80 = (0 + tl.arange(0, FULL_D1) * 1)
    v81 = (D1 - 0)
    v82 = (1 - 1)
    v83 = (v81 + v82)
    v84 = ((v83 // 1) - (((v83 % 1) != 0) & (((v83 % 1) < 0) != (1 < 0))))
    v85 = (v84 * 1)
    v86 = (0 + v85)
    v87 = tl.full((FULL_D1,), v86, tl.int64)
    v88 = (v80 < v87)
    v89 = (0 + tl.arange(0, FULL_D2) * 1)
    v90 = (D2 - 0)
    v91 = (1 - 1)
    v92 = (v90 + v91)
    v93 = ((v92 // 1) - (((v92 % 1) != 0) & (((v92 % 1) < 0) != (1 < 0))))
    v94 = (v93 * 1)
    v95 = (0 + v94)
    v96 = tl.full((FULL_D2,), v95, tl.int64)
    v97 = (v89 < v96)
    v98 = D1
    v99 = tl.full((FULL_D1,), v98, tl.int64)
    v100 = (v80 < v99)
    v101 = tl.broadcast_to(v100[:, None], (FULL_D1, FULL_D2))
    v102 = D2
    v103 = tl.full((FULL_D2,), v102, tl.int64)
    v104 = (v89 < v103)
    v105 = tl.broadcast_to(v104[None, :], (FULL_D1, FULL_D2))
    v106 = (v101 & v105)
    v107 = tl.full((FULL_D1, FULL_D2), 0.0, tl.float32)
    v108 = (0 + tl.arange(0, FULL_D1) * 1)
    v109 = (1 - 1)
    v110 = (v81 + v109)
    v111 = ((v110 // 1) - (((v110 % 1) != 0) & (((v110 % 1) < 0) != (1 < 0))))
    v112 = (v111 * 1)
    v113 = (0 + v112)
    v114 = tl.full((FULL_D1,), v113, tl.int64)
    v115 = (v108 < v114)
    v116 = (0 + tl.arange(0, FULL_D2) * 1)
    v117 = (1 - 1)
    v118 = (v90 + v117)
    v119 = ((v118 // 1) - (((v118 % 1) != 0) & (((v118 % 1) < 0) != (1 < 0))))
    v120 = (v119 * 1)
    v121 = (0 + v120)
    v122 = tl.full((FULL_D2,), v121, tl.int64)
    v123 = (v116 < v122)
    v124 = D1
    v125 = tl.full((FULL_D1,), v124, tl.int64)
    v126 = (v108 < v125)
    v127 = tl.broadcast_to(v126[:, None], (FULL_D1, FULL_D2))
    v128 = D2
    v129 = tl.full((FULL_D2,), v128, tl.int64)
    v130 = (v116 < v129)
    v131 = tl.broadcast_to(v130[None, :], (FULL_D1, FULL_D2))
    v132 = (v127 & v131)
    v133 = tl.reshape(v88, (FULL_D1, 1), can_reorder=False)
    v134 = tl.broadcast_to(v133, (FULL_D1, FULL_D2))
    v135 = (v106 & v134)
    v136 = tl.reshape(v97, (1, FULL_D2), can_reorder=False)
    v137 = tl.broadcast_to(v136, (FULL_D1, FULL_D2))
    v138 = (v135 & v137)
    v139 = tl.reshape(v115, (FULL_D1, 1), can_reorder=False)
    v140 = tl.broadcast_to(v139, (FULL_D1, FULL_D2))
    v141 = (v132 & v140)
    v142 = tl.reshape(v123, (1, FULL_D2), can_reorder=False)
    v143 = tl.broadcast_to(v142, (FULL_D1, FULL_D2))
    v144 = (v141 & v143)
    v145 = tl.load(tl.make_block_ptr(base=(b + tl.cast(0, tl.int64) * S1_0 + tl.cast(0, tl.int64) * S1_1), shape=((D1 - tl.cast(0, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    tl.store(tl.make_block_ptr(base=(rhs + tl.cast(0, tl.int64) * S3_0 + tl.cast(0, tl.int64) * S3_1), shape=((D1 - tl.cast(0, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FULL_D1, FULL_D2), order=(1, 0)), tl.cast(v145, tl.float32), boundary_check=(0, 1))

def launch(a, b, work, rhs):
    D1 = a.shape[0]
    D2 = b.shape[1]
    S0_0 = a.stride(0)
    S0_1 = a.stride(1)
    S1_0 = b.stride(0)
    S1_1 = b.stride(1)
    S2_0 = work.stride(0)
    S2_1 = work.stride(1)
    S3_0 = rhs.stride(0)
    S3_1 = rhs.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](a, b, work, rhs, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S3_0, S3_1)

def run(a, b):
    work = torch.empty((a.shape[0], a.shape[0]), device=a.device, dtype=torch.float32)
    rhs = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    launch(a, b, work, rhs)
    return (work, rhs)
