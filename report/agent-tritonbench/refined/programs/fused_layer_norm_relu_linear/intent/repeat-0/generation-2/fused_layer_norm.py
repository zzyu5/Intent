import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 1, "POINTWISE_CHUNK_S1_A1": 256, "REDUCE_CHUNK_1_A1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "POINTWISE_CHUNK_S1_A1": 256, "REDUCE_CHUNK_1_A1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "POINTWISE_CHUNK_S1_A1": 256, "REDUCE_CHUNK_1_A1": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "POINTWISE_CHUNK_S1_A1": 64, "REDUCE_CHUNK_1_A1": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "POINTWISE_CHUNK_S1_A1": 64, "REDUCE_CHUNK_1_A1": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "POINTWISE_CHUNK_S1_A1": 64, "REDUCE_CHUNK_1_A1": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "POINTWISE_CHUNK_S1_A1": 16, "REDUCE_CHUNK_1_A1": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "POINTWISE_CHUNK_S1_A1": 16, "REDUCE_CHUNK_1_A1": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "POINTWISE_CHUNK_S1_A1": 16, "REDUCE_CHUNK_1_A1": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "POINTWISE_CHUNK_S1_A1": 4096, "REDUCE_CHUNK_1_A1": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, eps, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, REDUCE_CHUNK_1_A1: tl.constexpr, POINTWISE_CHUNK_S1_A1: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v42 = tl.reshape(v41, (FRAGMENT_D1, 1), can_reorder=False)
    v43 = (v6 * 1)
    v44 = (0 + v43)
    v45 = (v44 + tl.arange(0, FRAGMENT_D1) * 1)
    v46 = (v45 < v10)
    v47 = (v45 < v13)
    v48 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v49 = v48
    for iv50 in range(0, D2, REDUCE_CHUNK_1_A1):
        v51 = (iv50 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v52 = tl.full((REDUCE_CHUNK_1_A1,), D2, tl.int64)
        v53 = (v51 < v52)
        v54 = tl.broadcast_to(v53[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v55 = (iv50 + tl.arange(0, REDUCE_CHUNK_1_A1) * 1)
        v56 = (v55 < v52)
        v57 = tl.broadcast_to(v56[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v58 = tl.full((REDUCE_CHUNK_1_A1,), v15, tl.int64)
        v59 = (v51 < v58)
        v60 = tl.broadcast_to(v59[None, :], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v61 = tl.broadcast_to(v47[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v62 = (v61 & v60)
        v63 = tl.broadcast_to(v46[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v64 = (v62 & v63)
        v65 = (v57 & v64)
        v66 = tl.full((FRAGMENT_D1, REDUCE_CHUNK_1_A1), 0.0, tl.float32)
        v67 = tl.broadcast_to(v42, (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v68 = tl.broadcast_to(v48[:, None], (FRAGMENT_D1, REDUCE_CHUNK_1_A1))
        v69 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v44, tl.int64) * S0_0 + tl.cast(iv50, tl.int64) * S0_1), shape=((D1 - tl.cast(v44, tl.int64)), (D2 - tl.cast(iv50, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, REDUCE_CHUNK_1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v70 = (v69 - v67)
        v71 = (v70 * v70)
        v72 = tl.where(v54, v71, v68)
        v73 = tl.sum(v72, axis=1)
        v74 = (v49 + v73)
        v49 = v74
    v75 = tl.full((FRAGMENT_D1,), v5, tl.float32)
    v76 = tl.fdiv(tl.cast(v49, tl.float32), tl.cast(v75, tl.float32), ieee_rounding=True)
    v77 = tl.reshape(v76, (FRAGMENT_D1, 1), can_reorder=False)
    v78 = tl.full((FRAGMENT_D1, 1), eps, tl.float32)
    v79 = tl.broadcast_to(v77, (FRAGMENT_D1, 1))
    v80 = (v79 + v78)
    v81 = libdevice.sqrt(tl.cast(v80, tl.float32))
    v82 = (v6 * 1)
    v83 = (0 + v82)
    v84 = (v83 + tl.arange(0, FRAGMENT_D1) * 1)
    v85 = (v84 < v10)
    v86 = D1
    v87 = tl.full((FRAGMENT_D1,), v86, tl.int64)
    v88 = (v84 < v87)
    v89 = D2
    v90 = (POINTWISE_CHUNK_S1_A1 * 1)
    for iv91 in range(0, D2, v90):
        v92 = (iv91 + tl.arange(0, POINTWISE_CHUNK_S1_A1) * 1)
        v93 = tl.full((POINTWISE_CHUNK_S1_A1,), D2, tl.int64)
        v94 = (v92 < v93)
        v95 = tl.broadcast_to(v47[:, None], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v96 = tl.full((POINTWISE_CHUNK_S1_A1,), v15, tl.int64)
        v97 = (v92 < v96)
        v98 = tl.broadcast_to(v97[None, :], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v99 = (v95 & v98)
        v100 = tl.broadcast_to(v46[:, None], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v101 = (v99 & v100)
        v102 = tl.full((FRAGMENT_D1, POINTWISE_CHUNK_S1_A1), 0.0, tl.float32)
        v103 = tl.broadcast_to(v94[None, :], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v104 = (v101 & v103)
        v105 = tl.broadcast_to(v42, (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v106 = tl.broadcast_to(v81, (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v107 = tl.broadcast_to(v88[:, None], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v108 = tl.full((POINTWISE_CHUNK_S1_A1,), v89, tl.int64)
        v109 = (v92 < v108)
        v110 = tl.broadcast_to(v109[None, :], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v111 = (v107 & v110)
        v112 = tl.broadcast_to(v85[:, None], (FRAGMENT_D1, POINTWISE_CHUNK_S1_A1))
        v113 = (v111 & v112)
        v114 = (v103 & v113)
        v115 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v44, tl.int64) * S0_0 + tl.cast(iv91, tl.int64) * S0_1), shape=((D1 - tl.cast(v44, tl.int64)), (D2 - tl.cast(iv91, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, POINTWISE_CHUNK_S1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v116 = (v115 - v105)
        v117 = tl.fdiv(tl.cast(v116, tl.float32), tl.cast(v106, tl.float32), ieee_rounding=True)
        tl.store(tl.make_block_ptr(base=(output + tl.cast(v83, tl.int64) * S1_0 + tl.cast(iv91, tl.int64) * S1_1), shape=((D1 - tl.cast(v83, tl.int64)), (D2 - tl.cast(iv91, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, POINTWISE_CHUNK_S1_A1), order=(1, 0)), tl.cast(v117, tl.float32), boundary_check=(0, 1))

def launch(input, output, eps):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, eps, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(input, eps):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output, eps)
    return output
