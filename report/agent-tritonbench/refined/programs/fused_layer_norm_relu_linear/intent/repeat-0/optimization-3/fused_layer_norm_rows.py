import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256, "REDUCE_CHUNK_8_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256, "REDUCE_CHUNK_8_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256, "REDUCE_CHUNK_8_A0": 64}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64, "REDUCE_CHUNK_8_A0": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64, "REDUCE_CHUNK_8_A0": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64, "REDUCE_CHUNK_8_A0": 128}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16, "REDUCE_CHUNK_8_A0": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16, "REDUCE_CHUNK_8_A0": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16, "REDUCE_CHUNK_8_A0": 32}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 1024}, num_warps=16, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096, "REDUCE_CHUNK_8_A0": 8192}, num_warps=16, num_stages=2, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, eps, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, REDUCE_CHUNK_8_A0: tl.constexpr, POINTWISE_CHUNK_S8_A0: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * 1)
    v6 = (0 + v5)
    v7 = D1
    v8 = (v6 < v7)
    v9 = (v6 >= 0)
    v10 = (v9 & v8)
    v11 = D2
    v12 = 0.0
    for iv13 in range(0, D2, REDUCE_CHUNK_8_A0):
        v14 = (iv13 + tl.arange(0, REDUCE_CHUNK_8_A0) * 1)
        v15 = tl.full((REDUCE_CHUNK_8_A0,), D2, tl.int64)
        v16 = (v14 < v15)
        v17 = v16
        v18 = (iv13 + tl.arange(0, REDUCE_CHUNK_8_A0) * 1)
        v19 = (v18 < v15)
        v20 = v19
        v21 = tl.full((REDUCE_CHUNK_8_A0,), v10, tl.int1)
        v22 = tl.full((REDUCE_CHUNK_8_A0,), v11, tl.int64)
        v23 = (v14 < v22)
        v24 = (v21 & v23)
        v25 = (v20 & v24)
        v26 = tl.full((REDUCE_CHUNK_8_A0,), 0.0, tl.float32)
        v27 = tl.full((REDUCE_CHUNK_8_A0,), 0.0, tl.float32)
        v28 = tl.load((input + (tl.full((REDUCE_CHUNK_8_A0,), v6, tl.int64)) * S0_0 + (tl.broadcast_to(v14, (REDUCE_CHUNK_8_A0,))) * S0_1), mask=v25, other=v26)
        v29 = tl.where(v17, v28, v27)
        v30 = tl.sum(v29, axis=0)
        v31 = (v12 + v30)
        v12 = v31
    v32 = tl.cast(D2, tl.float32)
    v33 = tl.fdiv(tl.cast(v12, tl.float32), tl.cast(v32, tl.float32), ieee_rounding=True)
    v34 = 0.0
    for iv35 in range(0, D2, REDUCE_CHUNK_8_A0):
        v36 = (iv35 + tl.arange(0, REDUCE_CHUNK_8_A0) * 1)
        v37 = tl.full((REDUCE_CHUNK_8_A0,), D2, tl.int64)
        v38 = (v36 < v37)
        v39 = v38
        v40 = (iv35 + tl.arange(0, REDUCE_CHUNK_8_A0) * 1)
        v41 = (v40 < v37)
        v42 = v41
        v43 = tl.full((REDUCE_CHUNK_8_A0,), v10, tl.int1)
        v44 = tl.full((REDUCE_CHUNK_8_A0,), v11, tl.int64)
        v45 = (v36 < v44)
        v46 = (v43 & v45)
        v47 = (v42 & v46)
        v48 = tl.full((REDUCE_CHUNK_8_A0,), 0.0, tl.float32)
        v49 = tl.full((REDUCE_CHUNK_8_A0,), v33, tl.float32)
        v50 = tl.full((REDUCE_CHUNK_8_A0,), 0.0, tl.float32)
        v51 = tl.load((input + (tl.full((REDUCE_CHUNK_8_A0,), v6, tl.int64)) * S0_0 + (tl.broadcast_to(v36, (REDUCE_CHUNK_8_A0,))) * S0_1), mask=v47, other=v48)
        v52 = (v51 - v49)
        v53 = (v52 * v52)
        v54 = tl.where(v39, v53, v50)
        v55 = tl.sum(v54, axis=0)
        v56 = (v34 + v55)
        v34 = v56
    v57 = tl.fdiv(tl.cast(v34, tl.float32), tl.cast(v32, tl.float32), ieee_rounding=True)
    v58 = (v57 + eps)
    v59 = libdevice.sqrt(tl.cast(v58, tl.float32))
    v60 = D1
    v61 = (v6 < v60)
    v62 = (v6 >= 0)
    v63 = (v62 & v61)
    v64 = D2
    v65 = (POINTWISE_CHUNK_S8_A0 * 1)
    for iv66 in range(0, D2, v65):
        v67 = (iv66 + tl.arange(0, POINTWISE_CHUNK_S8_A0) * 1)
        v68 = tl.full((POINTWISE_CHUNK_S8_A0,), D2, tl.int64)
        v69 = (v67 < v68)
        v70 = tl.full((POINTWISE_CHUNK_S8_A0,), v10, tl.int1)
        v71 = tl.full((POINTWISE_CHUNK_S8_A0,), v11, tl.int64)
        v72 = (v67 < v71)
        v73 = (v70 & v72)
        v74 = tl.full((POINTWISE_CHUNK_S8_A0,), 0.0, tl.float32)
        v75 = (v73 & v69)
        v76 = tl.full((POINTWISE_CHUNK_S8_A0,), v33, tl.float32)
        v77 = tl.full((POINTWISE_CHUNK_S8_A0,), v59, tl.float32)
        v78 = tl.full((POINTWISE_CHUNK_S8_A0,), v63, tl.int1)
        v79 = tl.full((POINTWISE_CHUNK_S8_A0,), v64, tl.int64)
        v80 = (v67 < v79)
        v81 = (v78 & v80)
        v82 = (v69 & v81)
        v83 = tl.load((input + (tl.full((POINTWISE_CHUNK_S8_A0,), v6, tl.int64)) * S0_0 + (tl.broadcast_to(v67, (POINTWISE_CHUNK_S8_A0,))) * S0_1), mask=v75, other=v74)
        v84 = (v83 - v76)
        v85 = tl.fdiv(tl.cast(v84, tl.float32), tl.cast(v77, tl.float32), ieee_rounding=True)
        tl.store((output + (tl.full((POINTWISE_CHUNK_S8_A0,), v6, tl.int64)) * S1_0 + (tl.broadcast_to(v67, (POINTWISE_CHUNK_S8_A0,))) * S1_1), v85, mask=v82)

def launch(input, output, eps):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    grid = lambda META: (D1,)
    return _intent_kernel[grid](input, output, eps, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(input, eps):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output, eps)
    return output
