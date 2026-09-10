import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

def _intent_cover_FULL_D2(args):
    bound = int(args["D2"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D2")

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"POINTWISE_CHUNK_S8_A0": 4096}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D2": _intent_cover_FULL_D2,
})
@triton.jit
def _intent_kernel(input, output, eps, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, FULL_D2: tl.constexpr, POINTWISE_CHUNK_S8_A0: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * 1)
    v6 = (0 + v5)
    v7 = (0 + tl.arange(0, FULL_D2) * 1)
    v8 = (D2 - 0)
    v9 = (1 - 1)
    v10 = (v8 + v9)
    v11 = ((v10 // 1) - (((v10 % 1) != 0) & (((v10 % 1) < 0) != (1 < 0))))
    v12 = (v11 * 1)
    v13 = (0 + v12)
    v14 = v13
    v15 = (v7 < v14)
    v16 = D1
    v17 = (v6 < v16)
    v18 = (v6 >= 0)
    v19 = (v18 & v17)
    v20 = tl.full((FULL_D2,), v19, tl.int1)
    v21 = D2
    v22 = tl.full((FULL_D2,), v21, tl.int64)
    v23 = (v7 < v22)
    v24 = (v20 & v23)
    v25 = tl.full((FULL_D2,), 0.0, tl.float32)
    v26 = (v24 & v15)
    v27 = (0 + D2)
    v28 = v27
    v29 = (v7 < v28)
    v30 = 0.0
    v31 = tl.load((input + (v6) * S0_0 + (tl.broadcast_to(v7, (FULL_D2, ))) * S0_1), mask=v26, other=v25)
    v32 = tl.where(v29, v31, v30)
    v33 = tl.sum(v32, axis=0)
    v34 = tl.cast(D2, tl.float32)
    v35 = tl.fdiv(tl.cast(v33, tl.float32), tl.cast(v34, tl.float32), ieee_rounding=True)
    v36 = v35
    v37 = 0.0
    v38 = (v31 - v36)
    v39 = (v38 * v38)
    v40 = tl.where(v29, v39, v37)
    v41 = tl.sum(v40, axis=0)
    v42 = tl.fdiv(tl.cast(v41, tl.float32), tl.cast(v34, tl.float32), ieee_rounding=True)
    v43 = (v42 + eps)
    v44 = libdevice.pow(v43, 0.5)
    v45 = D1
    v46 = (v6 < v45)
    v47 = (v6 >= 0)
    v48 = (v47 & v46)
    v49 = D2
    v50 = (POINTWISE_CHUNK_S8_A0 * 1)
    for iv51 in range(0, D2, v50):
        v52 = (iv51 + tl.arange(0, POINTWISE_CHUNK_S8_A0) * 1)
        v53 = D2
        v54 = (v52 < v53)
        v55 = tl.full((POINTWISE_CHUNK_S8_A0,), v19, tl.int1)
        v56 = tl.full((POINTWISE_CHUNK_S8_A0,), v21, tl.int64)
        v57 = (v52 < v56)
        v58 = (v55 & v57)
        v59 = tl.full((POINTWISE_CHUNK_S8_A0,), 0.0, tl.float32)
        v60 = (v58 & v54)
        v61 = v35
        v62 = v44
        v63 = tl.full((POINTWISE_CHUNK_S8_A0,), v48, tl.int1)
        v64 = tl.full((POINTWISE_CHUNK_S8_A0,), v49, tl.int64)
        v65 = (v52 < v64)
        v66 = (v63 & v65)
        v67 = (v54 & v66)
        v68 = tl.load((input + (v6) * S0_0 + (tl.broadcast_to(v52, (POINTWISE_CHUNK_S8_A0, ))) * S0_1), mask=v60, other=v59)
        v69 = (v68 - v61)
        v70 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v62, tl.float32), ieee_rounding=True)
        tl.store((output + (v6) * S2_0 + (tl.broadcast_to(v52, (POINTWISE_CHUNK_S8_A0, ))) * S2_1), v70, mask=v67)

def launch(input, output, eps):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (D1,)
    return _intent_kernel[grid](input, output, eps, D1, D2, S0_0, S0_1, S2_0, S2_1)

def run(input, eps):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output, eps)
    return output
