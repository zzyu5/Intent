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

def _intent_cover_FULL_D2(args):
    bound = int(args["D2"])
    for extent in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, ):
        if extent >= bound:
            return extent
    raise ValueError("no legal full-coverage extent for FULL_D2")

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 1}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.heuristics({
    "FULL_D2": _intent_cover_FULL_D2,
})
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, FULL_D2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v1 = (D1 - 0)
    v2 = (v1 + 0)
    v3 = ((v2 // 1) - (((v2 % 1) != 0) & (((v2 % 1) < 0) != (1 < 0))))
    v4 = (FRAGMENT_D1 - 1)
    v5 = (v3 + v4)
    v6 = ((v5 // FRAGMENT_D1) - (((v5 % FRAGMENT_D1) != 0) & (((v5 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v7 = tl.program_id(0)
    v8 = (v7 % v6)
    v9 = (v8 * FRAGMENT_D1)
    v10 = (v9 * 1)
    v11 = (0 + v10)
    v12 = (v11 + 1)
    v13 = (v3 - v9)
    v14 = (v13 * 1)
    v15 = (v11 + v14)
    v16 = (v11 + tl.arange(0, FRAGMENT_D1) * 1)
    v17 = tl.full((FRAGMENT_D1,), v15, tl.int64)
    v18 = (v16 < v17)
    v19 = (0 + tl.arange(0, FULL_D2) * 1)
    v20 = (D2 - 0)
    v21 = (1 - 1)
    v22 = (v20 + v21)
    v23 = ((v22 // 1) - (((v22 % 1) != 0) & (((v22 % 1) < 0) != (1 < 0))))
    v24 = (v23 * 1)
    v25 = (0 + v24)
    v26 = tl.full((FULL_D2,), v25, tl.int64)
    v27 = (v19 < v26)
    v28 = D1
    v29 = tl.full((FRAGMENT_D1,), v28, tl.int64)
    v30 = (v16 < v29)
    v31 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v32 = (v16 >= v31)
    v33 = (v32 & v30)
    v34 = tl.reshape(v33, (FRAGMENT_D1, 1), can_reorder=False)
    v35 = tl.broadcast_to(v34, (FRAGMENT_D1, FULL_D2))
    v36 = D2
    v37 = tl.full((FULL_D2,), v36, tl.int64)
    v38 = (v19 < v37)
    v39 = tl.broadcast_to(v38[None, :], (FRAGMENT_D1, FULL_D2))
    v40 = (v35 & v39)
    v41 = tl.broadcast_to(v18[:, None], (FRAGMENT_D1, FULL_D2))
    v42 = (v40 & v41)
    v43 = tl.full((FRAGMENT_D1, FULL_D2), 0.0, tl.float32)
    v44 = tl.reshape(v27, (1, FULL_D2), can_reorder=False)
    v45 = tl.broadcast_to(v44, (FRAGMENT_D1, FULL_D2))
    v46 = (v42 & v45)
    v47 = tl.full((FRAGMENT_D1,), -float("inf"), tl.float32)
    v48 = (0 + D2)
    v49 = tl.full((FULL_D2,), v48, tl.int64)
    v50 = (v19 < v49)
    v51 = tl.broadcast_to(v50[None, :], (FRAGMENT_D1, FULL_D2))
    v52 = tl.broadcast_to(v47[:, None], (FRAGMENT_D1, FULL_D2))
    v53 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v11, tl.int64) * S0_0 + tl.cast(0, tl.int64) * S0_1), shape=((D1 - tl.cast(v11, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FULL_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    v54 = tl.where(v51, v53, v52)
    v55 = tl.reduce(v54, axis=1, combine_fn=_intent_reduce_0)
    v56 = tl.broadcast_to(v55[:, None], (FRAGMENT_D1, FULL_D2))
    v57 = tl.full((FRAGMENT_D1, FULL_D2), 1.4426950216293335, tl.float32)
    v58 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v59 = tl.broadcast_to(v58[:, None], (FRAGMENT_D1, FULL_D2))
    v60 = (v53 - v56)
    v61 = (v60 * v57)
    v62 = libdevice.exp2(tl.cast(v61, tl.float32))
    v63 = tl.where(v51, v62, v59)
    v64 = tl.sum(v63, axis=1)
    v65 = tl.broadcast_to(v64[:, None], (FRAGMENT_D1, FULL_D2))
    v66 = (0 + tl.arange(0, FULL_D2) * 1)
    v67 = (1 - 1)
    v68 = (v20 + v67)
    v69 = ((v68 // 1) - (((v68 % 1) != 0) & (((v68 % 1) < 0) != (1 < 0))))
    v70 = (v69 * 1)
    v71 = (0 + v70)
    v72 = tl.full((FULL_D2,), v71, tl.int64)
    v73 = (v66 < v72)
    v74 = D1
    v75 = tl.full((FRAGMENT_D1,), v74, tl.int64)
    v76 = (v16 < v75)
    v77 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v78 = (v16 >= v77)
    v79 = (v78 & v76)
    v80 = tl.reshape(v79, (FRAGMENT_D1, 1), can_reorder=False)
    v81 = tl.broadcast_to(v80, (FRAGMENT_D1, FULL_D2))
    v82 = D2
    v83 = tl.full((FULL_D2,), v82, tl.int64)
    v84 = (v66 < v83)
    v85 = tl.broadcast_to(v84[None, :], (FRAGMENT_D1, FULL_D2))
    v86 = (v81 & v85)
    v87 = (v86 & v41)
    v88 = tl.reshape(v73, (1, FULL_D2), can_reorder=False)
    v89 = tl.broadcast_to(v88, (FRAGMENT_D1, FULL_D2))
    v90 = (v87 & v89)
    v91 = tl.fdiv(tl.cast(v62, tl.float32), tl.cast(v65, tl.float32), ieee_rounding=True)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v11, tl.int64) * S1_0 + tl.cast(0, tl.int64) * S1_1), shape=((D1 - tl.cast(v11, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FULL_D2), order=(1, 0)), tl.cast(v91, tl.float32), boundary_check=(0, 1))

def launch(input, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(input):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
