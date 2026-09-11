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

_intent_tuning_hooks = TuningHooks(("x", "output", ), (False, True, ))

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
def _intent_kernel(x, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, FULL_D2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v1 = (FRAGMENT_D1 - 1)
    v2 = (D1 + v1)
    v3 = ((v2 // FRAGMENT_D1) - (((v2 % FRAGMENT_D1) != 0) & (((v2 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v4 = tl.program_id(0)
    v5 = (v4 % v3)
    v6 = (v5 * FRAGMENT_D1)
    v7 = (v6 * 1)
    v8 = (0 + v7)
    v9 = (v8 + tl.arange(0, FRAGMENT_D1) * 1)
    v10 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v11 = (v9 < v10)
    v12 = (0 + tl.arange(0, FULL_D2) * 1)
    v13 = (D2 - 0)
    v14 = (1 - 1)
    v15 = (v13 + v14)
    v16 = ((v15 // 1) - (((v15 % 1) != 0) & (((v15 % 1) < 0) != (1 < 0))))
    v17 = (v16 * 1)
    v18 = (0 + v17)
    v19 = tl.full((FULL_D2,), v18, tl.int64)
    v20 = (v12 < v19)
    v21 = D1
    v22 = tl.full((FRAGMENT_D1,), v21, tl.int64)
    v23 = (v9 < v22)
    v24 = tl.broadcast_to(v23[:, None], (FRAGMENT_D1, FULL_D2))
    v25 = D2
    v26 = tl.full((FULL_D2,), v25, tl.int64)
    v27 = (v12 < v26)
    v28 = tl.broadcast_to(v27[None, :], (FRAGMENT_D1, FULL_D2))
    v29 = (v24 & v28)
    v30 = tl.full((FRAGMENT_D1, FULL_D2), 0.0, tl.float32)
    v31 = tl.full((FRAGMENT_D1,), -float("inf"), tl.float32)
    v32 = tl.broadcast_to(v11[:, None], (FRAGMENT_D1, FULL_D2))
    v33 = (v29 & v32)
    v34 = tl.reshape(v20, (1, FULL_D2), can_reorder=False)
    v35 = tl.broadcast_to(v34, (FRAGMENT_D1, FULL_D2))
    v36 = (v33 & v35)
    v37 = (0 + D2)
    v38 = tl.full((FULL_D2,), v37, tl.int64)
    v39 = (v12 < v38)
    v40 = tl.broadcast_to(v39[None, :], (FRAGMENT_D1, FULL_D2))
    v41 = tl.broadcast_to(v31[:, None], (FRAGMENT_D1, FULL_D2))
    v42 = tl.load(tl.make_block_ptr(base=(x + tl.cast(v8, tl.int64) * S0_0 + tl.cast(0, tl.int64) * S0_1), shape=((D1 - tl.cast(v8, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FULL_D2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    v43 = tl.where(v40, v42, v41)
    v44 = tl.reduce(v43, axis=1, combine_fn=_intent_reduce_0)
    v45 = tl.reshape(v44, (FRAGMENT_D1, 1), can_reorder=False)
    v46 = tl.broadcast_to(v45, (FRAGMENT_D1, FULL_D2))
    v47 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v48 = tl.broadcast_to(v47[:, None], (FRAGMENT_D1, FULL_D2))
    v49 = (v42 - v46)
    v50 = libdevice.exp(tl.cast(v49, tl.float32))
    v51 = tl.where(v40, v50, v48)
    v52 = tl.sum(v51, axis=1)
    v53 = tl.reshape(v52, (FRAGMENT_D1, 1), can_reorder=False)
    v54 = tl.broadcast_to(v53, (FRAGMENT_D1, FULL_D2))
    v55 = (v8 + tl.arange(0, FRAGMENT_D1) * 1)
    v56 = (v55 < v10)
    v57 = (0 + tl.arange(0, FULL_D2) * 1)
    v58 = (1 - 1)
    v59 = (v13 + v58)
    v60 = ((v59 // 1) - (((v59 % 1) != 0) & (((v59 % 1) < 0) != (1 < 0))))
    v61 = (v60 * 1)
    v62 = (0 + v61)
    v63 = tl.full((FULL_D2,), v62, tl.int64)
    v64 = (v57 < v63)
    v65 = D1
    v66 = tl.full((FRAGMENT_D1,), v65, tl.int64)
    v67 = (v55 < v66)
    v68 = tl.broadcast_to(v67[:, None], (FRAGMENT_D1, FULL_D2))
    v69 = D2
    v70 = tl.full((FULL_D2,), v69, tl.int64)
    v71 = (v57 < v70)
    v72 = tl.broadcast_to(v71[None, :], (FRAGMENT_D1, FULL_D2))
    v73 = (v68 & v72)
    v74 = tl.broadcast_to(v56[:, None], (FRAGMENT_D1, FULL_D2))
    v75 = (v73 & v74)
    v76 = tl.reshape(v64, (1, FULL_D2), can_reorder=False)
    v77 = tl.broadcast_to(v76, (FRAGMENT_D1, FULL_D2))
    v78 = (v75 & v77)
    v79 = tl.fdiv(tl.cast(v50, tl.float32), tl.cast(v54, tl.float32), ieee_rounding=True)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v8, tl.int64) * S1_0 + tl.cast(0, tl.int64) * S1_1), shape=((D1 - tl.cast(v8, tl.int64)), (D2 - tl.cast(0, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FULL_D2), order=(1, 0)), tl.cast(v79, tl.float32), boundary_check=(0, 1))

def launch(x, output):
    D1 = x.shape[0]
    D2 = x.shape[1]
    S0_0 = x.stride(0)
    S0_1 = x.stride(1)
    S1_0 = output.stride(0)
    S1_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](x, output, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(x):
    output = torch.empty((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32)
    launch(x, output)
    return output
