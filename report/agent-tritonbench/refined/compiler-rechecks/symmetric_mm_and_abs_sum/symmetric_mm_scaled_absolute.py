import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("product", "C", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_S1_A0": 1, "FRAGMENT_S1_A1": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 1, "FRAGMENT_S1_A1": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 1, "FRAGMENT_S1_A1": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 1, "FRAGMENT_S1_A1": 256}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 64, "FRAGMENT_S1_A1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 64, "FRAGMENT_S1_A1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 64, "FRAGMENT_S1_A1": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 64, "FRAGMENT_S1_A1": 64}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 4, "FRAGMENT_S1_A1": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 4, "FRAGMENT_S1_A1": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 4, "FRAGMENT_S1_A1": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 4, "FRAGMENT_S1_A1": 1024}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 16, "FRAGMENT_S1_A1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 16, "FRAGMENT_S1_A1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 16, "FRAGMENT_S1_A1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 16, "FRAGMENT_S1_A1": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 8}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 2}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 2}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 2}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S1_A0": 8, "FRAGMENT_S1_A1": 2}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "S0_0", "S0_1", "S1_0", "S1_1", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(product, C, output, alpha, beta, D1: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr, FRAGMENT_S1_A1: tl.constexpr, FRAGMENT_S1_A0: tl.constexpr):
    v0 = (FRAGMENT_S1_A0 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_S1_A0) - (((v1 % FRAGMENT_S1_A0) != 0) & (((v1 % FRAGMENT_S1_A0) < 0) != (FRAGMENT_S1_A0 < 0))))
    v3 = (FRAGMENT_S1_A1 - 1)
    v4 = (D1 + v3)
    v5 = ((v4 // FRAGMENT_S1_A1) - (((v4 % FRAGMENT_S1_A1) != 0) & (((v4 % FRAGMENT_S1_A1) < 0) != (FRAGMENT_S1_A1 < 0))))
    v6 = tl.program_id(0)
    v7 = tl.program_id(1)
    v8 = (v6 * v5)
    v9 = (v8 + v7)
    v10 = ((v9 // v5) % v2)
    v11 = (v9 % v5)
    v12 = (v10 * FRAGMENT_S1_A0)
    v13 = (v12 * 1)
    v14 = (0 + v13)
    v15 = (v14 + tl.arange(0, FRAGMENT_S1_A0) * 1)
    v16 = (v11 * FRAGMENT_S1_A1)
    v17 = (v16 * 1)
    v18 = (0 + v17)
    v19 = (v18 + tl.arange(0, FRAGMENT_S1_A1) * 1)
    v20 = D1
    v21 = tl.full((FRAGMENT_S1_A0,), v20, tl.int64)
    v22 = (v15 < v21)
    v23 = tl.broadcast_to(v22[:, None], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v24 = D1
    v25 = tl.full((FRAGMENT_S1_A1,), v24, tl.int64)
    v26 = (v19 < v25)
    v27 = tl.broadcast_to(v26[None, :], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v28 = (v23 & v27)
    v29 = tl.full((FRAGMENT_S1_A0, FRAGMENT_S1_A1), 0.0, tl.float32)
    v30 = tl.full((FRAGMENT_S1_A0, FRAGMENT_S1_A1), alpha, tl.float32)
    v31 = (v12 * 1)
    v32 = (0 + v31)
    v33 = (v32 + tl.arange(0, FRAGMENT_S1_A0) * 1)
    v34 = (v16 * 1)
    v35 = (0 + v34)
    v36 = (v35 + tl.arange(0, FRAGMENT_S1_A1) * 1)
    v37 = D1
    v38 = tl.full((FRAGMENT_S1_A0,), v37, tl.int64)
    v39 = (v33 < v38)
    v40 = tl.broadcast_to(v39[:, None], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v41 = D1
    v42 = tl.full((FRAGMENT_S1_A1,), v41, tl.int64)
    v43 = (v36 < v42)
    v44 = tl.broadcast_to(v43[None, :], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v45 = (v40 & v44)
    v46 = tl.full((FRAGMENT_S1_A0, FRAGMENT_S1_A1), 0.0, tl.float32)
    v47 = tl.full((FRAGMENT_S1_A0, FRAGMENT_S1_A1), beta, tl.float32)
    v48 = (v12 * 1)
    v49 = (0 + v48)
    v50 = (v49 + tl.arange(0, FRAGMENT_S1_A0) * 1)
    v51 = (v16 * 1)
    v52 = (0 + v51)
    v53 = (v52 + tl.arange(0, FRAGMENT_S1_A1) * 1)
    v54 = D1
    v55 = tl.full((FRAGMENT_S1_A0,), v54, tl.int64)
    v56 = (v50 < v55)
    v57 = tl.broadcast_to(v56[:, None], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v58 = D1
    v59 = tl.full((FRAGMENT_S1_A1,), v58, tl.int64)
    v60 = (v53 < v59)
    v61 = tl.broadcast_to(v60[None, :], (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v62 = (v57 & v61)
    v63 = tl.load(tl.make_block_ptr(base=(product + tl.cast(v14, tl.int64) * S0_0 + tl.cast(v18, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D1 - tl.cast(v18, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_S1_A0, FRAGMENT_S1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    v64 = (v30 * v63)
    v65 = tl.load(tl.make_block_ptr(base=(C + tl.cast(v32, tl.int64) * S1_0 + tl.cast(v35, tl.int64) * S1_1), shape=((D1 - tl.cast(v32, tl.int64)), (D1 - tl.cast(v35, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_S1_A0, FRAGMENT_S1_A1), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    v66 = (v47 * v65)
    v67 = tl.broadcast_to(v66, (FRAGMENT_S1_A0, FRAGMENT_S1_A1))
    v68 = (v64 + v67)
    v69 = (-v68)
    v70 = tl.maximum(v68, v69, propagate_nan=tl.PropagateNan.ALL)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v49, tl.int64) * S4_0 + tl.cast(v52, tl.int64) * S4_1), shape=((D1 - tl.cast(v49, tl.int64)), (D1 - tl.cast(v52, tl.int64))), strides=(S4_0, S4_1), offsets=(0, 0), block_shape=(FRAGMENT_S1_A0, FRAGMENT_S1_A1), order=(1, 0)), tl.cast(v70, tl.float32), boundary_check=(0, 1))

def launch(product, C, alpha, beta, output):
    D1 = product.shape[0]
    S0_0 = product.stride(0)
    S0_1 = product.stride(1)
    S1_0 = C.stride(0)
    S1_1 = C.stride(1)
    S4_0 = output.stride(0)
    S4_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_S1_A0"]), triton.cdiv(D1, META["FRAGMENT_S1_A1"]))
    return _intent_kernel[grid](product, C, output, alpha, beta, D1, S0_0, S0_1, S1_0, S1_1, S4_0, S4_1)

def run(product, C, alpha, beta):
    output = torch.empty((product.shape[0], product.shape[0]), device=product.device, dtype=torch.float32)
    launch(product, C, alpha, beta, output)
    return output
