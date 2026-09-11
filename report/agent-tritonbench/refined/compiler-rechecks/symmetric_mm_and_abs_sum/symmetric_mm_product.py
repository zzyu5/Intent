import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("A", "AT", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(A, AT, output, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr, BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr):
    v0 = D1
    v1 = D1
    v2 = (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0 - 1)
    v3 = (v0 + v2)
    v4 = ((v3 // BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) - (((v3 % BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) != 0) & (((v3 % BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) < 0) != (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0 < 0))))
    v5 = (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0 - 1)
    v6 = (v1 + v5)
    v7 = ((v6 // BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) - (((v6 % BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) != 0) & (((v6 % BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) < 0) != (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0 < 0))))
    v8 = tl.program_id(0)
    v9 = tl.program_id(1)
    v10 = tl.program_id(2)
    v11 = (v8 * v4)
    v12 = (v11 + v9)
    v13 = (v12 * v7)
    v14 = (v13 + v10)
    v15 = (((v14 // v7) // v4) % 1)
    v16 = ((v14 // v7) % v4)
    v17 = (v14 % v7)
    v18 = ((v16 // GROUP_SIZE_M) - (((v16 % GROUP_SIZE_M) != 0) & (((v16 % GROUP_SIZE_M) < 0) != (GROUP_SIZE_M < 0))))
    v19 = (v18 * GROUP_SIZE_M)
    v20 = (v4 - v19)
    v21 = tl.minimum(v20, GROUP_SIZE_M, propagate_nan=tl.PropagateNan.NONE)
    v22 = ((v16 % GROUP_SIZE_M) + (((v16 % GROUP_SIZE_M) != 0) & (((v16 % GROUP_SIZE_M) < 0) != (GROUP_SIZE_M < 0))) * GROUP_SIZE_M)
    v23 = (v22 * v7)
    v24 = (v23 + v17)
    v25 = ((v24 % v21) + (((v24 % v21) != 0) & (((v24 % v21) < 0) != (v21 < 0))) * v21)
    v26 = (v19 + v25)
    v27 = ((v24 // v21) - (((v24 % v21) != 0) & (((v24 % v21) < 0) != (v21 < 0))))
    v28 = (v27 * BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0)
    v29 = (0 + v28)
    v30 = BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0
    v31 = (v29 + tl.arange(0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
    v32 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
    v33 = (v31 >= v32)
    v34 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D1, tl.int64)
    v35 = (v31 < v34)
    v36 = (v33 & v35)
    v37 = (v26 * BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0)
    v38 = (0 + v37)
    v39 = BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0
    v40 = (v38 + tl.arange(0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
    v41 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
    v42 = (v40 >= v41)
    v43 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D1, tl.int64)
    v44 = (v40 < v43)
    v45 = (v42 & v44)
    v46 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float32)
    v47 = v46
    for iv48 in range(0, D2, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0):
        v49 = (iv48 + tl.arange(0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
        v50 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
        v51 = (v49 >= v50)
        v52 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D2, tl.int64)
        v53 = (v49 < v52)
        v54 = (v51 & v53)
        v55 = tl.reshape(v45, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, 1), can_reorder=False)
        v56 = tl.broadcast_to(v55, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v57 = tl.reshape(v54, (1, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0), can_reorder=False)
        v58 = tl.broadcast_to(v57, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v59 = (v56 & v58)
        v60 = tl.reshape(v54, (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, 1), can_reorder=False)
        v61 = tl.broadcast_to(v60, (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v62 = tl.reshape(v36, (1, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), can_reorder=False)
        v63 = tl.broadcast_to(v62, (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v64 = (v61 & v63)
        v65 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float32)
        v66 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float32)
        v67 = tl.load((AT + (tl.broadcast_to(v49[:, None], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))) * S1_0 + (tl.broadcast_to(v31[None, :], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))) * S1_1), mask=v64, other=v66)
        v68 = tl.load(tl.make_block_ptr(base=(A + tl.cast(v38, tl.int64) * S0_0 + tl.cast(iv48, tl.int64) * S0_1), shape=((D1 - tl.cast(v38, tl.int64)), (D2 - tl.cast(iv48, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v69 = tl.dot(v68, v67, v47, input_precision="ieee", out_dtype=tl.float32)
        v47 = v69
    v70 = tl.reshape(v45, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, 1), can_reorder=False)
    v71 = tl.broadcast_to(v70, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
    v72 = tl.reshape(v36, (1, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), can_reorder=False)
    v73 = tl.broadcast_to(v72, (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
    v74 = (v71 & v73)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v38, tl.int64) * S2_0 + tl.cast(v29, tl.int64) * S2_1), shape=((D1 - tl.cast(v38, tl.int64)), (D1 - tl.cast(v29, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), order=(1, 0)), tl.cast(v47, tl.float32), boundary_check=(0, 1))

def launch(A, AT, output):
    D1 = A.shape[0]
    D2 = A.shape[1]
    S0_0 = A.stride(0)
    S0_1 = A.stride(1)
    S1_0 = AT.stride(0)
    S1_1 = AT.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (1, triton.cdiv(D1, META["BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0"]), triton.cdiv(D1, META["BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0"]))
    return _intent_kernel[grid](A, AT, output, D1, D2, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(A, AT):
    output = torch.empty((A.shape[0], A.shape[0]), device=A.device, dtype=torch.float32)
    launch(A, AT, output)
    return output
