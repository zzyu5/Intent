import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 32, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 256, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 256, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 64, "BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 128, "BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0": 256, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, other, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, GROUP_SIZE_M: tl.constexpr, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr, BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0: tl.constexpr):
    v0 = D1
    v1 = D3
    v2 = (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0 - 1)
    v3 = (v0 + v2)
    v4 = ((v3 // BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) - (((v3 % BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) != 0) & (((v3 % BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) < 0) != (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0 < 0))))
    v5 = (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0 - 1)
    v6 = (v1 + v5)
    v7 = ((v6 // BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) - (((v6 % BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) != 0) & (((v6 % BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) < 0) != (BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0 < 0))))
    v8 = tl.program_id(0)
    v9 = tl.program_id(1)
    v10 = (v8 * v7)
    v11 = (v10 + v9)
    v12 = ((v11 // v7) % v4)
    v13 = (v11 % v7)
    v14 = ((v12 // GROUP_SIZE_M) - (((v12 % GROUP_SIZE_M) != 0) & (((v12 % GROUP_SIZE_M) < 0) != (GROUP_SIZE_M < 0))))
    v15 = (v14 * GROUP_SIZE_M)
    v16 = (v4 - v15)
    v17 = tl.minimum(v16, GROUP_SIZE_M, propagate_nan=tl.PropagateNan.NONE)
    v18 = ((v12 % GROUP_SIZE_M) + (((v12 % GROUP_SIZE_M) != 0) & (((v12 % GROUP_SIZE_M) < 0) != (GROUP_SIZE_M < 0))) * GROUP_SIZE_M)
    v19 = (v18 * v7)
    v20 = (v19 + v13)
    v21 = ((v20 % v17) + (((v20 % v17) != 0) & (((v20 % v17) < 0) != (v17 < 0))) * v17)
    v22 = (v15 + v21)
    v23 = ((v20 // v17) - (((v20 % v17) != 0) & (((v20 % v17) < 0) != (v17 < 0))))
    v24 = (v22 * BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0)
    v25 = (v24 * 1)
    v26 = (0 + v25)
    v27 = (v23 * BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0)
    v28 = (v27 * 1)
    v29 = (0 + v28)
    v30 = (v29 + tl.arange(0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
    v31 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
    v32 = (v30 >= v31)
    v33 = tl.full((BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D3, tl.int64)
    v34 = (v30 < v33)
    v35 = (v32 & v34)
    v36 = (v26 + tl.arange(0, BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
    v37 = tl.full((BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
    v38 = (v36 >= v37)
    v39 = tl.full((BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D1, tl.int64)
    v40 = (v36 < v39)
    v41 = (v38 & v40)
    v42 = tl.full((BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float32)
    v43 = v42
    for iv44 in range(0, D2, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0):
        v45 = (iv44 + tl.arange(0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0) * 1)
        v46 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), 0, tl.int64)
        v47 = (v45 >= v46)
        v48 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0,), D2, tl.int64)
        v49 = (v45 < v48)
        v50 = (v47 & v49)
        v51 = tl.broadcast_to(v41[:, None], (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v52 = tl.broadcast_to(v50[None, :], (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v53 = (v51 & v52)
        v54 = tl.broadcast_to(v50[:, None], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v55 = tl.broadcast_to(v35[None, :], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
        v56 = (v54 & v55)
        v57 = tl.full((BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float16)
        v58 = tl.full((BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), 0.0, tl.float16)
        v59 = tl.load((other + (tl.broadcast_to(v45[:, None], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))) * S1_0 + (tl.broadcast_to(v30[None, :], (BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))) * S1_1), mask=v56, other=v58)
        v60 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v26, tl.int64) * S0_0 + tl.cast(iv44, tl.int64) * S0_1), shape=((D1 - tl.cast(v26, tl.int64)), (D2 - tl.cast(iv44, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_K_1_0_0_1_1_0_2_0_0_2_1_0_3_0), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v61 = tl.dot(v60, v59, v43, input_precision="ieee")
        v43 = v61
    v62 = tl.broadcast_to(v41[:, None], (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
    v63 = tl.broadcast_to(v35[None, :], (BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0))
    v64 = (v62 & v63)
    v65 = tl.cast(v43, tl.float16)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v26, tl.int64) * S2_0 + tl.cast(v29, tl.int64) * S2_1), shape=((D1 - tl.cast(v26, tl.int64)), (D3 - tl.cast(v29, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0, BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0), order=(1, 0)), tl.cast(v65, tl.float16), boundary_check=(0, 1))

def launch(input, other, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = other.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = other.stride(0)
    S1_1 = other.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["BLOCK_M_1_0_0_1_1_0_2_0_0_2_1_0_3_0"]), triton.cdiv(D3, META["BLOCK_N_1_0_0_1_1_0_2_0_0_2_1_0_3_0"]))
    return _intent_kernel[grid](input, other, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(input, other):
    output = torch.empty((input.shape[0], other.shape[1]), device=input.device, dtype=torch.float16)
    launch(input, other, output)
    return output
