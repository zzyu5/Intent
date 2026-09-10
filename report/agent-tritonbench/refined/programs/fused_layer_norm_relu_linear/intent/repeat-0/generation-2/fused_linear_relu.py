import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "bias", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight, bias, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = (FRAGMENT_D3 - 1)
    v4 = (D3 + v3)
    v5 = ((v4 // FRAGMENT_D3) - (((v4 % FRAGMENT_D3) != 0) & (((v4 % FRAGMENT_D3) < 0) != (FRAGMENT_D3 < 0))))
    v6 = tl.program_id(0)
    v7 = tl.program_id(1)
    v8 = (v6 * v5)
    v9 = (v8 + v7)
    v10 = ((v9 // v5) % v2)
    v11 = (v9 % v5)
    v12 = (v10 * FRAGMENT_D1)
    v13 = (v12 * 1)
    v14 = (0 + v13)
    v15 = (v14 + tl.arange(0, FRAGMENT_D1) * 1)
    v16 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v17 = (v15 < v16)
    v18 = D1
    v19 = tl.full((FRAGMENT_D1,), v18, tl.int64)
    v20 = (v15 < v19)
    v21 = D2
    v22 = (v11 * FRAGMENT_D3)
    v23 = (v22 * 1)
    v24 = (0 + v23)
    v25 = (v24 + tl.arange(0, FRAGMENT_D3) * 1)
    v26 = tl.full((FRAGMENT_D3,), D3, tl.int64)
    v27 = (v25 < v26)
    v28 = D3
    v29 = tl.full((FRAGMENT_D3,), v28, tl.int64)
    v30 = (v25 < v29)
    v31 = D2
    v32 = (v22 * 1)
    v33 = (0 + v32)
    v34 = (v33 + tl.arange(0, FRAGMENT_D3) * 1)
    v35 = tl.full((FRAGMENT_D3,), D3, tl.int64)
    v36 = (v34 < v35)
    v37 = D3
    v38 = tl.full((FRAGMENT_D3,), v37, tl.int64)
    v39 = (v34 < v38)
    v40 = tl.full((FRAGMENT_D3,), 0.0, tl.float32)
    v41 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v42 = (v12 * 1)
    v43 = (0 + v42)
    v44 = (v43 + tl.arange(0, FRAGMENT_D1) * 1)
    v45 = (v44 < v16)
    v46 = (v22 * 1)
    v47 = (0 + v46)
    v48 = (v47 + tl.arange(0, FRAGMENT_D3) * 1)
    v49 = (v48 < v26)
    v50 = D1
    v51 = tl.full((FRAGMENT_D1,), v50, tl.int64)
    v52 = (v44 < v51)
    v53 = tl.broadcast_to(v52[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v54 = D3
    v55 = tl.full((FRAGMENT_D3,), v54, tl.int64)
    v56 = (v48 < v55)
    v57 = tl.broadcast_to(v56[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v58 = (v53 & v57)
    v59 = (v39 & v36)
    v60 = tl.load(tl.make_block_ptr(base=(bias + tl.cast(v33, tl.int64) * S2_0), shape=((D3 - tl.cast(v33, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D3,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v61 = tl.broadcast_to(v60[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v62 = v61
    for iv63 in range(0, D2, BLOCK_K_1_2):
        v64 = (iv63 + tl.arange(0, BLOCK_K_1_2) * 1)
        v65 = (iv63 - 0)
        v66 = (0 + v65)
        v67 = (v66 + tl.arange(0, BLOCK_K_1_2) * 1)
        v68 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v69 = (v64 < v68)
        v70 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v71 = (v67 < v70)
        v72 = tl.full((BLOCK_K_1_2,), v21, tl.int64)
        v73 = (v64 < v72)
        v74 = tl.broadcast_to(v73[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v75 = tl.broadcast_to(v20[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v76 = (v75 & v74)
        v77 = tl.broadcast_to(v17[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v78 = (v76 & v77)
        v79 = tl.broadcast_to(v69[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v80 = (v78 & v79)
        v81 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float32)
        v82 = tl.full((BLOCK_K_1_2,), v31, tl.int64)
        v83 = (v67 < v82)
        v84 = tl.broadcast_to(v83[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v85 = tl.broadcast_to(v30[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v86 = (v85 & v84)
        v87 = tl.broadcast_to(v27[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v88 = (v86 & v87)
        v89 = tl.broadcast_to(v71[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v90 = (v88 & v89)
        v91 = tl.full((FRAGMENT_D3, BLOCK_K_1_2), 0.0, tl.float32)
        v92 = tl.load(tl.make_block_ptr(base=(weight + tl.cast(v24, tl.int64) * S1_0 + tl.cast(v66, tl.int64) * S1_1), shape=((D3 - tl.cast(v24, tl.int64)), (D2 - tl.cast(v66, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D3, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v93 = tl.permute(v92, (1, 0))
        v94 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv63, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv63, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v95 = tl.dot(v94, v93, v62, input_precision="ieee")
        v62 = v95
    v96 = tl.maximum(v62, v41, propagate_nan=tl.PropagateNan.ALL)
    v97 = tl.broadcast_to(v45[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v98 = (v58 & v97)
    v99 = tl.broadcast_to(v49[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v100 = (v98 & v99)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v43, tl.int64) * S3_0 + tl.cast(v47, tl.int64) * S3_1), shape=((D1 - tl.cast(v43, tl.int64)), (D3 - tl.cast(v47, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v96, tl.float32), boundary_check=(0, 1))

def launch(input, weight, bias, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = weight.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S2_0 = bias.stride(0)
    S3_0 = output.stride(0)
    S3_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](input, weight, bias, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S3_0, S3_1)

def run(input, weight, bias):
    output = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output)
    return output
