import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("a", "b", "output", ), (False, False, True, ))

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
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(a, b, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v28 = D2
    v29 = D3
    v30 = tl.full((FRAGMENT_D3,), v29, tl.int64)
    v31 = (v25 < v30)
    v32 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v33 = v32
    for iv34 in range(0, D2, BLOCK_K_1_2):
        v35 = (iv34 + tl.arange(0, BLOCK_K_1_2) * 1)
        v36 = (iv34 - 0)
        v37 = (0 + v36)
        v38 = (v37 + tl.arange(0, BLOCK_K_1_2) * 1)
        v39 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v40 = (v35 < v39)
        v41 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v42 = (v38 < v41)
        v43 = tl.full((BLOCK_K_1_2,), v21, tl.int64)
        v44 = (v35 < v43)
        v45 = tl.broadcast_to(v44[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v46 = tl.broadcast_to(v20[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v47 = (v46 & v45)
        v48 = tl.broadcast_to(v17[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v49 = (v47 & v48)
        v50 = tl.broadcast_to(v40[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v51 = (v49 & v50)
        v52 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float32)
        v53 = tl.full((BLOCK_K_1_2,), v28, tl.int64)
        v54 = (v38 < v53)
        v55 = tl.broadcast_to(v54[:, None], (BLOCK_K_1_2, FRAGMENT_D3))
        v56 = tl.broadcast_to(v31[None, :], (BLOCK_K_1_2, FRAGMENT_D3))
        v57 = (v55 & v56)
        v58 = tl.broadcast_to(v27[None, :], (BLOCK_K_1_2, FRAGMENT_D3))
        v59 = (v57 & v58)
        v60 = tl.broadcast_to(v42[:, None], (BLOCK_K_1_2, FRAGMENT_D3))
        v61 = (v59 & v60)
        v62 = tl.full((BLOCK_K_1_2, FRAGMENT_D3), 0.0, tl.float32)
        v63 = tl.load(tl.make_block_ptr(base=(b + tl.cast(v37, tl.int64) * S1_0 + tl.cast(v24, tl.int64) * S1_1), shape=((D2 - tl.cast(v37, tl.int64)), (D3 - tl.cast(v24, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(BLOCK_K_1_2, FRAGMENT_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v64 = tl.load(tl.make_block_ptr(base=(a + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv34, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv34, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v65 = tl.dot(v64, v63, v33, input_precision="ieee")
        v33 = v65
    v66 = (0 + tl.arange(0, 1) * 1)
    v67 = (0 + tl.arange(0, 1) * 1)
    v68 = tl.full((FRAGMENT_D1, FRAGMENT_D3), True, tl.int1)
    v69 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v70 = tl.full((1,), 1, tl.int64)
    v71 = (v66 < v70)
    v72 = tl.broadcast_to(v71[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v73 = (v68 & v72)
    v74 = tl.full((1,), 1, tl.int64)
    v75 = (v67 < v74)
    v76 = tl.broadcast_to(v75[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v77 = (v73 & v76)
    v78 = v33
    v79 = tl.where(v77, v78, v69)
    v80 = libdevice.rsqrt(tl.cast(v79, tl.float32))
    v81 = (v12 * 1)
    v82 = (0 + v81)
    v83 = (v82 + tl.arange(0, FRAGMENT_D1) * 1)
    v84 = (v83 < v16)
    v85 = (v22 * 1)
    v86 = (0 + v85)
    v87 = (v86 + tl.arange(0, FRAGMENT_D3) * 1)
    v88 = (v87 < v26)
    v89 = D1
    v90 = tl.full((FRAGMENT_D1,), v89, tl.int64)
    v91 = (v83 < v90)
    v92 = tl.broadcast_to(v91[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v93 = D3
    v94 = tl.full((FRAGMENT_D3,), v93, tl.int64)
    v95 = (v87 < v94)
    v96 = tl.broadcast_to(v95[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v97 = (v92 & v96)
    v98 = tl.broadcast_to(v84[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v99 = (v97 & v98)
    v100 = tl.broadcast_to(v88[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v101 = (v99 & v100)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v82, tl.int64) * S2_0 + tl.cast(v86, tl.int64) * S2_1), shape=((D1 - tl.cast(v82, tl.int64)), (D3 - tl.cast(v86, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v80, tl.float32), boundary_check=(0, 1))

def launch(a, b, output):
    D1 = a.shape[0]
    D2 = a.shape[1]
    D3 = b.shape[1]
    S0_0 = a.stride(0)
    S0_1 = a.stride(1)
    S1_0 = b.stride(0)
    S1_1 = b.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](a, b, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(a, b):
    output = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    launch(a, b, output)
    return output
