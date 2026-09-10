import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("a", "b", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "S0_0", "S0_1", "S0_2", "S1_0", "S1_1", "S1_2", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(a, b, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D4: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = (FRAGMENT_D4 - 1)
    v4 = (D4 + v3)
    v5 = ((v4 // FRAGMENT_D4) - (((v4 % FRAGMENT_D4) != 0) & (((v4 % FRAGMENT_D4) < 0) != (FRAGMENT_D4 < 0))))
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
    v22 = D3
    v23 = (v11 * FRAGMENT_D4)
    v24 = (v23 * 1)
    v25 = (0 + v24)
    v26 = (v25 + tl.arange(0, FRAGMENT_D4) * 1)
    v27 = tl.full((FRAGMENT_D4,), D4, tl.int64)
    v28 = (v26 < v27)
    v29 = D2
    v30 = D3
    v31 = D4
    v32 = tl.full((FRAGMENT_D4,), v31, tl.int64)
    v33 = (v26 < v32)
    v34 = tl.full((FRAGMENT_D1, FRAGMENT_D4), 0.0, tl.float32)
    v35 = v34
    for iv36 in range(0, D2, 1):
        v37 = tl.broadcast_to(v20[:, None], (FRAGMENT_D1, 1))
        v38 = (iv36 < v21)
        v39 = tl.full((FRAGMENT_D1, 1), v38, tl.int1)
        v40 = (v37 & v39)
        v41 = (iv36 < v29)
        v42 = v35
        for iv43 in range(0, D3, BLOCK_K_1_2):
            v44 = (iv43 + tl.arange(0, BLOCK_K_1_2) * 1)
            v45 = (iv43 - 0)
            v46 = (0 + v45)
            v47 = (v46 + tl.arange(0, BLOCK_K_1_2) * 1)
            v48 = tl.full((BLOCK_K_1_2,), D3, tl.int64)
            v49 = (v44 < v48)
            v50 = tl.full((BLOCK_K_1_2,), D3, tl.int64)
            v51 = (v47 < v50)
            v52 = tl.full((BLOCK_K_1_2,), v22, tl.int64)
            v53 = (v44 < v52)
            v54 = tl.broadcast_to(v53[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
            v55 = tl.broadcast_to(v40, (FRAGMENT_D1, BLOCK_K_1_2))
            v56 = (v55 & v54)
            v57 = tl.broadcast_to(v17[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
            v58 = (v56 & v57)
            v59 = tl.broadcast_to(v49[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
            v60 = (v58 & v59)
            v61 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float16)
            v62 = tl.full((BLOCK_K_1_2,), v30, tl.int64)
            v63 = (v47 < v62)
            v64 = tl.broadcast_to(v63[:, None], (BLOCK_K_1_2, FRAGMENT_D4))
            v65 = tl.full((BLOCK_K_1_2, FRAGMENT_D4), v41, tl.int1)
            v66 = (v65 & v64)
            v67 = tl.broadcast_to(v33[None, :], (BLOCK_K_1_2, FRAGMENT_D4))
            v68 = (v66 & v67)
            v69 = tl.broadcast_to(v28[None, :], (BLOCK_K_1_2, FRAGMENT_D4))
            v70 = (v68 & v69)
            v71 = tl.broadcast_to(v51[:, None], (BLOCK_K_1_2, FRAGMENT_D4))
            v72 = (v70 & v71)
            v73 = tl.full((BLOCK_K_1_2, FRAGMENT_D4), 0.0, tl.float16)
            v74 = tl.load((b + (tl.full((BLOCK_K_1_2, FRAGMENT_D4), iv36, tl.int64)) * S1_0 + (tl.broadcast_to(v47[:, None], (BLOCK_K_1_2, FRAGMENT_D4))) * S1_1 + (tl.broadcast_to(v26[None, :], (BLOCK_K_1_2, FRAGMENT_D4))) * S1_2), mask=v72, other=v73)
            v75 = tl.load((a + (tl.broadcast_to(v15[:, None], (FRAGMENT_D1, BLOCK_K_1_2))) * S0_0 + (tl.full((FRAGMENT_D1, BLOCK_K_1_2), iv36, tl.int64)) * S0_1 + (tl.broadcast_to(v44[None, :], (FRAGMENT_D1, BLOCK_K_1_2))) * S0_2), mask=v60, other=v61)
            v76 = tl.dot(v75, v74, v42, input_precision="ieee")
            v42 = v76
        v35 = v42
    v77 = (0 + tl.arange(0, 1) * 1)
    v78 = (0 + tl.arange(0, 1) * 1)
    v79 = tl.full((FRAGMENT_D1, FRAGMENT_D4), True, tl.int1)
    v80 = tl.full((FRAGMENT_D1, FRAGMENT_D4), 0.0, tl.float32)
    v81 = tl.full((1,), 1, tl.int64)
    v82 = (v77 < v81)
    v83 = tl.broadcast_to(v82[:, None], (FRAGMENT_D1, FRAGMENT_D4))
    v84 = (v79 & v83)
    v85 = tl.full((1,), 1, tl.int64)
    v86 = (v78 < v85)
    v87 = tl.broadcast_to(v86[None, :], (FRAGMENT_D1, FRAGMENT_D4))
    v88 = (v84 & v87)
    v89 = v35
    v90 = tl.where(v88, v89, v80)
    v91 = tl.cast(v90, tl.float16)
    v92 = (v12 * 1)
    v93 = (0 + v92)
    v94 = (v93 + tl.arange(0, FRAGMENT_D1) * 1)
    v95 = (v94 < v16)
    v96 = (v23 * 1)
    v97 = (0 + v96)
    v98 = (v97 + tl.arange(0, FRAGMENT_D4) * 1)
    v99 = (v98 < v27)
    v100 = D1
    v101 = tl.full((FRAGMENT_D1,), v100, tl.int64)
    v102 = (v94 < v101)
    v103 = tl.broadcast_to(v102[:, None], (FRAGMENT_D1, FRAGMENT_D4))
    v104 = D4
    v105 = tl.full((FRAGMENT_D4,), v104, tl.int64)
    v106 = (v98 < v105)
    v107 = tl.broadcast_to(v106[None, :], (FRAGMENT_D1, FRAGMENT_D4))
    v108 = (v103 & v107)
    v109 = tl.broadcast_to(v95[:, None], (FRAGMENT_D1, FRAGMENT_D4))
    v110 = (v108 & v109)
    v111 = tl.broadcast_to(v99[None, :], (FRAGMENT_D1, FRAGMENT_D4))
    v112 = (v110 & v111)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v93, tl.int64) * S2_0 + tl.cast(v97, tl.int64) * S2_1), shape=((D1 - tl.cast(v93, tl.int64)), (D4 - tl.cast(v97, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D4), order=(1, 0)), tl.cast(v91, tl.float16), boundary_check=(0, 1))

def launch(a, b, output):
    D1 = a.shape[0]
    D2 = a.shape[1]
    D3 = a.shape[2]
    D4 = b.shape[2]
    S0_0 = a.stride(0)
    S0_1 = a.stride(1)
    S0_2 = a.stride(2)
    S1_0 = b.stride(0)
    S1_1 = b.stride(1)
    S1_2 = b.stride(2)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D4, META["FRAGMENT_D4"]))
    return _intent_kernel[grid](a, b, output, D1, D2, D3, D4, S0_0, S0_1, S0_2, S1_0, S1_1, S1_2, S2_0, S2_1)

def run(a, b):
    output = torch.empty((a.shape[0], b.shape[2]), device=a.device, dtype=torch.float16)
    launch(a, b, output)
    return output
