import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("a", "b", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 64, "FRAGMENT_D4": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 32, "FRAGMENT_D1": 128, "FRAGMENT_D4": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_11_11": 64, "FRAGMENT_D1": 128, "FRAGMENT_D4": 256}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "D4", "S0_0", "S0_1", "S0_2", "S1_0", "S1_1", "S1_2", "S2_0", "S2_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(a, b, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S1_2: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, BLOCK_K_11_11: tl.constexpr, FRAGMENT_D4: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v16 = D1
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
    v27 = D4
    v28 = (v26 < v27)
    v29 = D2
    v30 = D3
    v31 = D4
    v32 = tl.full((FRAGMENT_D4,), v31, tl.int64)
    v33 = (v26 < v32)
    v34 = tl.full((FRAGMENT_D1, FRAGMENT_D4), 0.0, tl.float32)
    v35 = v34
    for iv36 in range(0, D2, 1):
        v37 = v20[:, None]
        v38 = (iv36 < v21)
        v39 = tl.full((FRAGMENT_D1, 1), v38, tl.int1)
        v40 = (v37 & v39)
        v41 = (iv36 < v29)
        v42 = v35
        for iv43 in range(0, D3, BLOCK_K_11_11):
            v44 = (iv43 + tl.arange(0, BLOCK_K_11_11) * 1)
            v45 = (iv43 - 0)
            v46 = (0 + v45)
            v47 = (v46 + tl.arange(0, BLOCK_K_11_11) * 1)
            v48 = tl.full((BLOCK_K_11_11,), D3, tl.int64)
            v49 = (v44 < v48)
            v50 = (v47 < v48)
            v51 = tl.full((BLOCK_K_11_11,), v22, tl.int64)
            v52 = (v44 < v51)
            v53 = v52[None, :]
            v54 = tl.broadcast_to(v40, (FRAGMENT_D1, BLOCK_K_11_11, ))
            v55 = (v54 & v53)
            v56 = v17[:, None]
            v57 = (v55 & v56)
            v58 = v49[None, :]
            v59 = (v57 & v58)
            v60 = tl.full((FRAGMENT_D1, BLOCK_K_11_11), 0.0, tl.float16)
            v61 = tl.full((BLOCK_K_11_11,), v30, tl.int64)
            v62 = (v47 < v61)
            v63 = v62[:, None]
            v64 = tl.full((BLOCK_K_11_11, FRAGMENT_D4), v41, tl.int1)
            v65 = (v64 & v63)
            v66 = v33[None, :]
            v67 = (v65 & v66)
            v68 = v28[None, :]
            v69 = (v67 & v68)
            v70 = v50[:, None]
            v71 = (v69 & v70)
            v72 = tl.full((BLOCK_K_11_11, FRAGMENT_D4), 0.0, tl.float16)
            v73 = tl.load((b + (iv36) * S1_0 + (v47[:, None]) * S1_1 + (v26[None, :]) * S1_2), mask=v71, other=v72)
            v74 = tl.load((a + (v15[:, None]) * S0_0 + (iv36) * S0_1 + (v44[None, :]) * S0_2), mask=v59, other=v60)
            v75 = tl.dot(v74, v73, v42, input_precision="ieee")
            v42 = v75
        v35 = v42
    v76 = tl.cast(v35, tl.float16)
    v77 = (v14 + tl.arange(0, FRAGMENT_D1) * 1)
    v78 = (v77 < v16)
    v79 = (v25 + tl.arange(0, FRAGMENT_D4) * 1)
    v80 = (v79 < v27)
    v81 = D1
    v82 = tl.full((FRAGMENT_D1,), v81, tl.int64)
    v83 = (v77 < v82)
    v84 = v83[:, None]
    v85 = D4
    v86 = tl.full((FRAGMENT_D4,), v85, tl.int64)
    v87 = (v79 < v86)
    v88 = v87[None, :]
    v89 = (v84 & v88)
    v90 = v78[:, None]
    v91 = (v89 & v90)
    v92 = v80[None, :]
    v93 = (v91 & v92)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v14, tl.int64) * S2_0 + tl.cast(v25, tl.int64) * S2_1), shape=((D1 - tl.cast(v14, tl.int64)), (D4 - tl.cast(v25, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D4), order=(1, 0)), tl.cast(v76, tl.float16), boundary_check=(0, 1))

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
