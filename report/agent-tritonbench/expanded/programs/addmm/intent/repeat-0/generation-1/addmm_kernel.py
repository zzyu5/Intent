import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "mat1", "mat2", "output", ), (False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 128}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 32, "FRAGMENT_D1": 128, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_2_3": 64, "FRAGMENT_D1": 128, "FRAGMENT_D3": 256}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S2_1", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, mat1, mat2, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, BLOCK_K_2_3: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v16 = D1
    v17 = (v15 < v16)
    v18 = D1
    v19 = tl.full((FRAGMENT_D1,), v18, tl.int64)
    v20 = (v15 < v19)
    v21 = D2
    v22 = (v11 * FRAGMENT_D3)
    v23 = (v22 * 1)
    v24 = (0 + v23)
    v25 = (v24 + tl.arange(0, FRAGMENT_D3) * 1)
    v26 = D3
    v27 = (v25 < v26)
    v28 = D2
    v29 = D3
    v30 = tl.full((FRAGMENT_D3,), v29, tl.int64)
    v31 = (v25 < v30)
    v32 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v33 = v32
    for iv34 in range(0, D2, BLOCK_K_2_3):
        v35 = (iv34 + tl.arange(0, BLOCK_K_2_3) * 1)
        v36 = (iv34 - 0)
        v37 = (0 + v36)
        v38 = (v37 + tl.arange(0, BLOCK_K_2_3) * 1)
        v39 = tl.full((BLOCK_K_2_3,), D2, tl.int64)
        v40 = (v35 < v39)
        v41 = tl.full((BLOCK_K_2_3,), D2, tl.int64)
        v42 = (v38 < v41)
        v43 = tl.full((BLOCK_K_2_3,), v21, tl.int64)
        v44 = (v35 < v43)
        v45 = v44[None, :]
        v46 = v20[:, None]
        v47 = (v46 & v45)
        v48 = v17[:, None]
        v49 = (v47 & v48)
        v50 = v40[None, :]
        v51 = (v49 & v50)
        v52 = tl.full((FRAGMENT_D1, BLOCK_K_2_3), 0.0, tl.float16)
        v53 = tl.full((BLOCK_K_2_3,), v28, tl.int64)
        v54 = (v38 < v53)
        v55 = v54[:, None]
        v56 = v31[None, :]
        v57 = (v55 & v56)
        v58 = v27[None, :]
        v59 = (v57 & v58)
        v60 = v42[:, None]
        v61 = (v59 & v60)
        v62 = tl.full((BLOCK_K_2_3, FRAGMENT_D3), 0.0, tl.float16)
        v63 = tl.load(tl.make_block_ptr(base=(mat2 + tl.cast(v37, tl.int64) * S2_0 + tl.cast(v24, tl.int64) * S2_1), shape=((D2 - tl.cast(v37, tl.int64)), (D3 - tl.cast(v24, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(BLOCK_K_2_3, FRAGMENT_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v64 = tl.load(tl.make_block_ptr(base=(mat1 + tl.cast(v14, tl.int64) * S1_0 + tl.cast(iv34, tl.int64) * S1_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv34, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_2_3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v65 = tl.dot(v64, v63, v33, input_precision="ieee")
        v33 = v65
    v66 = (v12 * 1)
    v67 = (0 + v66)
    v68 = (v67 + tl.arange(0, FRAGMENT_D1) * 1)
    v69 = D1
    v70 = (v68 < v69)
    v71 = (v22 * 1)
    v72 = (0 + v71)
    v73 = (v72 + tl.arange(0, FRAGMENT_D3) * 1)
    v74 = D3
    v75 = (v73 < v74)
    v76 = D1
    v77 = tl.full((FRAGMENT_D1,), v76, tl.int64)
    v78 = (v68 < v77)
    v79 = v78[:, None]
    v80 = D3
    v81 = tl.full((FRAGMENT_D3,), v80, tl.int64)
    v82 = (v73 < v81)
    v83 = v82[None, :]
    v84 = (v79 & v83)
    v85 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float16)
    v86 = (0 + tl.arange(0, 1) * 1)
    v87 = (0 + tl.arange(0, 1) * 1)
    v88 = True
    v89 = 0.0
    v90 = tl.full((1,), 1, tl.int64)
    v91 = (v86 < v90)
    v92 = v91[:, None]
    v93 = (v88 & v92)
    v94 = tl.full((1,), 1, tl.int64)
    v95 = (v87 < v94)
    v96 = v95[None, :]
    v97 = (v93 & v96)
    v98 = v33
    v99 = tl.where(v97, v98, v89)
    v100 = tl.broadcast_to(v99, (FRAGMENT_D1, FRAGMENT_D3, ))
    v101 = (v67 + tl.arange(0, FRAGMENT_D1) * 1)
    v102 = (v101 < v69)
    v103 = (v72 + tl.arange(0, FRAGMENT_D3) * 1)
    v104 = (v103 < v74)
    v105 = D1
    v106 = tl.full((FRAGMENT_D1,), v105, tl.int64)
    v107 = (v101 < v106)
    v108 = v107[:, None]
    v109 = D3
    v110 = tl.full((FRAGMENT_D3,), v109, tl.int64)
    v111 = (v103 < v110)
    v112 = v111[None, :]
    v113 = (v108 & v112)
    v114 = v70[:, None]
    v115 = (v84 & v114)
    v116 = v75[None, :]
    v117 = (v115 & v116)
    v118 = v102[:, None]
    v119 = (v113 & v118)
    v120 = v104[None, :]
    v121 = (v119 & v120)
    v122 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v67, tl.int64) * S0_0 + tl.cast(v72, tl.int64) * S0_1), shape=((D1 - tl.cast(v67, tl.int64)), (D3 - tl.cast(v72, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
    v123 = tl.cast(v122, tl.float32)
    v124 = (v123 + v100)
    v125 = tl.cast(v124, tl.float16)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v67, tl.int64) * S3_0 + tl.cast(v72, tl.int64) * S3_1), shape=((D1 - tl.cast(v67, tl.int64)), (D3 - tl.cast(v72, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v125, tl.float16), boundary_check=(0, 1))

def launch(input, mat1, mat2, output):
    D1 = input.shape[0]
    D2 = mat1.shape[1]
    D3 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = mat1.stride(0)
    S1_1 = mat1.stride(1)
    S2_0 = mat2.stride(0)
    S2_1 = mat2.stride(1)
    S3_0 = output.stride(0)
    S3_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](input, mat1, mat2, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1, S3_0, S3_1)

def run(input, mat1, mat2):
    output = torch.empty((input.shape[0], input.shape[1]), device=input.device, dtype=torch.float16)
    launch(input, mat1, mat2, output)
    return output
