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
    v41 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.4426950216293335, tl.float32)
    v42 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v43 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.066666670143604279, tl.float32)
    v44 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.076923079788684845, tl.float32)
    v45 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.090909093618392944, tl.float32)
    v46 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1111111119389534, tl.float32)
    v47 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1428571492433548, tl.float32)
    v48 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.20000000298023224, tl.float32)
    v49 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.3333333432674408, tl.float32)
    v50 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.0, tl.float32)
    v51 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v52 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v53 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 20.0, tl.float32)
    v54 = (v39 & v36)
    v55 = tl.load(tl.make_block_ptr(base=(bias + tl.cast(v33, tl.int64) * S2_0), shape=((D3 - tl.cast(v33, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D3,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v56 = tl.broadcast_to(v55[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v57 = v56
    for iv58 in range(0, D2, BLOCK_K_1_2):
        v59 = (iv58 + tl.arange(0, BLOCK_K_1_2) * 1)
        v60 = (iv58 - 0)
        v61 = (0 + v60)
        v62 = (v61 + tl.arange(0, BLOCK_K_1_2) * 1)
        v63 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v64 = (v59 < v63)
        v65 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v66 = (v62 < v65)
        v67 = tl.full((BLOCK_K_1_2,), v21, tl.int64)
        v68 = (v59 < v67)
        v69 = tl.broadcast_to(v68[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v70 = tl.broadcast_to(v20[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v71 = (v70 & v69)
        v72 = tl.broadcast_to(v17[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v73 = (v71 & v72)
        v74 = tl.broadcast_to(v64[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v75 = (v73 & v74)
        v76 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float32)
        v77 = tl.full((BLOCK_K_1_2,), v31, tl.int64)
        v78 = (v62 < v77)
        v79 = tl.broadcast_to(v78[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v80 = tl.broadcast_to(v30[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v81 = (v80 & v79)
        v82 = tl.broadcast_to(v27[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v83 = (v81 & v82)
        v84 = tl.broadcast_to(v66[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v85 = (v83 & v84)
        v86 = tl.full((FRAGMENT_D3, BLOCK_K_1_2), 0.0, tl.float32)
        v87 = tl.load(tl.make_block_ptr(base=(weight + tl.cast(v24, tl.int64) * S1_0 + tl.cast(v61, tl.int64) * S1_1), shape=((D3 - tl.cast(v24, tl.int64)), (D2 - tl.cast(v61, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D3, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v88 = tl.permute(v87, (1, 0))
        v89 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv58, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv58, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v90 = tl.dot(v89, v88, v57, input_precision="ieee")
        v57 = v90
    v91 = (v57 > v53)
    v92 = (v12 * 1)
    v93 = (0 + v92)
    v94 = (v93 + tl.arange(0, FRAGMENT_D1) * 1)
    v95 = (v94 < v16)
    v96 = (v22 * 1)
    v97 = (0 + v96)
    v98 = (v97 + tl.arange(0, FRAGMENT_D3) * 1)
    v99 = (v98 < v26)
    v100 = D1
    v101 = tl.full((FRAGMENT_D1,), v100, tl.int64)
    v102 = (v94 < v101)
    v103 = tl.broadcast_to(v102[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v104 = D3
    v105 = tl.full((FRAGMENT_D3,), v104, tl.int64)
    v106 = (v98 < v105)
    v107 = tl.broadcast_to(v106[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v108 = (v103 & v107)
    v109 = tl.maximum(v57, v52, propagate_nan=tl.PropagateNan.ALL)
    v110 = (-v57)
    v111 = tl.maximum(v57, v110, propagate_nan=tl.PropagateNan.ALL)
    v112 = (-v111)
    v113 = (v112 * v41)
    v114 = libdevice.exp2(tl.cast(v113, tl.float32))
    v115 = (v42 + v114)
    v116 = tl.fdiv(tl.cast(v114, tl.float32), tl.cast(v115, tl.float32), ieee_rounding=True)
    v117 = (v51 * v116)
    v118 = (v116 * v116)
    v119 = (v118 * v43)
    v120 = (v44 + v119)
    v121 = (v118 * v120)
    v122 = (v45 + v121)
    v123 = (v118 * v122)
    v124 = (v46 + v123)
    v125 = (v118 * v124)
    v126 = (v47 + v125)
    v127 = (v118 * v126)
    v128 = (v48 + v127)
    v129 = (v118 * v128)
    v130 = (v49 + v129)
    v131 = (v118 * v130)
    v132 = (v50 + v131)
    v133 = (v117 * v132)
    v134 = (v109 + v133)
    v135 = tl.where(v91, v57, v134)
    v136 = tl.broadcast_to(v95[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v137 = (v108 & v136)
    v138 = tl.broadcast_to(v99[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v139 = (v137 & v138)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v93, tl.int64) * S3_0 + tl.cast(v97, tl.int64) * S3_1), shape=((D1 - tl.cast(v93, tl.int64)), (D3 - tl.cast(v97, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v135, tl.float32), boundary_check=(0, 1))

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
