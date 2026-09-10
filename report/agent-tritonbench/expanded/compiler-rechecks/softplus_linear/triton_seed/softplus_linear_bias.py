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
def _intent_kernel(input, weight, bias, output, beta, threshold, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v41 = tl.full((FRAGMENT_D1, FRAGMENT_D3), beta, tl.float32)
    v42 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v43 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.4426950216293335, tl.float32)
    v44 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v45 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.043478261679410934, tl.float32)
    v46 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0476190485060215, tl.float32)
    v47 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.052631579339504242, tl.float32)
    v48 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.058823529630899429, tl.float32)
    v49 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.066666670143604279, tl.float32)
    v50 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.076923079788684845, tl.float32)
    v51 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.090909093618392944, tl.float32)
    v52 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1111111119389534, tl.float32)
    v53 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1428571492433548, tl.float32)
    v54 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.20000000298023224, tl.float32)
    v55 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.3333333432674408, tl.float32)
    v56 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v57 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.0, tl.float32)
    v58 = tl.full((FRAGMENT_D1, FRAGMENT_D3), beta, tl.float32)
    v59 = tl.full((FRAGMENT_D1, FRAGMENT_D3), threshold, tl.float32)
    v60 = (v39 & v36)
    v61 = tl.load(tl.make_block_ptr(base=(bias + tl.cast(v33, tl.int64) * S2_0), shape=((D3 - tl.cast(v33, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D3,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v62 = tl.broadcast_to(v61[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v63 = v62
    for iv64 in range(0, D2, BLOCK_K_1_2):
        v65 = (iv64 + tl.arange(0, BLOCK_K_1_2) * 1)
        v66 = (iv64 - 0)
        v67 = (0 + v66)
        v68 = (v67 + tl.arange(0, BLOCK_K_1_2) * 1)
        v69 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v70 = (v65 < v69)
        v71 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v72 = (v68 < v71)
        v73 = tl.full((BLOCK_K_1_2,), v21, tl.int64)
        v74 = (v65 < v73)
        v75 = tl.broadcast_to(v74[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v76 = tl.broadcast_to(v20[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v77 = (v76 & v75)
        v78 = tl.broadcast_to(v17[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v79 = (v77 & v78)
        v80 = tl.broadcast_to(v70[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v81 = (v79 & v80)
        v82 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float32)
        v83 = tl.full((BLOCK_K_1_2,), v31, tl.int64)
        v84 = (v68 < v83)
        v85 = tl.broadcast_to(v84[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v86 = tl.broadcast_to(v30[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v87 = (v86 & v85)
        v88 = tl.broadcast_to(v27[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v89 = (v87 & v88)
        v90 = tl.broadcast_to(v72[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v91 = (v89 & v90)
        v92 = tl.full((FRAGMENT_D3, BLOCK_K_1_2), 0.0, tl.float32)
        v93 = tl.load(tl.make_block_ptr(base=(weight + tl.cast(v24, tl.int64) * S1_0 + tl.cast(v67, tl.int64) * S1_1), shape=((D3 - tl.cast(v24, tl.int64)), (D2 - tl.cast(v67, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D3, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v94 = tl.permute(v93, (1, 0))
        v95 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv64, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv64, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v96 = tl.dot(v95, v94, v63, input_precision="ieee")
        v63 = v96
    v97 = (v63 * v41)
    v98 = (v97 > v59)
    v99 = (v12 * 1)
    v100 = (0 + v99)
    v101 = (v100 + tl.arange(0, FRAGMENT_D1) * 1)
    v102 = (v101 < v16)
    v103 = (v22 * 1)
    v104 = (0 + v103)
    v105 = (v104 + tl.arange(0, FRAGMENT_D3) * 1)
    v106 = (v105 < v26)
    v107 = D1
    v108 = tl.full((FRAGMENT_D1,), v107, tl.int64)
    v109 = (v101 < v108)
    v110 = tl.broadcast_to(v109[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v111 = D3
    v112 = tl.full((FRAGMENT_D3,), v111, tl.int64)
    v113 = (v105 < v112)
    v114 = tl.broadcast_to(v113[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v115 = (v110 & v114)
    v116 = (-v97)
    v117 = tl.maximum(v97, v116, propagate_nan=tl.PropagateNan.ALL)
    v118 = (-v117)
    v119 = (v118 * v43)
    v120 = libdevice.exp2(tl.cast(v119, tl.float32))
    v121 = (v44 + v120)
    v122 = tl.fdiv(tl.cast(v120, tl.float32), tl.cast(v121, tl.float32), ieee_rounding=True)
    v123 = (v122 * v122)
    v124 = (v123 * v45)
    v125 = (v46 + v124)
    v126 = (v123 * v125)
    v127 = (v47 + v126)
    v128 = (v123 * v127)
    v129 = (v48 + v128)
    v130 = (v123 * v129)
    v131 = (v49 + v130)
    v132 = (v123 * v131)
    v133 = (v50 + v132)
    v134 = (v123 * v133)
    v135 = (v51 + v134)
    v136 = (v123 * v135)
    v137 = (v52 + v136)
    v138 = (v123 * v137)
    v139 = (v53 + v138)
    v140 = (v123 * v139)
    v141 = (v54 + v140)
    v142 = (v123 * v141)
    v143 = (v55 + v142)
    v144 = (v123 * v143)
    v145 = (v57 + v144)
    v146 = (v56 * v122)
    v147 = (v146 * v145)
    v148 = tl.maximum(v97, v42, propagate_nan=tl.PropagateNan.ALL)
    v149 = (v148 + v147)
    v150 = tl.fdiv(tl.cast(v149, tl.float32), tl.cast(v58, tl.float32), ieee_rounding=True)
    v151 = tl.where(v98, v63, v150)
    v152 = tl.broadcast_to(v102[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v153 = (v115 & v152)
    v154 = tl.broadcast_to(v106[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v155 = (v153 & v154)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v100, tl.int64) * S3_0 + tl.cast(v104, tl.int64) * S3_1), shape=((D1 - tl.cast(v100, tl.int64)), (D3 - tl.cast(v104, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v151, tl.float32), boundary_check=(0, 1))

def launch(input, weight, bias, output, beta, threshold):
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
    return _intent_kernel[grid](input, weight, bias, output, beta, threshold, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S3_0, S3_1)

def run(input, weight, bias, beta, threshold):
    output = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, weight, bias, output, beta, threshold)
    return output
