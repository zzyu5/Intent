import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight", "output", ), (False, False, True, ))

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
def _intent_kernel(input, weight, output, beta, threshold, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
        v53 = tl.full((BLOCK_K_1_2,), v31, tl.int64)
        v54 = (v38 < v53)
        v55 = tl.broadcast_to(v54[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v56 = tl.broadcast_to(v30[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v57 = (v56 & v55)
        v58 = tl.broadcast_to(v27[:, None], (FRAGMENT_D3, BLOCK_K_1_2))
        v59 = (v57 & v58)
        v60 = tl.broadcast_to(v42[None, :], (FRAGMENT_D3, BLOCK_K_1_2))
        v61 = (v59 & v60)
        v62 = tl.full((FRAGMENT_D3, BLOCK_K_1_2), 0.0, tl.float32)
        v63 = tl.load(tl.make_block_ptr(base=(weight + tl.cast(v24, tl.int64) * S1_0 + tl.cast(v37, tl.int64) * S1_1), shape=((D3 - tl.cast(v24, tl.int64)), (D2 - tl.cast(v37, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(FRAGMENT_D3, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v64 = tl.permute(v63, (1, 0))
        v65 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv34, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv34, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v66 = tl.dot(v65, v64, v33, input_precision="ieee")
        v33 = v66
    v67 = tl.full((FRAGMENT_D1, FRAGMENT_D3), beta, tl.float32)
    v68 = (v33 * v67)
    v69 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v70 = tl.maximum(v68, v69, propagate_nan=tl.PropagateNan.ALL)
    v71 = (-v68)
    v72 = tl.maximum(v68, v71, propagate_nan=tl.PropagateNan.ALL)
    v73 = (-v72)
    v74 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.4426950216293335, tl.float32)
    v75 = (v73 * v74)
    v76 = libdevice.exp2(tl.cast(v75, tl.float32))
    v77 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v78 = (v77 + v76)
    v79 = tl.fdiv(tl.cast(v76, tl.float32), tl.cast(v78, tl.float32), ieee_rounding=True)
    v80 = (v79 * v79)
    v81 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.043478261679410934, tl.float32)
    v82 = (v80 * v81)
    v83 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0476190485060215, tl.float32)
    v84 = (v83 + v82)
    v85 = (v80 * v84)
    v86 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.052631579339504242, tl.float32)
    v87 = (v86 + v85)
    v88 = (v80 * v87)
    v89 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.058823529630899429, tl.float32)
    v90 = (v89 + v88)
    v91 = (v80 * v90)
    v92 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.066666670143604279, tl.float32)
    v93 = (v92 + v91)
    v94 = (v80 * v93)
    v95 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.076923079788684845, tl.float32)
    v96 = (v95 + v94)
    v97 = (v80 * v96)
    v98 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.090909093618392944, tl.float32)
    v99 = (v98 + v97)
    v100 = (v80 * v99)
    v101 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1111111119389534, tl.float32)
    v102 = (v101 + v100)
    v103 = (v80 * v102)
    v104 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1428571492433548, tl.float32)
    v105 = (v104 + v103)
    v106 = (v80 * v105)
    v107 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.20000000298023224, tl.float32)
    v108 = (v107 + v106)
    v109 = (v80 * v108)
    v110 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.3333333432674408, tl.float32)
    v111 = (v110 + v109)
    v112 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v113 = (v112 * v79)
    v114 = (v80 * v111)
    v115 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.0, tl.float32)
    v116 = (v115 + v114)
    v117 = (v113 * v116)
    v118 = (v70 + v117)
    v119 = tl.full((FRAGMENT_D1, FRAGMENT_D3), beta, tl.float32)
    v120 = tl.fdiv(tl.cast(v118, tl.float32), tl.cast(v119, tl.float32), ieee_rounding=True)
    v121 = tl.full((FRAGMENT_D1, FRAGMENT_D3), threshold, tl.float32)
    v122 = (v68 > v121)
    v123 = tl.where(v122, v33, v120)
    v124 = (v12 * 1)
    v125 = (0 + v124)
    v126 = (v125 + tl.arange(0, FRAGMENT_D1) * 1)
    v127 = (v126 < v16)
    v128 = (v22 * 1)
    v129 = (0 + v128)
    v130 = (v129 + tl.arange(0, FRAGMENT_D3) * 1)
    v131 = (v130 < v26)
    v132 = D1
    v133 = tl.full((FRAGMENT_D1,), v132, tl.int64)
    v134 = (v126 < v133)
    v135 = tl.broadcast_to(v134[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v136 = D3
    v137 = tl.full((FRAGMENT_D3,), v136, tl.int64)
    v138 = (v130 < v137)
    v139 = tl.broadcast_to(v138[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v140 = (v135 & v139)
    v141 = tl.broadcast_to(v127[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v142 = (v140 & v141)
    v143 = tl.broadcast_to(v131[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v144 = (v142 & v143)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v125, tl.int64) * S2_0 + tl.cast(v129, tl.int64) * S2_1), shape=((D1 - tl.cast(v125, tl.int64)), (D3 - tl.cast(v129, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v123, tl.float32), boundary_check=(0, 1))

def launch(input, weight, output, beta, threshold):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = weight.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = weight.stride(0)
    S1_1 = weight.stride(1)
    S2_0 = output.stride(0)
    S2_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](input, weight, output, beta, threshold, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(input, weight, beta, threshold):
    output = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, weight, output, beta, threshold)
    return output
