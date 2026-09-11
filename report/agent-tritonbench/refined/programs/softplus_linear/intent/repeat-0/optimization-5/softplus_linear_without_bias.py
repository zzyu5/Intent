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
def _intent_kernel(input, weight, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S2_1: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v67 = (-v33)
    v68 = tl.maximum(v33, v67, propagate_nan=tl.PropagateNan.ALL)
    v69 = (-v68)
    v70 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.4426950216293335, tl.float32)
    v71 = (v69 * v70)
    v72 = libdevice.exp2(tl.cast(v71, tl.float32))
    v73 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v74 = (v73 + v72)
    v75 = tl.fdiv(tl.cast(v72, tl.float32), tl.cast(v74, tl.float32), ieee_rounding=True)
    v76 = (v75 * v75)
    v77 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.066666670143604279, tl.float32)
    v78 = (v76 * v77)
    v79 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.076923079788684845, tl.float32)
    v80 = (v79 + v78)
    v81 = (v76 * v80)
    v82 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.090909093618392944, tl.float32)
    v83 = (v82 + v81)
    v84 = (v76 * v83)
    v85 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1111111119389534, tl.float32)
    v86 = (v85 + v84)
    v87 = (v76 * v86)
    v88 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.1428571492433548, tl.float32)
    v89 = (v88 + v87)
    v90 = (v76 * v89)
    v91 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.20000000298023224, tl.float32)
    v92 = (v91 + v90)
    v93 = (v76 * v92)
    v94 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.3333333432674408, tl.float32)
    v95 = (v94 + v93)
    v96 = (v76 * v95)
    v97 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.0, tl.float32)
    v98 = (v97 + v96)
    v99 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 2.0, tl.float32)
    v100 = (v99 * v75)
    v101 = (v100 * v98)
    v102 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v103 = tl.maximum(v33, v102, propagate_nan=tl.PropagateNan.ALL)
    v104 = (v103 + v101)
    v105 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 20.0, tl.float32)
    v106 = (v33 > v105)
    v107 = tl.where(v106, v33, v104)
    v108 = (v12 * 1)
    v109 = (0 + v108)
    v110 = (v109 + tl.arange(0, FRAGMENT_D1) * 1)
    v111 = (v110 < v16)
    v112 = (v22 * 1)
    v113 = (0 + v112)
    v114 = (v113 + tl.arange(0, FRAGMENT_D3) * 1)
    v115 = (v114 < v26)
    v116 = D1
    v117 = tl.full((FRAGMENT_D1,), v116, tl.int64)
    v118 = (v110 < v117)
    v119 = tl.broadcast_to(v118[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v120 = D3
    v121 = tl.full((FRAGMENT_D3,), v120, tl.int64)
    v122 = (v114 < v121)
    v123 = tl.broadcast_to(v122[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v124 = (v119 & v123)
    v125 = tl.broadcast_to(v111[:, None], (FRAGMENT_D1, FRAGMENT_D3))
    v126 = (v124 & v125)
    v127 = tl.broadcast_to(v115[None, :], (FRAGMENT_D1, FRAGMENT_D3))
    v128 = (v126 & v127)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v109, tl.int64) * S2_0 + tl.cast(v113, tl.int64) * S2_1), shape=((D1 - tl.cast(v109, tl.int64)), (D3 - tl.cast(v113, tl.int64))), strides=(S2_0, S2_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v107, tl.float32), boundary_check=(0, 1))

def launch(input, weight, output):
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
    return _intent_kernel[grid](input, weight, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S2_1)

def run(input, weight):
    output = torch.empty((input.shape[0], weight.shape[0]), device=input.device, dtype=torch.float32)
    launch(input, weight, output)
    return output
