import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "weight1", "weight2", "bias", "output", ), (False, False, False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 64, "FRAGMENT_D1": 32, "FRAGMENT_D3": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_10_10": 32, "FRAGMENT_D1": 64, "FRAGMENT_D3": 32}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S3_0", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, weight1, weight2, bias, output, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr, S2_0: tl.constexpr, S3_0: tl.constexpr, S4_0: tl.constexpr, S4_1: tl.constexpr, BLOCK_K_10_10: tl.constexpr, FRAGMENT_D3: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v32 = (v24 + tl.arange(0, FRAGMENT_D3) * 1)
    v33 = (v32 < v26)
    v34 = D3
    v35 = tl.full((FRAGMENT_D3,), v34, tl.int64)
    v36 = (v32 < v35)
    v37 = tl.full((FRAGMENT_D3,), 0.0, tl.float32)
    v38 = (v24 + tl.arange(0, FRAGMENT_D3) * 1)
    v39 = (v38 < v26)
    v40 = D3
    v41 = tl.full((FRAGMENT_D3,), v40, tl.int64)
    v42 = (v38 < v41)
    v43 = tl.full((FRAGMENT_D3,), 0.0, tl.float32)
    v44 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 0.0, tl.float32)
    v45 = v44
    for iv46 in range(0, D2, BLOCK_K_10_10):
        v47 = (iv46 + tl.arange(0, BLOCK_K_10_10) * 1)
        v48 = (iv46 - 0)
        v49 = (0 + v48)
        v50 = (v49 + tl.arange(0, BLOCK_K_10_10) * 1)
        v51 = tl.full((BLOCK_K_10_10,), D2, tl.int64)
        v52 = (v47 < v51)
        v53 = (v50 < v51)
        v54 = tl.full((BLOCK_K_10_10,), v21, tl.int64)
        v55 = (v47 < v54)
        v56 = v55[None, :]
        v57 = v20[:, None]
        v58 = (v57 & v56)
        v59 = v17[:, None]
        v60 = (v58 & v59)
        v61 = v52[None, :]
        v62 = (v60 & v61)
        v63 = tl.full((FRAGMENT_D1, BLOCK_K_10_10), 0.0, tl.float32)
        v64 = tl.full((BLOCK_K_10_10,), v28, tl.int64)
        v65 = (v50 < v64)
        v66 = v65[:, None]
        v67 = v31[None, :]
        v68 = (v66 & v67)
        v69 = v27[None, :]
        v70 = (v68 & v69)
        v71 = v53[:, None]
        v72 = (v70 & v71)
        v73 = tl.full((BLOCK_K_10_10, FRAGMENT_D3), 0.0, tl.float32)
        v74 = tl.load(tl.make_block_ptr(base=(weight1 + tl.cast(v49, tl.int64) * S1_0 + tl.cast(v24, tl.int64) * S1_1), shape=((D2 - tl.cast(v49, tl.int64)), (D3 - tl.cast(v24, tl.int64))), strides=(S1_0, S1_1), offsets=(0, 0), block_shape=(BLOCK_K_10_10, FRAGMENT_D3), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v75 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v14, tl.int64) * S0_0 + tl.cast(iv46, tl.int64) * S0_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(iv46, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_10_10), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v76 = tl.dot(v75, v74, v45, input_precision="ieee")
        v45 = v76
    v77 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.4426950216293335, tl.float32)
    v78 = tl.full((FRAGMENT_D1, FRAGMENT_D3), 1.0, tl.float32)
    v79 = (-v45)
    v80 = (v79 * v77)
    v81 = libdevice.exp2(tl.cast(v80, tl.float32))
    v82 = (v78 + v81)
    v83 = tl.fdiv(tl.cast(v78, tl.float32), tl.cast(v82, tl.float32), ieee_rounding=True)
    v84 = tl.full((1,), 1.0, tl.float32)
    v85 = tl.reshape(v84, (1, 1), can_reorder=False)
    v86 = tl.broadcast_to(v85, (1, FRAGMENT_D3, ))
    v87 = tl.reshape(v84, (1, 1), can_reorder=False)
    v88 = tl.broadcast_to(v87, (1, FRAGMENT_D3, ))
    v89 = libdevice.tanh(tl.cast(v83, tl.float32))
    v90 = (0 + tl.arange(0, 1) * 1)
    v91 = (0 + tl.arange(0, 1) * 1)
    v92 = True
    v93 = 0.0
    v94 = tl.full((1,), 1, tl.int64)
    v95 = (v90 < v94)
    v96 = v95[:, None]
    v97 = (v92 & v96)
    v98 = tl.full((1,), 1, tl.int64)
    v99 = (v91 < v98)
    v100 = v99[None, :]
    v101 = (v97 & v100)
    v102 = (v14 + tl.arange(0, FRAGMENT_D1) * 1)
    v103 = (v102 < v16)
    v104 = (v24 + tl.arange(0, FRAGMENT_D3) * 1)
    v105 = (v104 < v26)
    v106 = D1
    v107 = tl.full((FRAGMENT_D1,), v106, tl.int64)
    v108 = (v102 < v107)
    v109 = v108[:, None]
    v110 = D3
    v111 = tl.full((FRAGMENT_D3,), v110, tl.int64)
    v112 = (v104 < v111)
    v113 = v112[None, :]
    v114 = (v109 & v113)
    v115 = (v42 & v39)
    v116 = (v36 & v33)
    v117 = v103[:, None]
    v118 = (v114 & v117)
    v119 = v105[None, :]
    v120 = (v118 & v119)
    v121 = tl.load(tl.make_block_ptr(base=(weight2 + tl.cast(v24, tl.int64) * S2_0), shape=((D3 - tl.cast(v24, tl.int64)),), strides=(S2_0,), offsets=(0,), block_shape=(FRAGMENT_D3,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v122 = v121[None, :]
    v123 = (v86 * v122)
    v124 = tl.broadcast_to(v123, (FRAGMENT_D1, FRAGMENT_D3, ))
    v125 = (v89 * v124)
    v126 = tl.load(tl.make_block_ptr(base=(bias + tl.cast(v24, tl.int64) * S3_0), shape=((D3 - tl.cast(v24, tl.int64)),), strides=(S3_0,), offsets=(0,), block_shape=(FRAGMENT_D3,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v127 = v126[None, :]
    v128 = (v88 * v127)
    v129 = tl.broadcast_to(v128, (FRAGMENT_D1, FRAGMENT_D3, ))
    v130 = (v125 + v129)
    v131 = v130
    v132 = tl.where(v101, v131, v93)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v14, tl.int64) * S4_0 + tl.cast(v24, tl.int64) * S4_1), shape=((D1 - tl.cast(v14, tl.int64)), (D3 - tl.cast(v24, tl.int64))), strides=(S4_0, S4_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D3), order=(1, 0)), tl.cast(v132, tl.float32), boundary_check=(0, 1))

def launch(input, weight1, weight2, bias, output):
    D1 = input.shape[0]
    D2 = input.shape[1]
    D3 = weight1.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = weight1.stride(0)
    S1_1 = weight1.stride(1)
    S2_0 = weight2.stride(0)
    S3_0 = bias.stride(0)
    S4_0 = output.stride(0)
    S4_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D3, META["FRAGMENT_D3"]))
    return _intent_kernel[grid](input, weight1, weight2, bias, output, D1, D2, D3, S0_0, S0_1, S1_0, S1_1, S2_0, S3_0, S4_0, S4_1)

def run(input, weight1, weight2, bias):
    output = torch.empty((input.shape[0], weight1.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, weight1, weight2, bias, output)
    return output
