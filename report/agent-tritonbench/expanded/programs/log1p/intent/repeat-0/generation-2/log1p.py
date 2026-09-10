import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = tl.cast(23, tl.uint32)
    v1 = tl.cast(255, tl.uint32)
    v2 = tl.cast(8388607, tl.uint32)
    v3 = tl.cast(1065353216, tl.uint32)
    v4 = (FRAGMENT_D1 - 1)
    v5 = (D1 + v4)
    v6 = ((v5 // FRAGMENT_D1) - (((v5 % FRAGMENT_D1) != 0) & (((v5 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v7 = tl.program_id(0)
    v8 = (v7 % v6)
    v9 = (v8 * FRAGMENT_D1)
    v10 = (v9 * 1)
    v11 = (0 + v10)
    v12 = (v11 + tl.arange(0, FRAGMENT_D1) * 1)
    v13 = D1
    v14 = (v12 < v13)
    v15 = D1
    v16 = tl.full((FRAGMENT_D1,), v15, tl.int64)
    v17 = (v12 < v16)
    v18 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v19 = 2.0
    v20 = 0.052631579339504242
    v21 = 0.058823529630899429
    v22 = 0.066666670143604279
    v23 = 0.076923079788684845
    v24 = 0.090909093618392944
    v25 = 0.1111111119389534
    v26 = 0.1428571492433548
    v27 = 0.20000000298023224
    v28 = 0.3333333432674408
    v29 = 2.0
    v30 = 1.0
    v31 = 1.0
    v32 = v0
    v33 = v1
    v34 = v2
    v35 = v3
    v36 = 1.0
    v37 = 1.0
    v38 = 127.0
    v39 = 0.69314718246459961
    v40 = 0.052631579339504242
    v41 = 0.058823529630899429
    v42 = 0.066666670143604279
    v43 = 0.076923079788684845
    v44 = 0.090909093618392944
    v45 = 0.1111111119389534
    v46 = 0.1428571492433548
    v47 = 0.20000000298023224
    v48 = 0.3333333432674408
    v49 = 2.0
    v50 = 1.0
    v51 = -0.5
    v52 = (v17 & v14)
    v53 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v11, tl.int64) * S0_0), shape=((D1 - tl.cast(v11, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v54 = (v53 > v51)
    v55 = 0.5
    v56 = (v53 < v55)
    v57 = (v54 & v56)
    v58 = (v11 + tl.arange(0, FRAGMENT_D1) * 1)
    v59 = (v58 < v13)
    v60 = D1
    v61 = tl.full((FRAGMENT_D1,), v60, tl.int64)
    v62 = (v58 < v61)
    v63 = (v62 & v59)
    v64 = (v53 + v19)
    v65 = tl.inline_asm_elementwise("div.approx.f32 $0, $1, $2;", constraints="=f,f,f", args=[v53, v64], dtype=tl.float32, is_pure=True, pack=1)
    v66 = (v29 * v65)
    v67 = (v65 * v65)
    v68 = (v67 * v20)
    v69 = (v21 + v68)
    v70 = (v67 * v69)
    v71 = (v22 + v70)
    v72 = (v67 * v71)
    v73 = (v23 + v72)
    v74 = (v67 * v73)
    v75 = (v24 + v74)
    v76 = (v67 * v75)
    v77 = (v25 + v76)
    v78 = (v67 * v77)
    v79 = (v26 + v78)
    v80 = (v67 * v79)
    v81 = (v27 + v80)
    v82 = (v67 * v81)
    v83 = (v28 + v82)
    v84 = (v67 * v83)
    v85 = (v30 + v84)
    v86 = (v66 * v85)
    v87 = (v53 + v31)
    v88 = tl.cast(v87, tl.uint32, bitcast=True)
    v89 = (v88 >> v32)
    v90 = (v89 & v33)
    v91 = tl.cast(v90, tl.float32)
    v92 = (v91 - v38)
    v93 = (v92 * v39)
    v94 = (v88 & v34)
    v95 = (v94 | v35)
    v96 = tl.cast(v95, tl.float32, bitcast=True)
    v97 = (v96 - v36)
    v98 = (v96 + v37)
    v99 = tl.inline_asm_elementwise("div.approx.f32 $0, $1, $2;", constraints="=f,f,f", args=[v97, v98], dtype=tl.float32, is_pure=True, pack=1)
    v100 = (v49 * v99)
    v101 = (v99 * v99)
    v102 = (v101 * v40)
    v103 = (v41 + v102)
    v104 = (v101 * v103)
    v105 = (v42 + v104)
    v106 = (v101 * v105)
    v107 = (v43 + v106)
    v108 = (v101 * v107)
    v109 = (v44 + v108)
    v110 = (v101 * v109)
    v111 = (v45 + v110)
    v112 = (v101 * v111)
    v113 = (v46 + v112)
    v114 = (v101 * v113)
    v115 = (v47 + v114)
    v116 = (v101 * v115)
    v117 = (v48 + v116)
    v118 = (v101 * v117)
    v119 = (v50 + v118)
    v120 = (v100 * v119)
    v121 = (v93 + v120)
    v122 = tl.where(v57, v86, v121)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v11, tl.int64) * S1_0), shape=((D1 - tl.cast(v11, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v122, tl.float32), boundary_check=(0,))

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, S0_0, S1_0)

def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
