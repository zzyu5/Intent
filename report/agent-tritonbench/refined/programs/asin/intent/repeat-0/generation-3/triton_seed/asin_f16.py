import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 512}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 512}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 128}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 32}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4096}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16384}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (D1 - 0)
    v1 = (v0 + 0)
    v2 = ((v1 // 1) - (((v1 % 1) != 0) & (((v1 % 1) < 0) != (1 < 0))))
    v3 = (FRAGMENT_D1 - 1)
    v4 = (v2 + v3)
    v5 = ((v4 // FRAGMENT_D1) - (((v4 % FRAGMENT_D1) != 0) & (((v4 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v6 = tl.program_id(0)
    v7 = (v6 % v5)
    v8 = (v7 * FRAGMENT_D1)
    v9 = (v8 * 1)
    v10 = (0 + v9)
    v11 = (v10 + 1)
    v12 = (v2 - v8)
    v13 = (v12 * 1)
    v14 = (v10 + v13)
    v15 = (v10 + tl.arange(0, FRAGMENT_D1) * 1)
    v16 = tl.full((FRAGMENT_D1,), v14, tl.int64)
    v17 = (v15 < v16)
    v18 = D1
    v19 = tl.full((FRAGMENT_D1,), v18, tl.int64)
    v20 = (v15 < v19)
    v21 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v22 = (v15 >= v21)
    v23 = (v22 & v20)
    v24 = tl.cast(0, tl.float32)
    v25 = tl.fdiv(tl.cast(v24, tl.float32), tl.cast(v24, tl.float32), ieee_rounding=True)
    v26 = (v23 & v17)
    v27 = tl.full((FRAGMENT_D1,), 0.0, tl.float16)
    v28 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v29 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v10, tl.int64) * S0_0), shape=((D1 - tl.cast(v10, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v30 = tl.cast(v29, tl.float32)
    v31 = (v30 < v28)
    v32 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v33 = (-v30)
    v34 = tl.where(v31, v33, v30)
    v35 = (v34 <= v32)
    v36 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v37 = (v34 < v36)
    v38 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v39 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v40 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v41 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v42 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v43 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v44 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v45 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v46 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v47 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v48 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v49 = tl.full((FRAGMENT_D1,), -0.018729299306869507, tl.float32)
    v50 = tl.full((FRAGMENT_D1,), 0.074261002242565155, tl.float32)
    v51 = tl.full((FRAGMENT_D1,), 0.21211439371109009, tl.float32)
    v52 = tl.full((FRAGMENT_D1,), 1.5707287788391113, tl.float32)
    v53 = tl.full((FRAGMENT_D1,), 1.5707963705062866, tl.float32)
    v54 = tl.full((FRAGMENT_D1,), 1.5707963705062866, tl.float32)
    v55 = tl.full((FRAGMENT_D1,), v25, tl.float32)
    v56 = D1
    v57 = tl.full((FRAGMENT_D1,), v56, tl.int64)
    v58 = (v15 < v57)
    v59 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v60 = (v15 >= v59)
    v61 = (v60 & v58)
    v62 = (v61 & v17)
    v63 = (v49 * v34)
    v64 = (v63 + v50)
    v65 = (v64 * v34)
    v66 = (v65 - v51)
    v67 = (v66 * v34)
    v68 = (v67 + v52)
    v69 = (v38 - v34)
    v70 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v39, tl.float32), ieee_rounding=True)
    v71 = (v40 + v70)
    v72 = (v41 * v71)
    v73 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v72, tl.float32), ieee_rounding=True)
    v74 = (v72 + v73)
    v75 = (v42 * v74)
    v76 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v75, tl.float32), ieee_rounding=True)
    v77 = (v75 + v76)
    v78 = (v43 * v77)
    v79 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v78, tl.float32), ieee_rounding=True)
    v80 = (v78 + v79)
    v81 = (v44 * v80)
    v82 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v81, tl.float32), ieee_rounding=True)
    v83 = (v81 + v82)
    v84 = (v45 * v83)
    v85 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v84, tl.float32), ieee_rounding=True)
    v86 = (v84 + v85)
    v87 = (v46 * v86)
    v88 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v87, tl.float32), ieee_rounding=True)
    v89 = (v87 + v88)
    v90 = (v47 * v89)
    v91 = tl.fdiv(tl.cast(v69, tl.float32), tl.cast(v90, tl.float32), ieee_rounding=True)
    v92 = (v90 + v91)
    v93 = (v48 * v92)
    v94 = (v93 * v68)
    v95 = (v53 - v94)
    v96 = tl.where(v37, v95, v54)
    v97 = tl.where(v35, v96, v55)
    v98 = (-v97)
    v99 = tl.where(v31, v98, v97)
    v100 = tl.cast(v99, tl.float16)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v10, tl.int64) * S1_0), shape=((D1 - tl.cast(v10, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v100, tl.float16), boundary_check=(0,))

def launch(input, output):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, output, D1, S0_0, S1_0)

def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float16)
    launch(input, output)
    return output
