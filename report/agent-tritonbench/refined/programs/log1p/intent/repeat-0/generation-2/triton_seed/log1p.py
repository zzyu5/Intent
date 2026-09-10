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
    v4 = tl.cast(255, tl.uint32)
    v5 = (FRAGMENT_D1 - 1)
    v6 = (D1 + v5)
    v7 = ((v6 // FRAGMENT_D1) - (((v6 % FRAGMENT_D1) != 0) & (((v6 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v8 = tl.program_id(0)
    v9 = (v8 % v7)
    v10 = (v9 * FRAGMENT_D1)
    v11 = (v10 * 1)
    v12 = (0 + v11)
    v13 = (v12 + tl.arange(0, FRAGMENT_D1) * 1)
    v14 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v15 = (v13 < v14)
    v16 = D1
    v17 = tl.full((FRAGMENT_D1,), v16, tl.int64)
    v18 = (v13 < v17)
    v19 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v20 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v21 = tl.full((FRAGMENT_D1,), v0, tl.uint32)
    v22 = tl.full((FRAGMENT_D1,), v1, tl.uint32)
    v23 = tl.full((FRAGMENT_D1,), v2, tl.uint32)
    v24 = tl.full((FRAGMENT_D1,), v3, tl.uint32)
    v25 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v26 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v27 = tl.full((FRAGMENT_D1,), 2.0, tl.float32)
    v28 = tl.full((FRAGMENT_D1,), 11.0, tl.float32)
    v29 = tl.full((FRAGMENT_D1,), 0.1111111119389534, tl.float32)
    v30 = tl.full((FRAGMENT_D1,), 0.1428571492433548, tl.float32)
    v31 = tl.full((FRAGMENT_D1,), 0.20000000298023224, tl.float32)
    v32 = tl.full((FRAGMENT_D1,), 0.3333333432674408, tl.float32)
    v33 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v34 = tl.full((FRAGMENT_D1,), 127.0, tl.float32)
    v35 = tl.full((FRAGMENT_D1,), 0.69314718246459961, tl.float32)
    v36 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v37 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v38 = tl.full((FRAGMENT_D1,), -1.0, tl.float32)
    v39 = tl.full((FRAGMENT_D1,), v4, tl.uint32)
    v40 = (v18 & v15)
    v41 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v12, tl.int64) * S0_0), shape=((D1 - tl.cast(v12, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v42 = (v41 + v20)
    v43 = tl.cast(v42, tl.uint32, bitcast=True)
    v44 = (v43 >> v21)
    v45 = (v44 & v22)
    v46 = (v45 == v39)
    v47 = tl.full((FRAGMENT_D1,), 0.0, tl.float32)
    v48 = (v42 > v47)
    v49 = tl.full((FRAGMENT_D1,), -1.0, tl.float32)
    v50 = (v41 == v49)
    v51 = tl.full((FRAGMENT_D1,), -1.0, tl.float32)
    v52 = (v41 < v51)
    v53 = (v12 + tl.arange(0, FRAGMENT_D1) * 1)
    v54 = (v53 < v14)
    v55 = D1
    v56 = tl.full((FRAGMENT_D1,), v55, tl.int64)
    v57 = (v53 < v56)
    v58 = (v57 & v54)
    v59 = tl.fdiv(tl.cast(v38, tl.float32), tl.cast(v42, tl.float32), ieee_rounding=True)
    v60 = (v43 & v23)
    v61 = (v60 | v24)
    v62 = tl.cast(v61, tl.float32, bitcast=True)
    v63 = (v62 - v25)
    v64 = (v62 + v26)
    v65 = tl.fdiv(tl.cast(v63, tl.float32), tl.cast(v64, tl.float32), ieee_rounding=True)
    v66 = (v27 * v65)
    v67 = (v65 * v65)
    v68 = tl.fdiv(tl.cast(v67, tl.float32), tl.cast(v28, tl.float32), ieee_rounding=True)
    v69 = (v29 + v68)
    v70 = (v67 * v69)
    v71 = (v30 + v70)
    v72 = (v67 * v71)
    v73 = (v31 + v72)
    v74 = (v67 * v73)
    v75 = (v32 + v74)
    v76 = (v67 * v75)
    v77 = (v33 + v76)
    v78 = (v66 * v77)
    v79 = tl.cast(v45, tl.float32)
    v80 = (v79 - v34)
    v81 = (v80 * v35)
    v82 = (v81 + v78)
    v83 = (v42 - v36)
    v84 = (v41 - v83)
    v85 = tl.fdiv(tl.cast(v84, tl.float32), tl.cast(v42, tl.float32), ieee_rounding=True)
    v86 = (v82 + v85)
    v87 = (v41 - v41)
    v88 = tl.fdiv(tl.cast(v37, tl.float32), tl.cast(v87, tl.float32), ieee_rounding=True)
    v89 = tl.where(v48, v42, v88)
    v90 = tl.where(v46, v89, v86)
    v91 = tl.where(v50, v59, v90)
    v92 = tl.where(v52, v88, v91)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v12, tl.int64) * S1_0), shape=((D1 - tl.cast(v12, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v92, tl.float32), boundary_check=(0,))

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
