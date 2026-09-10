import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "index", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 512}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 512}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 512}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 1, "FRAGMENT_D2": 512}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 64, "FRAGMENT_D2": 64}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 4, "FRAGMENT_D2": 1024}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 16, "FRAGMENT_D2": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 16}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 8}, num_warps=2, num_stages=5, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_D1": 8, "FRAGMENT_D2": 2}, num_warps=2, num_stages=5, num_ctas=1),
    ],
    key=["D1", "D2", "D3", "S0_0", "S0_1", "S1_0", "S3_0", "S3_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, index, output, other, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S3_0: tl.constexpr, S3_1: tl.constexpr, FRAGMENT_D2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = (FRAGMENT_D2 - 1)
    v4 = (D2 + v3)
    v5 = ((v4 // FRAGMENT_D2) - (((v4 % FRAGMENT_D2) != 0) & (((v4 % FRAGMENT_D2) < 0) != (FRAGMENT_D2 < 0))))
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
    v21 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v22 = (v11 * FRAGMENT_D2)
    v23 = (v22 * 1)
    v24 = (0 + v23)
    v25 = (v24 + tl.arange(0, FRAGMENT_D2) * 1)
    v26 = D2
    v27 = (v25 < v26)
    v28 = D3
    v29 = tl.full((FRAGMENT_D1,), v28, tl.int64)
    v30 = tl.full((FRAGMENT_D1,), 0, tl.int64)
    v31 = (v20 & v17)
    v32 = tl.load(tl.make_block_ptr(base=(index + tl.cast(v14, tl.int64) * S1_0), shape=((D1 - tl.cast(v14, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v33 = tl.cast(v32, tl.int64)
    v34 = (v33 < v29)
    v35 = (v33 >= v30)
    v36 = (v35 & v34)
    v37 = v36[:, None]
    v38 = D2
    v39 = tl.full((FRAGMENT_D2,), v38, tl.int64)
    v40 = (v25 < v39)
    v41 = v40[None, :]
    v42 = (v37 & v41)
    v43 = tl.full((FRAGMENT_D1, FRAGMENT_D2), 0.0, tl.float32)
    v44 = other
    v45 = v17[:, None]
    v46 = (v42 & v45)
    v47 = v27[None, :]
    v48 = (v46 & v47)
    v49 = tl.load((input + (v33[:, None]) * S0_0 + (v25[None, :]) * S0_1), mask=v48, other=v43)
    v50 = (v49 == v44)
    v51 = tl.cast(v50, tl.int8)
    v52 = (v14 + tl.arange(0, FRAGMENT_D1) * 1)
    v53 = (v52 < v16)
    v54 = (v24 + tl.arange(0, FRAGMENT_D2) * 1)
    v55 = (v54 < v26)
    v56 = D1
    v57 = tl.full((FRAGMENT_D1,), v56, tl.int64)
    v58 = (v52 < v57)
    v59 = v58[:, None]
    v60 = D2
    v61 = tl.full((FRAGMENT_D2,), v60, tl.int64)
    v62 = (v54 < v61)
    v63 = v62[None, :]
    v64 = (v59 & v63)
    v65 = v53[:, None]
    v66 = (v64 & v65)
    v67 = v55[None, :]
    v68 = (v66 & v67)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v14, tl.int64) * S3_0 + tl.cast(v24, tl.int64) * S3_1), shape=((D1 - tl.cast(v14, tl.int64)), (D2 - tl.cast(v24, tl.int64))), strides=(S3_0, S3_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, FRAGMENT_D2), order=(1, 0)), tl.cast(v51, tl.int8), boundary_check=(0, 1))

def launch(input, index, output, other):
    D1 = index.shape[0]
    D2 = input.shape[1]
    D3 = input.shape[0]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = index.stride(0)
    S3_0 = output.stride(0)
    S3_1 = output.stride(1)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]), triton.cdiv(D2, META["FRAGMENT_D2"]))
    return _intent_kernel[grid](input, index, output, other, D1, D2, D3, S0_0, S0_1, S1_0, S3_0, S3_1)

def run(input, index, other):
    output = torch.empty((index.shape[0], input.shape[1]), device=input.device, dtype=torch.int8)
    launch(input, index, output, other)
    return output
