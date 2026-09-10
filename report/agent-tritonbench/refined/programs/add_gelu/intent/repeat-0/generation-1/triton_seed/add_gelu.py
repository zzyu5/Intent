import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "other", "output", ), (False, False, True, ))

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
    key=["D1", "S0_0", "S1_0", "S3_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, other, output, alpha, D1: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, S3_0: tl.constexpr, FRAGMENT_D1: tl.constexpr):
    v0 = (FRAGMENT_D1 - 1)
    v1 = (D1 + v0)
    v2 = ((v1 // FRAGMENT_D1) - (((v1 % FRAGMENT_D1) != 0) & (((v1 % FRAGMENT_D1) < 0) != (FRAGMENT_D1 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * FRAGMENT_D1)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v9 = tl.full((FRAGMENT_D1,), D1, tl.int64)
    v10 = (v8 < v9)
    v11 = D1
    v12 = tl.full((FRAGMENT_D1,), v11, tl.int64)
    v13 = (v8 < v12)
    v14 = tl.full((FRAGMENT_D1,), 0.0, tl.float16)
    v15 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v16 = (v15 < v9)
    v17 = D1
    v18 = tl.full((FRAGMENT_D1,), v17, tl.int64)
    v19 = (v15 < v18)
    v20 = tl.full((FRAGMENT_D1,), 0.0, tl.float16)
    v21 = tl.full((FRAGMENT_D1,), alpha, tl.float32)
    v22 = tl.full((FRAGMENT_D1,), 0.044714998453855515, tl.float32)
    v23 = tl.full((FRAGMENT_D1,), 0.79788458347320557, tl.float32)
    v24 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v25 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v26 = (v7 + tl.arange(0, FRAGMENT_D1) * 1)
    v27 = (v26 < v9)
    v28 = D1
    v29 = tl.full((FRAGMENT_D1,), v28, tl.int64)
    v30 = (v26 < v29)
    v31 = (v19 & v16)
    v32 = (v13 & v10)
    v33 = (v30 & v27)
    v34 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v35 = tl.cast(v34, tl.float32)
    v36 = tl.load(tl.make_block_ptr(base=(other + tl.cast(v7, tl.int64) * S1_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v37 = tl.cast(v36, tl.float32)
    v38 = (v21 * v37)
    v39 = (v35 + v38)
    v40 = (v24 * v39)
    v41 = (v39 * v39)
    v42 = (v41 * v39)
    v43 = (v22 * v42)
    v44 = (v39 + v43)
    v45 = (v23 * v44)
    v46 = libdevice.tanh(tl.cast(v45, tl.float32))
    v47 = (v25 + v46)
    v48 = (v40 * v47)
    v49 = tl.cast(v48, tl.float16)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v7, tl.int64) * S3_0), shape=((D1 - tl.cast(v7, tl.int64)),), strides=(S3_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v49, tl.float16), boundary_check=(0,))

def launch(input, other, output, alpha):
    D1 = input.shape[0]
    S0_0 = input.stride(0)
    S1_0 = other.stride(0)
    S3_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, other, output, alpha, D1, S0_0, S1_0, S3_0)

def run(input, other, alpha):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float16)
    launch(input, other, output, alpha)
    return output
