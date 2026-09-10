import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "vec", "output", ), (False, False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 32, "FRAGMENT_D1": 64}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_K_1_2": 64, "FRAGMENT_D1": 32}, num_warps=4, num_stages=4, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S4_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, vec, output, other, alpha, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S4_0: tl.constexpr, BLOCK_K_1_2: tl.constexpr, FRAGMENT_D1: tl.constexpr):
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
    v14 = D2
    v15 = D2
    v16 = tl.full((FRAGMENT_D1, 1), 0.0, tl.float32)
    v17 = v16
    for iv18 in range(0, D2, BLOCK_K_1_2):
        v19 = (iv18 + tl.arange(0, BLOCK_K_1_2) * 1)
        v20 = (iv18 - 0)
        v21 = (0 + v20)
        v22 = (v21 + tl.arange(0, BLOCK_K_1_2) * 1)
        v23 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v24 = (v19 < v23)
        v25 = tl.full((BLOCK_K_1_2,), D2, tl.int64)
        v26 = (v22 < v25)
        v27 = tl.full((BLOCK_K_1_2,), v14, tl.int64)
        v28 = (v19 < v27)
        v29 = tl.broadcast_to(v28[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v30 = tl.broadcast_to(v13[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v31 = (v30 & v29)
        v32 = tl.broadcast_to(v10[:, None], (FRAGMENT_D1, BLOCK_K_1_2))
        v33 = (v31 & v32)
        v34 = tl.broadcast_to(v24[None, :], (FRAGMENT_D1, BLOCK_K_1_2))
        v35 = (v33 & v34)
        v36 = tl.full((FRAGMENT_D1, BLOCK_K_1_2), 0.0, tl.float32)
        v37 = tl.full((BLOCK_K_1_2,), v15, tl.int64)
        v38 = (v22 < v37)
        v39 = (v38 & v26)
        v40 = tl.full((BLOCK_K_1_2,), 0.0, tl.float32)
        v41 = tl.load(tl.make_block_ptr(base=(vec + tl.cast(v21, tl.int64) * S1_0), shape=((D2 - tl.cast(v21, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(BLOCK_K_1_2,), order=(0,)), boundary_check=(0,), padding_option="zero")
        v42 = tl.reshape(v41, (BLOCK_K_1_2, 1), can_reorder=False)
        v43 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0 + tl.cast(iv18, tl.int64) * S0_1), shape=((D1 - tl.cast(v7, tl.int64)), (D2 - tl.cast(iv18, tl.int64))), strides=(S0_0, S0_1), offsets=(0, 0), block_shape=(FRAGMENT_D1, BLOCK_K_1_2), order=(1, 0)), boundary_check=(0, 1), padding_option="zero")
        v44 = tl.dot(v43, v42, v17, input_precision="ieee")
        v17 = v44
    v45 = tl.reshape(v17, (FRAGMENT_D1,), can_reorder=False)
    v46 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v47 = (v46 * v45)
    v48 = libdevice.tanh(tl.cast(v47, tl.float32))
    v49 = tl.full((FRAGMENT_D1,), 1.0, tl.float32)
    v50 = (v48 + v49)
    v51 = tl.full((FRAGMENT_D1,), 0.5, tl.float32)
    v52 = (v51 * v50)
    v53 = (alpha * other)
    v54 = tl.full((FRAGMENT_D1,), v53, tl.float32)
    v55 = (v52 - v54)
    v56 = (v5 * 1)
    v57 = (0 + v56)
    v58 = (v57 + tl.arange(0, FRAGMENT_D1) * 1)
    v59 = (v58 < v9)
    v60 = D1
    v61 = tl.full((FRAGMENT_D1,), v60, tl.int64)
    v62 = (v58 < v61)
    v63 = (v62 & v59)
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v57, tl.int64) * S4_0), shape=((D1 - tl.cast(v57, tl.int64)),), strides=(S4_0,), offsets=(0,), block_shape=(FRAGMENT_D1,), order=(0,)), tl.cast(v55, tl.float32), boundary_check=(0,))

def launch(input, vec, output, other, alpha):
    D1 = input.shape[0]
    D2 = input.shape[1]
    S0_0 = input.stride(0)
    S0_1 = input.stride(1)
    S1_0 = vec.stride(0)
    S4_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(D1, META["FRAGMENT_D1"]),)
    return _intent_kernel[grid](input, vec, output, other, alpha, D1, D2, S0_0, S0_1, S1_0, S4_0)

def run(input, vec, other, alpha):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, vec, output, other, alpha)
    return output
