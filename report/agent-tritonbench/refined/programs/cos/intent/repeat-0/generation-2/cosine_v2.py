import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("input", "output", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 256}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 16}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({"FRAGMENT_S4_A0": 8192}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D2", "D3", "S0_0", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(input, output, D2: tl.constexpr, D3: tl.constexpr, S0_0: tl.constexpr, S1_0: tl.constexpr, FRAGMENT_S4_A0: tl.constexpr):
    v0 = (FRAGMENT_S4_A0 - 1)
    v1 = (1048576 + v0)
    v2 = ((v1 // FRAGMENT_S4_A0) - (((v1 % FRAGMENT_S4_A0) != 0) & (((v1 % FRAGMENT_S4_A0) < 0) != (FRAGMENT_S4_A0 < 0))))
    v3 = tl.program_id(0)
    v4 = (v3 % v2)
    v5 = (v4 * FRAGMENT_S4_A0)
    v6 = (v5 * 1)
    v7 = (0 + v6)
    v8 = (v7 + tl.arange(0, FRAGMENT_S4_A0) * 1)
    v9 = tl.full((FRAGMENT_S4_A0,), 1048576, tl.int64)
    v10 = (v8 < v9)
    v11 = tl.full((FRAGMENT_S4_A0,), 1048576, tl.int64)
    v12 = (v8 < v11)
    v13 = tl.full((FRAGMENT_S4_A0,), 0.0, tl.float32)
    v14 = (v7 + tl.arange(0, FRAGMENT_S4_A0) * 1)
    v15 = (v14 < v9)
    v16 = tl.full((FRAGMENT_S4_A0,), 1048576, tl.int64)
    v17 = (v14 < v16)
    v18 = (v12 & v10)
    v19 = (v17 & v15)
    v20 = tl.load(tl.make_block_ptr(base=(input + tl.cast(v7, tl.int64) * S0_0), shape=((1048576 - tl.cast(v7, tl.int64)),), strides=(S0_0,), offsets=(0,), block_shape=(FRAGMENT_S4_A0,), order=(0,)), boundary_check=(0,), padding_option="zero")
    v21 = libdevice.cos(tl.cast(v20, tl.float32))
    tl.store(tl.make_block_ptr(base=(output + tl.cast(v7, tl.int64) * S1_0), shape=((1048576 - tl.cast(v7, tl.int64)),), strides=(S1_0,), offsets=(0,), block_shape=(FRAGMENT_S4_A0,), order=(0,)), tl.cast(v21, tl.float32), boundary_check=(0,))

def launch(input, output):
    D2 = input.shape[0]
    D3 = output.shape[0]
    S0_0 = input.stride(0)
    S1_0 = output.stride(0)
    grid = lambda META: (triton.cdiv(1048576, META["FRAGMENT_S4_A0"]),)
    return _intent_kernel[grid](input, output, D2, D3, S0_0, S1_0)

def run(input):
    output = torch.empty((1048576,), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
