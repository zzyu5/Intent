import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("A", "At", ), (False, True, ))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=2, num_stages=3, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=8, num_stages=2, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=1, num_ctas=1),
        triton.Config({}, num_warps=4, num_stages=3, num_ctas=1),
    ],
    key=["D1", "D2", "S0_0", "S0_1", "S1_0", "S1_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(A, At, D1: tl.constexpr, D2: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S1_0: tl.constexpr, S1_1: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 1)
    for iv2 in range(0, D1, 1):
        for iv3 in range(0, D2, 1):
            v4 = D1
            v5 = (iv2 < v4)
            v6 = (iv2 >= 0)
            v7 = (v6 & v5)
            v8 = D2
            v9 = (iv3 < v8)
            v10 = (iv3 >= 0)
            v11 = (v10 & v9)
            v12 = (v7 & v11)
            v13 = D2
            v14 = (iv3 < v13)
            v15 = (iv3 >= 0)
            v16 = (v15 & v14)
            v17 = D1
            v18 = (iv2 < v17)
            v19 = (iv2 >= 0)
            v20 = (v19 & v18)
            v21 = (v16 & v20)
            v22 = tl.load((A + (iv2) * S0_0 + (iv3) * S0_1), mask=v12, other=0.0)
            tl.store((At + (iv3) * S1_0 + (iv2) * S1_1), v22, mask=v21)

def launch(A, At):
    D1 = A.shape[0]
    D2 = A.shape[1]
    S0_0 = A.stride(0)
    S0_1 = A.stride(1)
    S1_0 = At.stride(0)
    S1_1 = At.stride(1)
    grid = lambda META: (1,)
    return _intent_kernel[grid](A, At, D1, D2, S0_0, S0_1, S1_0, S1_1)

def run(A):
    At = torch.empty((A.shape[1], A.shape[0]), device=A.device, dtype=torch.float32)
    launch(A, At)
    return At
