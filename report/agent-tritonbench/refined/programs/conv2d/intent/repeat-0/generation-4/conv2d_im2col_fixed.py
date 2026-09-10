import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor
from intent.runtime.triton import TuningHooks

_intent_tuning_hooks = TuningHooks(("x", "columns", ), (False, True, ))

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
    key=["D2", "D3", "D4", "D5", "D6", "S0_0", "S0_1", "S0_2", "S0_3", "S1_0"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _intent_kernel(x, columns, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr, D5: tl.constexpr, D6: tl.constexpr, S0_0: tl.constexpr, S0_1: tl.constexpr, S0_2: tl.constexpr, S0_3: tl.constexpr, S1_0: tl.constexpr):
    v0 = tl.program_id(0)
    v1 = (v0 % 903168)
    v2 = (v1 * 1)
    v3 = (0 + v2)
    v4 = ((v3 % 196) + (((v3 % 196) != 0) & (((v3 % 196) < 0) != (196 < 0))) * 196)
    v5 = ((v3 // 196) - (((v3 % 196) != 0) & (((v3 % 196) < 0) != (196 < 0))))
    v6 = ((v5 % 3) + (((v5 % 3) != 0) & (((v5 % 3) < 0) != (3 < 0))) * 3)
    v7 = ((v5 // 3) - (((v5 % 3) != 0) & (((v5 % 3) < 0) != (3 < 0))))
    v8 = ((v7 % 3) + (((v7 % 3) != 0) & (((v7 % 3) < 0) != (3 < 0))) * 3)
    v9 = ((v7 // 3) - (((v7 % 3) != 0) & (((v7 % 3) < 0) != (3 < 0))))
    v10 = ((v9 % 64) + (((v9 % 64) != 0) & (((v9 % 64) < 0) != (64 < 0))) * 64)
    v11 = ((v9 // 64) - (((v9 % 64) != 0) & (((v9 % 64) < 0) != (64 < 0))))
    v12 = ((v4 // 14) - (((v4 % 14) != 0) & (((v4 % 14) < 0) != (14 < 0))))
    v13 = ((v4 % 14) + (((v4 % 14) != 0) & (((v4 % 14) < 0) != (14 < 0))) * 14)
    v14 = (v12 + v8)
    v15 = (v13 + v6)
    v16 = (v11 < 8)
    v17 = (v11 >= 0)
    v18 = (v17 & v16)
    v19 = (v10 < 64)
    v20 = (v10 >= 0)
    v21 = (v20 & v19)
    v22 = (v18 & v21)
    v23 = (v3 < 903168)
    v24 = (v3 >= 0)
    v25 = (v24 & v23)
    v26 = tl.load((x + (v11) * S0_0 + (v10) * S0_1 + (v14) * S0_2 + (v15) * S0_3), mask=v22, other=0.0)
    tl.store((columns + (v3) * S1_0), v26, mask=v25)

def launch(x, columns):
    D2 = x.shape[0]
    D3 = x.shape[1]
    D4 = x.shape[2]
    D5 = x.shape[3]
    D6 = columns.shape[0]
    S0_0 = x.stride(0)
    S0_1 = x.stride(1)
    S0_2 = x.stride(2)
    S0_3 = x.stride(3)
    S1_0 = columns.stride(0)
    grid = lambda META: (903168,)
    return _intent_kernel[grid](x, columns, D2, D3, D4, D5, D6, S0_0, S0_1, S0_2, S0_3, S1_0)

def run(x):
    columns = torch.empty((903168,), device=x.device, dtype=torch.float32)
    launch(x, columns)
    return columns
