import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from intent.runtime.triton import TuningHooks


_intent_tuning_hooks = TuningHooks(
    ("input", "weight1", "weight2", "bias", "output"),
    (False, False, False, False, True),
)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4, num_ctas=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3, num_ctas=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=2, num_ctas=1),
    ],
    key=["M", "K", "N", "S0_0", "S0_1", "S1_0", "S1_1", "S2_0", "S3_0", "S4_0", "S4_1"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _fused_kernel(
    input,
    weight1,
    weight2,
    bias,
    output,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    S0_0: tl.constexpr,
    S0_1: tl.constexpr,
    S1_0: tl.constexpr,
    S1_1: tl.constexpr,
    S2_0: tl.constexpr,
    S3_0: tl.constexpr,
    S4_0: tl.constexpr,
    S4_1: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        lhs = tl.load(input + rows[:, None] * S0_0 + ks[None, :] * S0_1)
        rhs = tl.load(weight1 + ks[:, None] * S1_0 + cols[None, :] * S1_1)
        acc = tl.dot(lhs, rhs, acc, input_precision="ieee")

    sigmoid = tl.fdiv(
        1.0,
        1.0 + libdevice.exp2(-acc * 1.4426950216293335),
        ieee_rounding=False,
    )
    activated = libdevice.tanh(sigmoid)
    scale = tl.load(weight2 + cols * S2_0)
    offset = tl.load(bias + cols * S3_0)
    result = activated * scale[None, :] + offset[None, :]
    tl.store(output + rows[:, None] * S4_0 + cols[None, :] * S4_1, result)


def launch(input, weight1, weight2, bias, output):
    M = input.shape[0]
    K = input.shape[1]
    N = weight1.shape[1]
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )
    return _fused_kernel[grid](
        input,
        weight1,
        weight2,
        bias,
        output,
        M,
        K,
        N,
        input.stride(0),
        input.stride(1),
        weight1.stride(0),
        weight1.stride(1),
        weight2.stride(0),
        bias.stride(0),
        output.stride(0),
        output.stride(1),
    )


def run(input, weight1, weight2, bias):
    output = torch.empty((input.shape[0], weight1.shape[1]), device=input.device, dtype=torch.float32)
    launch(input, weight1, weight2, bias, output)
    return output
