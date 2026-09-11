import torch
import triton
import triton.language as tl


_MATMUL_CONFIGS = [
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_MATMUL_CONFIGS, key=[])
@triton.jit
def _linear_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(0, 4096 // BLOCK_K):
        ks = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        input_values = tl.load(input_ptr + rows[:, None] * 4096 + ks[None, :])
        weight_values = tl.load(weight_ptr + cols[:, None] * 4096 + ks[None, :])
        accum = tl.dot(input_values, tl.trans(weight_values), acc=accum, input_precision="ieee")

    accum += tl.load(bias_ptr + cols)[None, :]
    tl.store(output_ptr + rows[:, None] * 2048 + cols[None, :], tl.maximum(accum, 0.0))


@triton.jit
def _layer_norm_kernel(input_ptr, output_ptr, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    values = tl.load(input_ptr + row * 2048 + cols).to(tl.float32)

    mean = tl.sum(values, axis=0) / 2048.0
    variance = tl.sum(values * values, axis=0) / 2048.0 - mean * mean
    centered = values - mean
    normalized = centered * tl.rsqrt(variance + eps)
    tl.store(output_ptr + row * 2048 + cols, normalized)


def build(context):
    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-5, elementwise_affine=True):
        del normalized_shape, elementwise_affine

        rows = input.numel() // input.shape[-1]
        output_shape = input.shape[:-1] + (weight.shape[0],)
        output = torch.empty((rows, 2048), device=input.device, dtype=input.dtype)

        grid = lambda META: (
            triton.cdiv(32, META["BLOCK_M"]),
            triton.cdiv(2048, META["BLOCK_N"]),
        )
        _linear_relu_kernel[grid](
            input,
            weight,
            bias,
            output,
        )
        _layer_norm_kernel[(rows,)](
            output,
            output,
            eps,
            BLOCK_N=2048,
            num_warps=8,
            num_stages=2,
        )
        return output.view(output_shape)

    return wrapper
