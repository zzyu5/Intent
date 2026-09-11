import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({"BLOCK_N": 1, "BLOCK_K": 32}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_N": 1, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 1, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 2, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 4, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 8, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 16, "BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 64}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["K", "N"])
@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    K,
    N,
    HAS_BIAS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    row = pid // num_pid_n
    pid_n = pid - row * num_pid_n

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)
    input_row = input_ptr + row * K

    for k_start in tl.range(0, K, BLOCK_K, num_stages=2):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K
        input_values = tl.load(input_row + k_offsets, mask=k_mask, other=0.0)
        weight_values = tl.load(
            weight_ptr + n_offsets[:, None] * K + k_offsets[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(weight_values * input_values[None, :], axis=1)

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)

    scaled_value = accumulator * BETA
    softplus_value = tl.log(1.0 + tl.exp(scaled_value)) / BETA
    result = tl.where(scaled_value > THRESHOLD, accumulator, softplus_value)
    tl.store(output_ptr + row * N + n_offsets, result, mask=n_mask)


def build(context):
    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        m_size = input.shape[0]
        n_size = weight.shape[0]
        k_size = input.shape[1]
        output = torch.empty((m_size, n_size), device=input.device, dtype=input.dtype)

        grid = lambda meta: (m_size * triton.cdiv(n_size, meta["BLOCK_N"]),)
        softplus_linear_kernel[grid](
            input,
            weight,
            bias if bias is not None else input,
            output,
            k_size,
            n_size,
            HAS_BIAS=bias is not None,
            BETA=beta,
            THRESHOLD=threshold,
        )
        return output

    return wrapper
