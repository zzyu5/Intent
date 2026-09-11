import torch
import triton
import triton.language as tl


@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    k_size,
    n_size,
    HAS_BIAS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (n_size + BLOCK_N - 1) // BLOCK_N
    row = pid // num_pid_n
    pid_n = pid % num_pid_n

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < n_size
    input_row = input_ptr + row * k_size
    linear_value = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in tl.range(0, k_size, BLOCK_K, num_stages=2):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k_size
        input_values = tl.load(input_row + k_offsets, mask=k_mask, other=0.0)
        weight_values = tl.load(
            weight_ptr + n_offsets[:, None] * k_size + k_offsets[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        linear_value += tl.sum(weight_values * input_values[None, :], axis=1)

    if HAS_BIAS:
        linear_value += tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)

    scaled_value = linear_value * BETA
    softplus_value = tl.log(1.0 + tl.exp(scaled_value)) / BETA
    result = tl.where(scaled_value > THRESHOLD, linear_value, softplus_value)

    tl.store(output_ptr + row * n_size + n_offsets, result, mask=n_mask)


def build(context):
    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        m_size = input.shape[0]
        n_size = weight.shape[0]
        k_size = input.shape[1]
        output = torch.empty((m_size, n_size), device=input.device, dtype=input.dtype)

        grid = (m_size * triton.cdiv(n_size, 256),)
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
            BLOCK_N=256,
            BLOCK_K=32,
            num_warps=8,
            num_stages=2,
        )
        return output

    return wrapper
