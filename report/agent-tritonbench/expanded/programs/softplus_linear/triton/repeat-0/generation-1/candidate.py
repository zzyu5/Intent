import torch
import triton
import triton.language as tl


@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    m_size,
    n_size,
    k_size,
    HAS_BIAS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_index = tl.program_id(0)
    row = output_index // n_size
    column = output_index % n_size

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < k_size
    input_offsets = row * k_size + k_offsets
    weight_offsets = column * k_size + k_offsets

    input_values = tl.load(input_ptr + input_offsets, mask=k_mask, other=0.0)
    weight_values = tl.load(weight_ptr + weight_offsets, mask=k_mask, other=0.0)
    linear_value = tl.sum(input_values * weight_values, axis=0)

    if HAS_BIAS:
        linear_value += tl.load(bias_ptr + column)

    scaled_value = linear_value * BETA
    softplus_value = tl.log(1.0 + tl.exp(scaled_value)) / BETA
    result = tl.where(scaled_value > THRESHOLD, linear_value, softplus_value)

    output_offsets = row * n_size + column
    tl.store(output_ptr + output_offsets, result)


def build(context):
    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        m_size = input.shape[0]
        n_size = weight.shape[0]
        k_size = input.shape[1]
        output = torch.empty((m_size, n_size), device=input.device, dtype=input.dtype)

        grid = (m_size * n_size,)
        softplus_linear_kernel[grid](
            input,
            weight,
            bias if bias is not None else input,
            output,
            m_size,
            n_size,
            k_size,
            HAS_BIAS=bias is not None,
            BETA=beta,
            THRESHOLD=threshold,
            BLOCK_K=1024,
        )
        return output

    return wrapper
