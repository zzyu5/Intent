import torch
import triton
import triton.language as tl


# The supplied profile always reduces 1024 input values for each output.
# Keep the search focused on the reduction shape used by this GEMV.
_CONFIGS = [
    triton.Config({"BLOCK_K": 64}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_K": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_K": 128}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_K": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_K": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_K": 512}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 512}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_K": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_K": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_K": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_K": 1024}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["M"])
@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M,
    HAS_BIAS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // 1024
    col = pid - row * 1024

    accumulator = tl.zeros((1,), dtype=tl.float32)
    input_row = input_ptr + row * 1024
    weight_row = weight_ptr + col * 1024

    # Every selected tile divides 1024, so all accesses are in bounds.
    for k_start in tl.range(0, 1024, BLOCK_K, num_stages=2):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        input_values = tl.load(input_row + k_offsets)
        weight_values = tl.load(weight_row + k_offsets)
        accumulator += tl.sum(input_values * weight_values, axis=0)

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + col)

    scaled_value = accumulator * BETA
    softplus_value = tl.log(1.0 + tl.exp(scaled_value)) / BETA
    result = tl.where(scaled_value > THRESHOLD, accumulator, softplus_value)
    tl.store(output_ptr + row * 1024 + col, result)


def build(context):
    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        m_size = input.shape[0]
        output = torch.empty((m_size, 1024), device=input.device, dtype=input.dtype)

        softplus_linear_kernel[(m_size * 1024,)](
            input,
            weight,
            bias if bias is not None else input,
            output,
            m_size,
            HAS_BIAS=bias is not None,
            BETA=beta,
            THRESHOLD=threshold,
        )
        return output

    return wrapper
