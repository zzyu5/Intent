import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 1024, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 32}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 512, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 256, "BLOCK_K": 128}, num_warps=4, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["K", "N"])
@triton.jit
def softplus_linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    row = pid // num_pid_n
    pid_n = pid % num_pid_n
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((1, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K, num_stages=2):
        current_k = k_start + k_offsets
        input_values = tl.load(input_ptr + row * K + current_k, mask=current_k < K, other=0.0)
        weight_values = tl.load(
            weight_ptr + n_offsets[:, None] * K + current_k[None, :],
            mask=n_mask[:, None] & (current_k < K)[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            input_values[None, :],
            tl.trans(weight_values),
            acc=accumulator,
            input_precision="ieee",
        )

    if HAS_BIAS:
        bias_values = tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0)
        accumulator += bias_values[None, :]

    linear_value = tl.reshape(accumulator, (BLOCK_N,))
    scaled_value = linear_value * BETA
    softplus_value = tl.log(1.0 + tl.exp(scaled_value)) / BETA
    result = tl.where(scaled_value > THRESHOLD, linear_value, softplus_value)

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
            K=k_size,
            N=n_size,
            HAS_BIAS=bias is not None,
            BETA=beta,
            THRESHOLD=threshold,
        )
        return output

    return wrapper
