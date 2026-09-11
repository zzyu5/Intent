import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _add_gelu_tensor_full(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(input_ptr + offsets).to(tl.float32)
    other_value = tl.load(other_ptr + offsets).to(tl.float32)

    if ALPHA_IS_ONE:
        x = x + other_value
    else:
        x = x + alpha * other_value

    result = x * (0.5 + 0.5 * tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, result)


@triton.jit
def _add_gelu_tensor_masked(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    other_value = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if ALPHA_IS_ONE:
        x = x + other_value
    else:
        x = x + alpha * other_value

    if APPROXIMATE_TANH:
        x3 = x * x * x
        z = 0.7978845608028654 * (x + 0.044715 * x3)
        ez = tl.exp(-2.0 * tl.abs(z))
        tanh_z = (1.0 - ez) / (1.0 + ez)
        tanh_z = tl.where(z < 0.0, -tanh_z, tanh_z)
        result = 0.5 * x * (1.0 + tanh_z)
    else:
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _add_gelu_scalar(
    input_ptr,
    other,
    output_ptr,
    n_elements,
    alpha,
    ALPHA_IS_ONE: tl.constexpr,
    APPROXIMATE_TANH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if ALPHA_IS_ONE:
        x = x + other
    else:
        x = x + alpha * other

    if APPROXIMATE_TANH:
        x3 = x * x * x
        z = 0.7978845608028654 * (x + 0.044715 * x3)
        ez = tl.exp(-2.0 * tl.abs(z))
        tanh_z = (1.0 - ez) / (1.0 + ez)
        tanh_z = tl.where(z < 0.0, -tanh_z, tanh_z)
        result = 0.5 * x * (1.0 + tanh_z)
    else:
        result = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other, alpha=1, approximate="none", out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        alpha_value = float(alpha)
        alpha_is_one = alpha_value == 1.0
        approximate_tanh = approximate == "tanh"

        if isinstance(other, torch.Tensor):
            if (
                not approximate_tanh
                and input.is_contiguous()
                and other.is_contiguous()
                and output.is_contiguous()
                and n_elements % 2048 == 0
            ):
                full_grid = lambda meta: (n_elements // meta["BLOCK_SIZE"],)
                _add_gelu_tensor_full[full_grid](
                    input,
                    other,
                    output,
                    n_elements,
                    alpha_value,
                    ALPHA_IS_ONE=alpha_is_one,
                )
            else:
                _add_gelu_tensor_masked[(triton.cdiv(n_elements, 1024),)](
                    input,
                    other,
                    output,
                    n_elements,
                    alpha_value,
                    ALPHA_IS_ONE=alpha_is_one,
                    APPROXIMATE_TANH=approximate_tanh,
                    BLOCK_SIZE=1024,
                    num_warps=4,
                )
        else:
            _add_gelu_scalar[(triton.cdiv(n_elements, 1024),)](
                input,
                other,
                output,
                n_elements,
                alpha_value,
                ALPHA_IS_ONE=alpha_is_one,
                APPROXIMATE_TANH=approximate_tanh,
                BLOCK_SIZE=1024,
                num_warps=4,
            )
        return output

    return wrapper
