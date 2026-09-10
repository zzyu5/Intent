import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    row_ptr = input_ptr + row * BLOCK_SIZE
    values = tl.load(row_ptr + offsets)
    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators, axis=0)
    result = numerators / denominator

    tl.store(output_ptr + row * BLOCK_SIZE + offsets, result)


def build(context):
    def wrapper(input, dim, dtype=None):
        if dim < 0:
            dim += input.ndim
        if dim != 1:
            raise ValueError("This implementation supports dim=1 only")

        output_dtype = input.dtype if dtype is None else dtype
        output = torch.empty(input.shape, device=input.device, dtype=output_dtype)
        n_rows = input.shape[0]
        _softmax_kernel[(n_rows,)](
            input,
            output,
            BLOCK_SIZE=1024,
        )
        return output

    return wrapper
