import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    input_ptr,
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    row_ptr = input_ptr + row * n_cols
    values = tl.load(row_ptr + offsets, mask=mask, other=-float("inf"))
    values = values.to(tl.float32)
    row_max = tl.max(values, axis=0)
    numerators = tl.exp(values - row_max)
    denominator = tl.sum(numerators, axis=0)
    result = numerators / denominator

    tl.store(output_ptr + row * n_cols + offsets, result, mask=mask)


def build(context):
    def wrapper(input, dim, dtype=None):
        if dim < 0:
            dim += input.ndim
        if dim != 1:
            raise ValueError("This implementation supports dim=1 only")

        output_dtype = input.dtype if dtype is None else dtype
        output = torch.empty(input.shape, device=input.device, dtype=output_dtype)
        n_rows, n_cols = input.shape
        _softmax_kernel[(n_rows,)](
            input,
            output,
            n_cols,
            BLOCK_SIZE=1024,
            num_warps=8,
        )
        return output

    return wrapper
