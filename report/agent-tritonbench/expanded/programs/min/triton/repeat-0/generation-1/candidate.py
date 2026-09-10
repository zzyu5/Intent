import torch
import triton
import triton.language as tl


@triton.jit
def _min_dim0_kernel(
    input_ptr,
    values_ptr,
    indices_ptr,
    n_rows,
    STRIDE0: tl.constexpr,
    STRIDE1: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    columns = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    input_offsets = columns * STRIDE1

    min_value = tl.load(input_ptr + input_offsets)
    min_index = tl.zeros((BLOCK_N,), dtype=tl.int64)

    for row in tl.range(1, n_rows, num_stages=4):
        value = tl.load(input_ptr + row * STRIDE0 + input_offsets)
        value_is_nan = tl.isnan(value)
        min_is_nan = tl.isnan(min_value)
        better = (value < min_value) | (value_is_nan & ~min_is_nan)
        min_value = tl.where(better, value, min_value)
        min_index = tl.where(better, row, min_index)

    tl.store(values_ptr + columns * VALUE_STRIDE, min_value)
    tl.store(indices_ptr + columns * INDEX_STRIDE, min_index)


def build(context):
    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim < 0:
            dim += input.ndim
        if dim != 0:
            raise ValueError("the fixed kernel only supports dim=0")

        n_columns = input.shape[1]
        if out is None:
            output_shape = (1, n_columns) if keepdim else (n_columns,)
            values = torch.empty(output_shape, device=input.device, dtype=input.dtype)
            indices = torch.empty(output_shape, device=input.device, dtype=torch.long)
        else:
            values, indices = out

        block_n = 128
        _min_dim0_kernel[(triton.cdiv(n_columns, block_n),)](
            input,
            values,
            indices,
            input.shape[0],
            STRIDE0=input.stride(0),
            STRIDE1=input.stride(1),
            VALUE_STRIDE=values.stride(-1),
            INDEX_STRIDE=indices.stride(-1),
            BLOCK_N=block_n,
            num_warps=4,
        )
        return values, indices

    return wrapper
