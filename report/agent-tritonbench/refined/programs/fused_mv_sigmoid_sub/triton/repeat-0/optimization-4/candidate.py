import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROWS": 1}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 1}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 2}, num_warps=2),
        triton.Config({"BLOCK_ROWS": 2}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 2}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=2),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 4}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 8}, num_warps=16),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=8),
        triton.Config({"BLOCK_ROWS": 16}, num_warps=16),
        triton.Config({"BLOCK_ROWS": 32}, num_warps=4),
        triton.Config({"BLOCK_ROWS": 32}, num_warps=8),
    ],
    key=[],
)
@triton.jit
def _fused_mv_sigmoid_sub_fixed(
    input_ptr,
    vec_ptr,
    output_ptr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, 1024)

    input_tile = input_ptr + rows[:, None] * 1024 + cols[None, :]
    values = tl.load(input_tile, cache_modifier=".cg")
    vector = tl.load(vec_ptr + cols, cache_modifier=".ca")
    dot = tl.sum(values * vector[None, :], axis=1)

    sigmoid = 1.0 / (1.0 + tl.exp(-dot))
    tl.store(output_ptr + rows, sigmoid - 0.5)


@triton.jit
def _fused_mv_sigmoid_sub_scalar(
    input_ptr,
    vec_ptr,
    output_ptr,
    n_rows,
    n_cols,
    input_stride_row,
    input_stride_col,
    vec_stride,
    other,
    alpha,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n_rows
    col_mask = cols < n_cols
    mask = row_mask[:, None] & col_mask[None, :]

    input_tile = input_ptr + rows[:, None] * input_stride_row + cols[None, :] * input_stride_col
    values = tl.load(input_tile, mask=mask, other=0.0)
    vector = tl.load(vec_ptr + cols * vec_stride, mask=col_mask, other=0.0)
    dot = tl.sum(values * vector[None, :], axis=1)

    sigmoid = 1.0 / (1.0 + tl.exp(-dot))
    result = sigmoid - alpha * other
    tl.store(output_ptr + rows, result, mask=row_mask)


@triton.jit
def _fused_mv_sigmoid_sub_tensor(
    input_ptr,
    vec_ptr,
    output_ptr,
    other_ptr,
    n_rows,
    n_cols,
    input_stride_row,
    input_stride_col,
    vec_stride,
    other_stride,
    alpha,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    input_row = input_ptr + row * input_stride_row + cols * input_stride_col
    values = tl.load(input_row, mask=mask, other=0.0)
    vector = tl.load(vec_ptr + cols * vec_stride, mask=mask, other=0.0)
    dot = tl.sum(values * vector, axis=0)

    sigmoid = 1.0 / (1.0 + tl.exp(-dot))
    other_value = tl.load(other_ptr + row * other_stride)
    result = sigmoid - alpha * other_value
    tl.store(output_ptr + row, result, mask=row < n_rows)


@triton.jit
def _copy_1d(source_ptr, destination_ptr, count, destination_stride, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < count
    values = tl.load(source_ptr + offsets, mask=mask)
    tl.store(destination_ptr + offsets * destination_stride, values, mask=mask)


def build(context):
    del context

    def wrapper(input, vec, other, alpha=1, *, out=None):
        n_rows = input.shape[0]
        n_cols = input.shape[1]

        if out is None:
            output = torch.empty((n_rows,), device=input.device, dtype=input.dtype)
            copy_back = None
        else:
            output = torch.empty((n_rows,), device=input.device, dtype=input.dtype)
            copy_back = out

        launch_args = (input, vec, output)
        shape_args = (
            n_rows,
            n_cols,
            input.stride(0),
            input.stride(1),
            vec.stride(0),
        )

        if (
            not isinstance(other, torch.Tensor)
            and n_rows == 1024
            and n_cols == 1024
            and input.stride(1) == 1
            and input.stride(0) == n_cols
            and vec.stride(0) == 1
            and other == 0.5
            and alpha == 1
        ):
            grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
            _fused_mv_sigmoid_sub_fixed[grid](
                input,
                vec,
                output,
            )
        elif isinstance(other, torch.Tensor):
            grid = (n_rows,)
            if other.numel() == 1:
                _fused_mv_sigmoid_sub_tensor[grid](
                    *launch_args,
                    other,
                    *shape_args,
                    0,
                    alpha,
                    BLOCK_SIZE=1024,
                    num_warps=8,
                )
            else:
                _fused_mv_sigmoid_sub_tensor[grid](
                    *launch_args,
                    other,
                    *shape_args,
                    other.stride(0),
                    alpha,
                    BLOCK_SIZE=1024,
                    num_warps=8,
                )
        else:
            _fused_mv_sigmoid_sub_scalar[(triton.cdiv(n_rows, 4),)](
                *launch_args,
                *shape_args,
                other,
                alpha,
                BLOCK_ROWS=4,
                BLOCK_SIZE=1024,
                num_warps=8,
            )

        if copy_back is not None:
            _copy_1d[(triton.cdiv(n_rows, 256),)](
                output,
                copy_back,
                n_rows,
                copy_back.stride(0),
                BLOCK_SIZE=256,
            )
            return copy_back
        return output

    return wrapper
