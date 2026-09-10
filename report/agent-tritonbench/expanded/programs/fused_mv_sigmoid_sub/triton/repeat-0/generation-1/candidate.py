import torch
import triton
import triton.language as tl


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
    result = sigmoid - alpha * other
    tl.store(output_ptr + row, result, mask=row < n_rows)


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

        grid = (n_rows,)
        launch_args = (input, vec, output)
        shape_args = (
            n_rows,
            n_cols,
            input.stride(0),
            input.stride(1),
            vec.stride(0),
        )

        if isinstance(other, torch.Tensor):
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
            _fused_mv_sigmoid_sub_scalar[grid](
                *launch_args,
                *shape_args,
                other,
                alpha,
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
