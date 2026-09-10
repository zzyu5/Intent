import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_masked_fill(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    n_cols,
    input_stride0,
    input_stride1,
    index_stride0,
    index_stride1,
    mask_stride0,
    mask_stride1,
    output_stride0,
    output_stride1,
    value,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    valid = cols < n_cols

    index = tl.load(
        index_ptr + row * index_stride0 + cols * index_stride1,
        mask=valid,
        other=0,
    )
    gathered = tl.load(
        input_ptr + index * input_stride0 + cols * input_stride1,
        mask=valid,
        other=0.0,
    )
    fill = tl.load(
        mask_ptr + row * mask_stride0 + cols * mask_stride1,
        mask=valid,
        other=0,
    )
    result = tl.where(fill, value, gathered)
    tl.store(
        output_ptr + row * output_stride0 + cols * output_stride1,
        result,
        mask=valid,
    )


def build(context):
    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        if out is None:
            out = torch.empty_like(index, dtype=input.dtype)

        _fused_gather_masked_fill[(index.shape[0],)](
            input,
            index,
            mask,
            out,
            index.shape[1],
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            mask.stride(0),
            mask.stride(1),
            out.stride(0),
            out.stride(1),
            value,
            BLOCK_COLS=1024,
        )
        return out

    return wrapper
