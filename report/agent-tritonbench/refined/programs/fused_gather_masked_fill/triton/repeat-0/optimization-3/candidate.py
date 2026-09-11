import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({}, num_warps=4),
    triton.Config({}, num_warps=8),
    triton.Config({}, num_warps=16),
    triton.Config({}, num_warps=32),
]


@triton.autotune(configs=_CONFIGS, key=["n_rows"])
@triton.jit
def _fused_gather_masked_fill(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    value,
    n_rows,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COLS)
    offsets = row * BLOCK_COLS + cols

    fill = tl.load(mask_ptr + offsets)
    gather = fill == 0
    index = tl.load(index_ptr + offsets, mask=gather, other=0).to(tl.int32)
    gathered = tl.load(
        input_ptr + index * BLOCK_COLS + cols,
        mask=gather,
        other=value,
    )
    tl.store(output_ptr + offsets, tl.where(fill, value, gathered))


def build(context):
    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        if out is None:
            out = torch.empty_like(index, dtype=input.dtype)

        _fused_gather_masked_fill[(index.shape[0],)](
            input,
            index,
            mask,
            out,
            value,
            index.shape[0],
            BLOCK_COLS=1024,
        )
        return out

    return wrapper
