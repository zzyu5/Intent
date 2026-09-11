import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 0}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 1024, "SKIP_MASKED": 0}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024, "SKIP_MASKED": 0}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 1024, "SKIP_MASKED": 0}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024, "SKIP_MASKED": 1}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512, "SKIP_MASKED": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 512, "SKIP_MASKED": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512, "SKIP_MASKED": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 1, "BLOCK_N": 256, "SKIP_MASKED": 1}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 2, "BLOCK_N": 512, "SKIP_MASKED": 0}, num_warps=4, num_stages=2),
    ],
    key=[],
)
@triton.jit
def _gather_masked_fill(
    input_ptr,
    index_ptr,
    mask_ptr,
    output_ptr,
    value,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SKIP_MASKED: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    offsets = rows * 1024 + cols

    selected = tl.load(mask_ptr + offsets)
    if SKIP_MASKED:
        active = ~selected
        indices = tl.load(index_ptr + offsets, mask=active, other=0).to(tl.int32)
        input_offsets = indices * 1024 + cols
        gathered = tl.load(input_ptr + input_offsets, mask=active, other=value)
    else:
        indices = tl.load(index_ptr + offsets).to(tl.int32)
        input_offsets = indices * 1024 + cols
        gathered = tl.load(input_ptr + input_offsets)
        gathered = tl.where(selected, value, gathered)
    result = gathered
    tl.store(output_ptr + offsets, result)


def launch(input, index, mask, output, value):
    _gather_masked_fill[
        lambda meta: (1024 // meta["BLOCK_M"], 1024 // meta["BLOCK_N"])
    ](
        input,
        index,
        mask,
        output,
        value,
    )


def run(input, index, mask, value):
    output = torch.empty_like(input)
    launch(input, index, mask, output, value)
    return output
