import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
    ],
    key=["n_elements", "input_stride", "other_stride", "output_stride"],
)
@triton.jit
def _mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    input_stride: tl.constexpr,
    other_stride: tl.constexpr,
    output_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    lhs = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    rhs = tl.load(other_ptr + offsets * other_stride, mask=mask, other=0.0)
    value = tl.maximum(lhs * rhs, 0.0, propagate_nan=tl.PropagateNan.ALL)
    tl.store(output_ptr + offsets * output_stride, value, mask=mask)


def launch(input, other, output):
    n_elements = input.shape[0]
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)
    return _mul_relu_kernel[grid](
        input,
        other,
        output,
        n_elements,
        input_stride=input.stride(0),
        other_stride=other.stride(0),
        output_stride=output.stride(0),
    )


def run(input, other):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, other, output)
    return output
