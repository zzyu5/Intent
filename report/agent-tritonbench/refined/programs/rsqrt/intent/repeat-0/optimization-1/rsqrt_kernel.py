import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["n_elements"])
@triton.jit
def _rsqrt_contiguous(input, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets, mask=mask)
    tl.store(output + offsets, tl.rsqrt(values), mask=mask)


@triton.jit
def _rsqrt_strided(
    input,
    output,
    n_elements,
    input_stride,
    output_stride,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input + offsets * input_stride, mask=mask)
    tl.store(output + offsets * output_stride, tl.rsqrt(values), mask=mask)


def launch(input, output):
    n_elements = input.numel()
    if input.stride(0) == 1 and output.stride(0) == 1:
        grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
        return _rsqrt_contiguous[grid](input, output, n_elements)

    grid = (triton.cdiv(n_elements, 256),)
    return _rsqrt_strided[
        grid
    ](input, output, n_elements, input.stride(0), output.stride(0), BLOCK=256)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
