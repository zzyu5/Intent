import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_COS_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK": 2048}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_COS_CONFIGS, key=["n_elements"])
@triton.jit
def _cos_contiguous(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, libdevice.cos(values), mask=mask)


@triton.autotune(configs=_COS_CONFIGS, key=["n_elements"])
@triton.jit
def _cos_strided(input_ptr, output_ptr, n_elements, input_stride, output_stride,
                 BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets * input_stride, mask=mask, other=0.0)
    tl.store(output_ptr + offsets * output_stride, libdevice.cos(values), mask=mask)


def launch(input, output):
    n_elements = input.numel()
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
    if input.stride(0) == 1 and output.stride(0) == 1:
        return _cos_contiguous[grid](input, output, n_elements)
    return _cos_strided[grid](
        input, output, n_elements, input.stride(0), output.stride(0)
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
