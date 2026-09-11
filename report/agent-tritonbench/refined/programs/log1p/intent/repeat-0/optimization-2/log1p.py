import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _contiguous_kernel(input, output, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input + offsets)
    tl.store(output + offsets, libdevice.log1p(values))


@triton.jit
def _strided_kernel(
    input,
    output,
    N,
    S0_0,
    S1_0,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(input + offsets * S0_0, mask=mask, other=0.0)
    tl.store(output + offsets * S1_0, libdevice.log1p(values), mask=mask)


def launch(input, output):
    N = input.shape[0]
    if N == 1048576 and input.stride(0) == 1 and output.stride(0) == 1:
        _contiguous_kernel[(4096,)](input, output, BLOCK=256, num_warps=2)
        return

    _strided_kernel[(triton.cdiv(N, 256),)](
        input,
        output,
        N,
        input.stride(0),
        output.stride(0),
        BLOCK=256,
        num_warps=2,
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float32)
    launch(input, output)
    return output
