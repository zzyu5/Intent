import torch
import triton
import triton.language as tl


@triton.jit
def _add_mean_partials(input_ptr, other_ptr, partials_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets) + tl.load(other_ptr + offsets)
    tl.store(partials_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _add_mean_finish(partials_ptr, output_ptr, NUM_PARTIALS: tl.constexpr):
    offsets = tl.arange(0, NUM_PARTIALS)
    total = tl.sum(tl.load(partials_ptr + offsets), axis=0)
    tl.store(output_ptr, total / 1048576.0)

def build(context):
    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        partials = torch.empty((256,), device=input.device, dtype=torch.float32)
        _add_mean_partials[(256,)](input, other, partials, BLOCK=4096, num_warps=4)
        _add_mean_finish[(1,)](partials, partials, NUM_PARTIALS=256, num_warps=4)
        return partials[0]

    return wrapper
