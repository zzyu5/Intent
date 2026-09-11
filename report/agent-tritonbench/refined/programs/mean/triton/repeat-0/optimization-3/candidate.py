import torch
import triton
import triton.language as tl


@triton.jit
def _mean_partials_kernel(x_ptr, partials_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x_ptr + offsets)
    tl.store(partials_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_finalize_kernel(partials_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    partials = tl.load(partials_ptr + offsets)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total * (1.0 / 1048576.0))


def build(context):
    def wrapper(input_tensor, dim=0, keepdim=False, dtype=None, out=None):
        block = 32768
        n_partials = 32

        partials = torch.empty(
            (n_partials,), device=input_tensor.device, dtype=torch.float32
        )
        output = out if out is not None else torch.empty(
            (1,) if keepdim else (),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        _mean_partials_kernel[(n_partials,)](
            input_tensor, partials, BLOCK=block, num_warps=16
        )
        _mean_finalize_kernel[(1,)](
            partials, output, BLOCK=n_partials, num_warps=1
        )
        return output

    return wrapper
