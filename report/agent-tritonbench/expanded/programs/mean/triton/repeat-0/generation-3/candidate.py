import torch
import triton
import triton.language as tl


@triton.jit
def _mean_partials_kernel(x_ptr, partials_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(partials_ptr + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_finalize_kernel(partials_ptr, output_ptr, n_partials, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(partials, axis=0)
    tl.store(output_ptr, total / n_elements)


def build(context):
    def wrapper(input_tensor, dim=0, keepdim=False, dtype=None, out=None):
        n_elements = input_tensor.numel()
        block = 4096
        n_partials = triton.cdiv(n_elements, block)

        partials = torch.empty(
            (n_partials,), device=input_tensor.device, dtype=torch.float32
        )
        output = out if out is not None else torch.empty(
            (1,) if keepdim else (),
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )

        _mean_partials_kernel[(n_partials,)](
            input_tensor, partials, n_elements, BLOCK=block, num_warps=8
        )
        _mean_finalize_kernel[(1,)](
            partials, output, n_partials, n_elements, BLOCK=256, num_warps=4
        )
        return output

    return wrapper
