import torch
import triton
import triton.language as tl


_PARTIAL_BLOCK = 4096
_FINAL_BLOCK = 256


@triton.jit
def _mean_partial(input_tensor, partial, n_elements, input_stride, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_tensor + offsets * input_stride, mask=mask, other=0.0)
    tl.store(partial + pid, tl.sum(values, axis=0))


@triton.jit
def _mean_final(partial, output, n_partials, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_partials
    values = tl.load(partial + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    mean = total / tl.cast(n_elements, tl.float32)
    tl.store(output, mean)


def launch(input_tensor, output):
    n_elements = input_tensor.shape[0]
    n_partials = triton.cdiv(n_elements, _PARTIAL_BLOCK)
    partial = torch.empty((n_partials,), device=input_tensor.device, dtype=torch.float32)

    _mean_partial[(n_partials,)](
        input_tensor,
        partial,
        n_elements,
        input_tensor.stride(0),
        BLOCK=_PARTIAL_BLOCK,
        num_warps=8,
        num_stages=2,
    )
    _mean_final[(1,)](
        partial,
        output,
        n_partials,
        n_elements,
        BLOCK=_FINAL_BLOCK,
        num_warps=4,
        num_stages=2,
    )


def run(input_tensor):
    output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
    launch(input_tensor, output)
    return output
