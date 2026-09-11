import torch
import triton
import triton.language as tl


_BLOCK = 4096


@triton.jit
def _mean(input_tensor, output, n_elements, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    total = tl.zeros([BLOCK], dtype=tl.float32)

    for base in tl.range(0, n_elements, BLOCK):
        indices = base + offsets
        values = tl.load(input_tensor + indices, mask=indices < n_elements, other=0.0)
        total += values

    mean = tl.sum(total, axis=0) / tl.cast(n_elements, tl.float32)
    tl.store(output, mean)


def launch(input_tensor, output):
    n_elements = input_tensor.shape[0]
    _mean[(1,)](
        input_tensor,
        output,
        n_elements,
        BLOCK=_BLOCK,
        num_warps=8,
        num_stages=2,
    )


def run(input_tensor):
    output = torch.empty((), device=input_tensor.device, dtype=torch.float32)
    launch(input_tensor, output)
    return output
