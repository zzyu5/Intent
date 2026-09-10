import torch
import intent
from intent import language as tl


@intent.kernel
def abs_kernel(x: tl.Tensor, n_elements: tl.int32, out: tl.Tensor):
    pid = tl.program_id(0)
    offsets = pid * 256 + tl.arange(0, 256)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask, other=0)
    tl.store(out + offsets, tl.abs(values), mask=mask)


def build(context):
    kernel = context.compile(
        "abs_elementwise_kernel",
        abs_kernel,
        constexprs={},
    )

    def wrapper(x):
        out = torch.empty_like(x)
        kernel(x, x.numel(), out)
        return out

    return wrapper
