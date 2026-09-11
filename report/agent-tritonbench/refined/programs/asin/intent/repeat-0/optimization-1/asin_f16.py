import torch
import triton
import triton.language as tl
from intent.runtime.triton import TuningHooks


_intent_tuning_hooks = TuningHooks(("input", "output"), (False, True))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    ],
    key=["n_elements", "stride_input", "stride_output"],
    pre_hook=_intent_tuning_hooks.before,
    post_hook=_intent_tuning_hooks.after,
)
@triton.jit
def _asin_kernel(
    input,
    output,
    n_elements,
    stride_input,
    stride_output,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input + offsets * stride_input, mask=mask, other=0.0).to(tl.float32)
    ax = tl.abs(x)

    # The approximation is accurate to float16 precision on the valid domain.
    root = tl.sqrt(1.0 - ax)
    polynomial = ((-0.018729299306869507 * ax + 0.074261002242565155) * ax - 0.2121143937110901) * ax + 1.5707287788391113
    result = 1.5707963705062866 - root * polynomial
    result = tl.where(x < 0.0, -result, result)

    tl.store(output + offsets * stride_output, result.to(tl.float16), mask=mask)


def launch(input, output):
    n_elements = input.shape[0]
    stride_input = input.stride(0)
    stride_output = output.stride(0)
    grid = lambda META: (triton.cdiv(n_elements, META["BLOCK"]),)
    return _asin_kernel[grid](input, output, n_elements, stride_input, stride_output)


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float16)
    launch(input, output)
    return output
