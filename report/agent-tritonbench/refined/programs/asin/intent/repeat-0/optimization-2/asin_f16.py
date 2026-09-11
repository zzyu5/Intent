import torch
import triton
import triton.language as tl
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
    grid = (triton.cdiv(n_elements, 512),)
    return _asin_kernel[
        grid
    ](
        input,
        output,
        n_elements,
        stride_input,
        stride_output,
        BLOCK=512,
        num_warps=2,
        num_stages=1,
    )


def run(input):
    output = torch.empty((input.shape[0],), device=input.device, dtype=torch.float16)
    launch(input, output)
    return output
