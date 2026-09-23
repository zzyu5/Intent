import torch
import intent
import intent.language as I
from typing import Tuple


@intent.kernel
def erfc_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    erfc_out: I.Out[I.f32, ("N",)],
    sqrt_out: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    for element in I.parallel(elements):
        value = input[element]
        erfc_out[element] = I.erfc(value)
        sqrt_out[element] = I.sqrt(value)


def build(context):
    artifact = context.compile("erfc_sqrt_pointwise", erfc_sqrt_kernel)

    def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        erfc_out = torch.empty_like(input)
        sqrt_out = torch.empty_like(input)
        artifact(input, erfc_out, sqrt_out)
        return erfc_out, sqrt_out

    return erfc_sqrt
