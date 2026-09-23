import torch
import intent
import intent.language as I


@intent.kernel
def floor_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        output[index] = I.floor(input[index])


def build(context):
    compiled_floor = context.compile("floor", floor_kernel)

    def floor(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = torch.empty_like(input) if out is None else out
        compiled_floor(input, result)
        return result

    return floor
