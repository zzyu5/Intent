import torch
import intent
import intent.language as I


@intent.kernel
def trunc_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    zero = I.cast(0.0, I.f32)
    negative = values < zero
    positive_result = I.floor(values)
    negative_result = -I.floor(-values)
    output[elements] = I.select(negative, negative_result, positive_result)


def build(context):
    artifact = context.compile("trunc_pointwise", trunc_kernel)

    def trunc(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return trunc
