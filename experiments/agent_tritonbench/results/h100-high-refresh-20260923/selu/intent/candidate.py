import torch
import intent
import intent.language as I


@intent.kernel
def selu_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)

    for i in I.parallel(I.domain(0, n)):
        x = input[i]
        positive = I.maximum(x, zero)
        negative = I.minimum(x, alpha * (I.exp(x) - one))
        output[i] = scale * (positive + negative)


def build(context):
    artifact = context.compile("selu_elementwise", selu_kernel)

    def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = torch.empty_like(input)
        artifact(input, output)
        return output

    return selu
