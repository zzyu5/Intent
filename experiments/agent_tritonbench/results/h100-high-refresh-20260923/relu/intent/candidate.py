import torch
import intent
import intent.language as I


def build(context):
    @intent.kernel
    def relu_kernel(
        x: I.In[I.f32, ("N",)],
        y: I.Out[I.f32, ("N",)],
    ):
        elements = I.domain(0, x.shape[0])
        value = x[elements]
        y[elements] = I.select(
            value > I.cast(0.0, I.f32),
            value,
            I.cast(0.0, I.f32),
        )

    relu_artifact = context.compile("relu_pointwise", relu_kernel)

    def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = input if inplace else torch.empty_like(input)
        relu_artifact(input, output)
        return output

    return relu
