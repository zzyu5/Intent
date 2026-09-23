import torch
import intent
import intent.language as I


@intent.kernel
def tanh_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    for index in I.parallel(I.domain(0, n)):
        output_tensor[index] = I.tanh(input_tensor[index])


def build(context):
    tanh_artifact = context.compile("tanh_kernel", tanh_kernel)

    def tanh(input_tensor, out_tensor=None):
        if out_tensor is None:
            out_tensor = torch.empty_like(input_tensor)
        tanh_artifact(input_tensor, out_tensor)
        return out_tensor

    return tanh
