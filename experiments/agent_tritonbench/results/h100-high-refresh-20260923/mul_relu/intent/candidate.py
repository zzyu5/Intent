import torch
import intent
import intent.language as I


@intent.kernel
def _mul_relu_tensor(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    zero = I.cast(0.0, I.f32)
    for index in I.parallel(elements):
        product = input[index] * other[index]
        out[index] = I.maximum(product, zero)


@intent.kernel
def _mul_relu_scalar(
    input: I.In[I.f32, ("N",)],
    other: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    zero = I.cast(0.0, I.f32)
    for index in I.parallel(elements):
        product = input[index] * other
        out[index] = I.maximum(product, zero)


def build(context):
    tensor_kernel = context.compile("mul_relu_tensor", _mul_relu_tensor)
    scalar_kernel = context.compile("mul_relu_scalar", _mul_relu_scalar)

    def mul_relu(input, other, inplace=False, out=None):
        if inplace:
            result = input
        elif out is None:
            result = torch.empty_like(input)
        else:
            result = out

        if isinstance(other, torch.Tensor):
            tensor_kernel(input, other, result)
        else:
            scalar_kernel(input, float(other), result)
        return result

    return mul_relu
