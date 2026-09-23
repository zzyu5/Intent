import torch
import intent
import intent.language as I


@intent.kernel
def _relu_sqrt_out(
    source: I.In[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
):
    n = source.shape[0]
    for index in I.parallel(I.domain(0, n)):
        value = source[index]
        clipped = I.maximum(value, I.cast(0.0, I.f32))
        result[index] = I.sqrt(clipped)


@intent.kernel
def _relu_sqrt_inplace(
    data: I.InOut[I.f32, ("N",)],
):
    n = data.shape[0]
    for index in I.parallel(I.domain(0, n)):
        value = data[index]
        clipped = I.maximum(value, I.cast(0.0, I.f32))
        data[index] = I.sqrt(clipped)


@intent.kernel
def _relu_sqrt_inplace_to_out(
    data: I.InOut[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
):
    n = data.shape[0]
    for index in I.parallel(I.domain(0, n)):
        value = data[index]
        clipped = I.maximum(value, I.cast(0.0, I.f32))
        computed = I.sqrt(clipped)
        data[index] = computed
        result[index] = computed


def build(context):
    out_kernel = context.compile("relu_sqrt_out", _relu_sqrt_out)
    inplace_kernel = context.compile("relu_sqrt_inplace", _relu_sqrt_inplace)
    inplace_to_out_kernel = context.compile(
        "relu_sqrt_inplace_to_out", _relu_sqrt_inplace_to_out
    )

    def relu_sqrt(input, inplace=False, out=None):
        if inplace:
            if out is None or out is input:
                inplace_kernel(input)
                return input
            inplace_to_out_kernel(input, out)
            return out

        result = out if out is not None else torch.empty_like(input)
        out_kernel(input, result)
        return result

    return relu_sqrt
