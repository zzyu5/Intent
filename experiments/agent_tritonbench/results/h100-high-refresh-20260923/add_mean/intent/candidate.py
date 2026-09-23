import torch
import intent
import intent.language as I


@intent.kernel
def _add_mean_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
    alpha: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements] + other[elements] * alpha
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    output[()] = total / I.cast(n, I.f32)


def build(context):
    kernel = context.compile("add_mean_full_reduce", _add_mean_kernel)

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        if out is None:
            output = torch.empty((), device=input.device, dtype=torch.float32)
        else:
            output = out
        kernel(input, other, output, float(alpha))
        return output

    return add_mean
