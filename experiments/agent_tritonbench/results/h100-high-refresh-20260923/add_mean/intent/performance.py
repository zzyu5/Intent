import torch
import intent
import intent.language as I


@intent.kernel
def _add_mean_all(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ()],
    alpha: I.f32,
):
    n, = input.shape
    elements = I.domain(0, n)
    value = input[elements] + alpha * other[elements]
    total = I.reduce.sum(value, axis=0, acc_dtype=I.f32)
    out[()] = total / I.cast(n, I.f32)


def build(context):
    add_mean_kernel = context.compile("add_mean_all", _add_mean_all, constexprs={})

    def add_mean(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        result = out
        if result is None:
            result = torch.empty((), device=input.device, dtype=torch.float32)
        add_mean_kernel(input, other, result, float(alpha))
        return result

    return add_mean
