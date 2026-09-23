import torch
import intent
import intent.language as I


@intent.kernel
def _log_tanh_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    for index in I.parallel(elements):
        out[index] = I.tanh(I.log(input[index]))


def build(context):
    compiled = context.compile("log_tanh_elementwise", _log_tanh_kernel)

    def log_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        compiled(input, out)
        return out

    return log_tanh
