import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    one = I.cast(1.0, I.f32)
    rows = I.domain(0, input.shape[0])
    for row in I.parallel(rows):
        x = input[row]
        out[row] = I.fdiv(one, one + I.exp(-x))


def build(context):
    sigmoid_artifact = context.compile("sigmoid_pointwise", _sigmoid_kernel)

    def sigmoid(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        sigmoid_artifact(input, out)
        return out

    return sigmoid
