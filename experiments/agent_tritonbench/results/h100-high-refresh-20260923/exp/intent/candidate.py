import torch
import intent
import intent.language as I


@intent.kernel
def exp_kernel(
    input_view: I.In[I.f32, ("N",)],
    output_view: I.Out[I.f32, ("N",)],
):
    length = input_view.shape[0]
    for index in I.parallel(I.domain(0, length)):
        output_view[index] = I.exp(input_view[index])


def build(context):
    compiled_exp = context.compile("exp_pointwise", exp_kernel)

    def exp(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        compiled_exp(input_tensor, out)
        return out

    return exp
