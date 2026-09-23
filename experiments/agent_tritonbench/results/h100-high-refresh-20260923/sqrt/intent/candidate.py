import torch
import intent
import intent.language as I


@intent.kernel
def sqrt_kernel(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n, = x.shape
    elements = I.domain(0, n)
    for i in I.parallel(elements):
        out[i] = I.sqrt(x[i])


def build(context):
    sqrt_artifact = context.compile("sqrt_elementwise", sqrt_kernel)

    def run(x):
        x_flat = x.reshape(-1)
        out = torch.empty_like(x)
        out_flat = out.reshape(-1)
        sqrt_artifact(x_flat, out_flat)
        return out

    return run
