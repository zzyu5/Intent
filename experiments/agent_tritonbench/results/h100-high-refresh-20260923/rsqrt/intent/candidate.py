import torch
import intent
import intent.language as I


@intent.kernel
def rsqrt_kernel(
    input: I.In[I.f32, (1048576,)],
    out: I.Out[I.f32, (1048576,)],
):
    for i in I.parallel(I.domain(0, 1048576)):
        out[i] = I.rsqrt(input[i])


def build(context):
    artifact = context.compile("rsqrt_kernel", rsqrt_kernel)

    def rsqrt(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        artifact(input, out)
        return out

    return rsqrt
