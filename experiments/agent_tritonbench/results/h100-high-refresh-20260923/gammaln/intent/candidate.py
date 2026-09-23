import torch
import intent
import intent.language as I


@intent.kernel
def _gammaln_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n, = input.shape
    elements = I.domain(0, n)
    output[elements] = I.lgamma(input[elements])


def build(context):
    kernel = context.compile("gammaln_elementwise", _gammaln_kernel, constexprs={})

    def gammaln(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return gammaln
