import torch
import intent
import intent.language as I


@intent.kernel
def _chebyshev_t3(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n_elements = input.shape[0]
    two = I.cast(2.0, I.f32)
    one = I.cast(1.0, I.f32)

    for index in I.parallel(I.domain(0, n_elements)):
        x = input[index]
        previous = one
        current = x
        for degree in range(1, 3):
            following = two * x * current - previous
            previous = current
            current = following
        out[index] = current


def build(context):
    kernel = context.compile("chebyshev_polynomial_t", _chebyshev_t3)

    def chebyshev_polynomial_t(input, n, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out)
        return out

    return chebyshev_polynomial_t
