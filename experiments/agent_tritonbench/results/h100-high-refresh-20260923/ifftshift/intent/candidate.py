import torch
import intent
import intent.language as I


N = 1048576
SHIFT = N // 2


@intent.kernel
def _ifftshift_kernel(
    input: I.In[I.f32, (N,)],
    output: I.Out[I.f32, (N,)],
):
    positions = I.domain(0, N)
    for position in I.parallel(positions):
        source = (position + SHIFT) % N
        output[position] = input[source]


def build(context):
    artifact = context.compile("ifftshift", _ifftshift_kernel)

    def ifftshift(input, dim=None):
        output = torch.empty_like(input)
        artifact(input, output)
        return output

    return ifftshift
