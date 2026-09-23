import torch
import intent
import intent.language as I


@intent.kernel
def _reverse4_copy(
    input: I.In[I.f32, ("D0", "D1", "D2", "D3")],
    output: I.Out[I.f32, ("D3", "D2", "D1", "D0")],
):
    d0, d1, d2, d3 = input.shape
    axis0 = I.domain(0, d0)
    axis1 = I.domain(0, d1)
    axis2 = I.domain(0, d2)
    axis3 = I.domain(0, d3)

    source = input[axis0, axis1, axis2, axis3]
    output[axis3, axis2, axis1, axis0] = I.transpose(source, (3, 2, 1, 0))


def build(context):
    reverse4 = context.compile("permute_copy_reverse4", _reverse4_copy)

    def permute_copy(input, dims):
        output_shape = tuple(input.shape[d] for d in dims)
        output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        reverse4(input, output)
        return output

    return permute_copy
