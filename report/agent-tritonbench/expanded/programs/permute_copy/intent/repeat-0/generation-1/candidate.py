import intent
import intent.language as I


@intent.kernel
def permute_copy_kernel(
    input: I.In[I.f32, ("D0", "D1", "D2", "D3")],
    output: I.Out[I.f32, ("D3", "D2", "D1", "D0")],
):
    d0 = I.domain(0, input.shape[0])
    d1 = I.domain(0, input.shape[1])
    d2 = I.domain(0, input.shape[2])
    d3 = I.domain(0, input.shape[3])
    output[d3, d2, d1, d0] = input[d0, d1, d2, d3]


def build(context):
    compiled = context.compile("permute_copy", permute_copy_kernel)

    def wrapper(input, dims):
        return compiled.run(input)

    return wrapper
