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
    for i0 in I.parallel(d0):
        for i1 in I.parallel(d1):
            for i2 in I.parallel(d2):
                for i3 in I.parallel(d3):
                    output[i3, i2, i1, i0] = input[i0, i1, i2, i3]


def build(context):
    compiled = context.compile("permute_copy", permute_copy_kernel)

    def wrapper(input, dims):
        return compiled.run(input)

    return wrapper
