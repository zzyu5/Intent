import intent
import intent.language as I


@intent.kernel
def permute_copy_kernel(
    input: I.In[I.f32, ("D", "D", "D", "D")],
    output: I.Out[I.f32, ("D", "D", "D", "D")],
):
    size = input.shape[0]
    elements = I.domain(0, size * size * size * size)

    out0 = elements // (size * size * size)
    rem0 = elements % (size * size * size)
    out1 = rem0 // (size * size)
    rem1 = rem0 % (size * size)
    out2 = rem1 // size
    out3 = rem1 % size

    output[out0, out1, out2, out3] = input[out3, out2, out1, out0]


def build(context):
    compiled = context.compile("permute_copy", permute_copy_kernel)

    def wrapper(input, dims):
        return compiled.run(input)

    return wrapper
