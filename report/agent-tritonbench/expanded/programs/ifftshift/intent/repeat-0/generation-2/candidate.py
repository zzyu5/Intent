import intent
import intent.language as I


@intent.kernel
def ifftshift_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    destination = I.domain(0, input.shape[0])
    indices = I.indices(destination)
    shift = input.shape[0] // 2
    source = (indices + shift) % input.shape[0]

    output[destination] = input[source]


def build(context):
    compiled = context.compile("ifftshift", ifftshift_kernel)

    def wrapper(input, dim=None):
        return compiled.run(input)

    return wrapper
