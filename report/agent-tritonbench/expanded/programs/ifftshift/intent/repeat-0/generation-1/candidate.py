import intent
import intent.language as I


@intent.kernel
def ifftshift_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    source = I.domain(0, input.shape[0])
    shift = input.shape[0] // 2
    split = input.shape[0] - shift

    output[source[:split]] = input[source[shift:]]
    output[source[split:]] = input[source[:shift]]


def build(context):
    compiled = context.compile("ifftshift", ifftshift_kernel)

    def wrapper(input, dim=None):
        return compiled.run(input)

    return wrapper
