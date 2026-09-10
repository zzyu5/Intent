import intent
import intent.language as I


@intent.kernel
def ifftshift_1d(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    source = (elements + input.shape[0] // 2) % input.shape[0]
    output[elements] = input[source]


def build(context):
    compiled = context.compile("ifftshift_1d", ifftshift_1d)

    def wrapper(input, dim=None):
        return compiled.run(input)

    return wrapper
