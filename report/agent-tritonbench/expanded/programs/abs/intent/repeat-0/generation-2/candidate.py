import intent
import intent.language as I


@intent.kernel
def absolute_value(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    output[elements] = I.maximum(values, -values)


def build(context):
    compiled = context.compile("absolute_value", absolute_value)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
