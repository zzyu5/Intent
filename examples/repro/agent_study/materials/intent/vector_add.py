import intent
import intent.language as I


@intent.kernel
def vector_add(
    x: I.In[I.f32, ("N",)],
    y: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, x.shape[0])
    output[elements] = x[elements] + y[elements]


def build(context):
    compiled = context.compile("vector_add", vector_add)

    def wrapper(x, y):
        return compiled.run(x, y)

    return wrapper
