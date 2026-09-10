import intent
import intent.language as I


@intent.kernel
def abs_kernel(
    x: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, x.shape[0])
    output[elements] = I.abs(x[elements])


def build(context):
    compiled = context.compile("abs_kernel", abs_kernel)

    def wrapper(x):
        return compiled.run(x)

    return wrapper
