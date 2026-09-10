import intent
import intent.language as I


@intent.kernel
def mean_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input_tensor.shape[0])
    total = I.reduce.sum(input_tensor[elements], axis=0)
    output[()] = total / I.cast(input_tensor.shape[0], I.f32)


def build(context):
    compiled = context.compile("mean", mean_kernel)

    def wrapper(input_tensor, dim, keepdim, dtype=None, out=None):
        return compiled.run(input_tensor)

    return wrapper
