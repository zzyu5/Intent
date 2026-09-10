import intent
import intent.language as I


@intent.kernel
def mean_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    total = I.reduce.sum(input_tensor, axis=0)
    count = I.cast(input_tensor.shape[0], I.f32)
    output[()] = I.fdiv(total, count)


def build(context):
    compiled = context.compile("mean", mean_kernel)

    def wrapper(input_tensor, dim, keepdim, dtype=None, out=None):
        if out is None:
            return compiled.run(input_tensor)
        compiled(input_tensor, out)
        return out

    return wrapper
