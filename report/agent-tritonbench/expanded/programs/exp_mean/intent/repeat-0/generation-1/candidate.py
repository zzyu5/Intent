import intent
import intent.language as I


@intent.kernel
def exp_mean_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    exponentials = I.exp2(input[elements] * 1.4426950408889634)
    total = I.reduce.sum(exponentials, axis=0)
    output[()] = I.fdiv(total, I.cast(input.shape[0], I.f32))


def build(context):
    compiled = context.compile("exp_mean", exp_mean_kernel)

    def wrapper(input, dim=None, keepdim=False, dtype=None, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
