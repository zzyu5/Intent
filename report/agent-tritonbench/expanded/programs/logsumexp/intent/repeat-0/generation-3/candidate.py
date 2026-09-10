import intent
import intent.language as I


@intent.kernel
def logsumexp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    maximum = I.reduce.max(values, axis=0)
    shifted = values - maximum
    total = I.reduce.sum(I.exp(shifted), axis=0)
    output[I.domain(0, 1)] = I.log(total) + maximum


def build(context):
    compiled = context.compile("logsumexp", logsumexp_kernel)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            return compiled.run(input).reshape(())
        compiled(input, out.reshape(1))
        return out

    return wrapper
