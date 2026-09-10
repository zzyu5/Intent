import intent
import intent.language as I


@intent.kernel
def logsumexp_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ()],
):
    maximum = I.reduce.max(input, axis=0)
    shifted = input - maximum
    total = I.reduce.sum(I.exp(shifted), axis=0)
    output[()] = I.log(total) + maximum


def build(context):
    compiled = context.compile("logsumexp", logsumexp_kernel)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
