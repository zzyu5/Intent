import intent
import intent.language as I


N = 1048576


@intent.kernel
def cosine(
    input: I.In[I.f32, (N,)],
    output: I.Out[I.f32, (N,)],
):
    elements = I.domain(0, N)
    output[elements] = I.cos(input[elements])


def build(context):
    compiled = context.compile("cosine_v2", cosine)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
