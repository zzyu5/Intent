import intent
import intent.language as I


@intent.kernel
def log_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = I.log(input[elements])


def build(context):
    compiled = context.compile("log_candidate_2", log_kernel)

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out

    return wrapper
