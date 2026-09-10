import intent
import intent.language as I


@intent.kernel
def mul_sub_kernel(
    input: I.In[I.f32, ("N",)],
    other_mul: I.In[I.f32, ("N",)],
    other_sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    output[elements] = input[elements] * other_mul[elements] - alpha * other_sub[elements]


def build(context):
    compiled = context.compile("mul_sub", mul_sub_kernel)

    def wrapper(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            return compiled.run(input, other_mul, other_sub, alpha)
        compiled(input, other_mul, other_sub, alpha, out)
        return out

    return wrapper
