import intent
import intent.language as I


@intent.kernel
def fused_mv_sigmoid_sub(
    input: I.In[I.f32, ("N", "M")],
    vec: I.In[I.f32, ("M",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    rows = I.domain(0, input.shape[0])
    values = I.matvec(input, vec, acc_dtype=I.f32)
    sigmoid = 0.5 * (I.tanh(0.5 * values) + 1.0)
    output[rows] = sigmoid - alpha * other


def build(context):
    compiled = context.compile("fused_mv_sigmoid_sub", fused_mv_sigmoid_sub)

    def wrapper(input, vec, other, alpha=1, *, out=None):
        if out is None:
            return compiled.run(input, vec, other, alpha)
        compiled(input, vec, other, alpha, out)
        return out

    return wrapper
