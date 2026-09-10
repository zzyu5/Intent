import intent
import intent.language as I


@intent.kernel
def fused_mv_sigmoid_sub_kernel(
    matrix: I.In[I.f32, ("M", "K")],
    vector: I.In[I.f32, ("K",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("M",)],
):
    product = I.matvec(matrix, vector, acc_dtype=I.f32)
    sigmoid = 0.5 * (I.tanh(0.5 * product) + 1.0)
    elements = I.domain(0, matrix.shape[0])
    output[elements] = sigmoid[elements] - alpha * other


def build(context):
    compiled = context.compile("fused_mv_sigmoid_sub", fused_mv_sigmoid_sub_kernel)

    def wrapper(input, vec, other, alpha=1, *, out=None):
        if out is None:
            return compiled.run(input, vec, other, alpha)
        compiled(input, vec, other, alpha, out)
        return out

    return wrapper
