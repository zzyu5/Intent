import intent
import intent.language as I


@intent.kernel
def _softmax_kernel(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    values = x[rows, columns]
    row_max = I.reduce.max(values, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (M, 1))

    weights = I.exp(values - row_max)
    denominator = I.reduce.sum(weights, axis=1, acc_dtype=I.f32)
    denominator = I.reshape(denominator, (M, 1))

    output[rows, columns] = weights / denominator


def build(context):
    artifact = context.compile("softmax_f32_rows", _softmax_kernel, constexprs={})

    def softmax(input, dim, dtype=None):
        return artifact.run(input)

    return softmax
