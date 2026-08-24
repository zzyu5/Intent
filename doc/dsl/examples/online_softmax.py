import intent
import intent.language as I


@intent.fn
def merge_softmax_summary(lhs, rhs):
    maximum = I.maximum(lhs.maximum, rhs.maximum)
    denominator = (
        I.exp(lhs.maximum - maximum) * lhs.denominator
        + I.exp(rhs.maximum - maximum) * rhs.denominator
    )
    return I.record(maximum=maximum, denominator=denominator)


@intent.kernel
def online_softmax(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)

    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        element_summaries = I.record(
            maximum=values,
            denominator=I.full((N,), fill=1.0, dtype=I.f32),
        )
        summary = I.reduce(
            element_summaries,
            axis=0,
            identity=I.record(maximum=-I.inf, denominator=0.0),
            combine=merge_softmax_summary,
        )
        output[row, columns] = (
            I.exp(values - summary.maximum) / summary.denominator
        )
