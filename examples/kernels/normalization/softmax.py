import intent
import intent.language as I

from kernels.streaming.online_softmax import online_softmax_summary


ROWS = 8192
COLUMNS = 8192
ROW_MAJOR_NOALIAS = I.constraints(
    strides=(None, 1),
    noalias=True,
)


@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
    y: I.Out[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        maximum = I.reduce.max(values, axis=0, identity=-I.inf)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0, identity=0.0)
        y[row, columns] = numerator / denominator


@intent.kernel
def stable_softmax_f16(
    x: I.In[I.f16, ("M", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        maximum = I.reduce.max(values, axis=0, identity=-I.inf)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0, identity=0.0)
        y[row, columns] = I.cast(numerator / denominator, I.f16)


@intent.kernel
def chunked_softmax_bf16(
    x: I.In[I.bf16, ("M", "N")],
    y: I.Out[I.bf16, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = I.cast(x[row, columns], I.f32)
        summary = online_softmax_summary(values)
        safe_denominator = I.select(summary.valid, summary.denominator, 1.0)
        normalized = I.select(
            summary.valid,
            I.exp(values - summary.maximum) / safe_denominator,
            0.0,
        )
        y[row, columns] = I.cast(normalized, I.bf16)
