import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192
ROW_MAJOR_NOALIAS = I.constraints(
    strides=(None, 1),
    layout="row_major",
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
        statistics = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(I.cast(-I.inf, I.f32), I.cast(0.0, I.f32)),
        )
        with statistics:
            for column_region, (maximum, denominator) in statistics:
                values = I.cast(x[row, column_region], I.f32)
                local_maximum = I.reduce.max(values, axis=0, identity=-I.inf)
                next_maximum = I.maximum(maximum, local_maximum)
                statistics.yield_(
                    next_maximum,
                    I.exp(maximum - next_maximum) * denominator
                    + I.reduce.sum(
                        I.exp(values - next_maximum),
                        axis=0,
                        identity=0.0,
                    ),
                )
        maximum, denominator = statistics.result
        writer = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(maximum, denominator),
        )
        with writer:
            for column_region, (final_maximum, final_denominator) in writer:
                y[row, column_region] = I.cast(
                    I.exp(I.cast(x[row, column_region], I.f32) - final_maximum)
                    / final_denominator,
                    I.bf16,
                )
                writer.yield_(final_maximum, final_denominator)
