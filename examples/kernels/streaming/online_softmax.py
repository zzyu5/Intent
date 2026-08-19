import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192


@intent.kernel
def streamed_online_softmax(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
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
                values = x[row, column_region]
                local_maximum = I.reduce.max(values, axis=0, identity=-I.inf)
                next_maximum = I.maximum(maximum, local_maximum)
                scale = I.exp(maximum - next_maximum)
                local_sum = I.reduce.sum(
                    I.exp(values - next_maximum), axis=0, identity=0.0
                )
                statistics.yield_(
                    next_maximum,
                    scale * denominator + local_sum,
                )
        maximum, denominator = statistics.result

        output = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(maximum, denominator),
        )
        with output:
            for column_region, (final_maximum, final_denominator) in output:
                values = x[row, column_region]
                y[row, column_region] = (
                    I.exp(values - final_maximum) / final_denominator
                )
                output.yield_(final_maximum, final_denominator)


@intent.kernel
def streamed_online_softmax_f16(
    x: I.In[I.f16, ("M", "N")],
    y: I.Out[I.f16, ("M", "N")],
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
                        I.exp(values - next_maximum), axis=0, identity=0.0
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
                    I.f16,
                )
                writer.yield_(final_maximum, final_denominator)
