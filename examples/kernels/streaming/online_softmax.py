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
