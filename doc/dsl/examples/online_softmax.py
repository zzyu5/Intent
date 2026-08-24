import intent
import intent.language as I


@intent.kernel
def online_softmax(
    x: I.In[I.f32, ("M", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)

    for row in I.parallel(I.domain(0, M)):
        stream = I.state_stream(
            columns,
            init=(I.cast(-I.inf, I.f32), I.cast(0.0, I.f32)),
            stop=I.end(columns),
        )
        with stream:
            for region, (maximum, denominator) in stream:
                values = x[row, region]
                local_maximum = I.reduce.max(
                    values, axis=0, identity=-I.inf
                )
                next_maximum = I.maximum(maximum, local_maximum)
                next_denominator = (
                    I.exp(maximum - next_maximum) * denominator
                    + I.reduce.sum(
                        I.exp(values - next_maximum),
                        axis=0,
                        identity=0.0,
                    )
                )
                stream.yield_(next_maximum, next_denominator)

        maximum, denominator = stream.result
        output[row, columns] = I.exp(x[row, columns] - maximum) / denominator
