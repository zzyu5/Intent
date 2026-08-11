import intent
import intent.language as I


@intent.kernel
def stable_softmax_online(
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
            for region, (maximum, denominator) in statistics:
                values = x[row, region]
                local_maximum = I.reduce.max(values, axis=0, identity=-I.inf)
                next_maximum = I.maximum(maximum, local_maximum)
                local_sum = I.reduce.sum(
                    I.exp(values - next_maximum),
                    axis=0,
                    identity=0.0,
                )
                statistics.yield_(
                    next_maximum,
                    I.exp(maximum - next_maximum) * denominator + local_sum,
                )
        maximum, denominator = statistics.result
        output = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(maximum, denominator),
        )
        with output:
            for region, (final_maximum, final_denominator) in output:
                y[row, region] = (
                    I.exp(x[row, region] - final_maximum) / final_denominator
                )
                output.yield_(final_maximum, final_denominator)


@intent.kernel
def weighted_layer_norm_second_moment(
    x: I.In[I.f32, ("M", "N")],
    weight: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        mean = I.reduce.sum(values, axis=0, identity=0.0) * inverse_features
        normalized = (values - mean) * I.rsqrt(
            I.reduce.sum(values * values, axis=0, identity=0.0)
            * inverse_features
            - mean * mean
            + epsilon
        )
        y[row, columns] = normalized * weight[columns] + bias[columns]
