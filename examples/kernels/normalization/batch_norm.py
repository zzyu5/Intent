import intent
import intent.language as I


BATCH = 32
CHANNELS = 64
SPATIAL = 4096


@intent.fn
def welford_combine(left, right):
    count = left.count + right.count
    delta = right.mean - left.mean
    safe_count = I.maximum(count, I.cast(1, I.i32))
    safe_count_f32 = I.cast(safe_count, I.f32)
    left_count = I.cast(left.count, I.f32)
    right_count = I.cast(right.count, I.f32)
    mean = left.mean + delta * right_count / safe_count_f32
    m2 = (
        left.m2
        + right.m2
        + delta * delta * left_count * right_count / safe_count_f32
    )
    return I.record(count=count, mean=mean, m2=m2)


@intent.kernel
def batch_norm_training(
    x: I.In[I.f16, ("B", "C", "S")],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    running_mean: I.InOut[I.f32, ("C",)],
    running_variance: I.InOut[I.f32, ("C",)],
    output: I.Out[I.f16, ("B", "C", "S")],
    saved_mean: I.Out[I.f32, ("C",)],
    saved_rstd: I.Out[I.f32, ("C",)],
    epsilon: I.f32,
    momentum: I.f32,
):
    B, C, S = x.shape
    batches = I.domain(0, B)
    spatials = I.domain(0, S)
    for channel in I.parallel(I.domain(0, C)):
        stream = I.state_stream(
            spatials,
            extent=I.auto("S_TILE"),
            init=(
                I.cast(0, I.i32),
                I.cast(0.0, I.f32),
                I.cast(0.0, I.f32),
            ),
        )
        with stream:
            for spatial_region, (count, mean, m2) in stream:
                values = I.cast(x[batches, channel, spatial_region], I.f32)
                chunk = I.reduce(
                    I.record(
                        count=I.full(
                            (batches, spatial_region), 1, dtype=I.i32
                        ),
                        mean=values,
                        m2=I.zeros((batches, spatial_region), dtype=I.f32),
                    ),
                    axis=(0, 1),
                    identity=I.record(
                        count=I.cast(0, I.i32),
                        mean=I.cast(0.0, I.f32),
                        m2=I.cast(0.0, I.f32),
                    ),
                    combine=welford_combine,
                )
                merged = welford_combine(
                    I.record(count=count, mean=mean, m2=m2), chunk
                )
                stream.yield_(merged.count, merged.mean, merged.m2)
        count, mean, m2 = stream.result
        total_count = I.cast(count, I.f32)
        variance = m2 / total_count
        rstd = I.rsqrt(variance + epsilon)
        running_mean[channel] = (
            (1.0 - momentum) * running_mean[channel] + momentum * mean
        )
        running_variance[channel] = (
            (1.0 - momentum) * running_variance[channel]
            + momentum
            * variance
            * total_count
            / (total_count - 1.0)
        )
        output_stream = I.state_stream(
            spatials,
            extent=I.auto("S_TILE"),
            init=(mean, rstd),
        )
        with output_stream:
            for spatial_region, (
                carried_mean,
                carried_rstd,
            ) in output_stream:
                values = I.cast(x[batches, channel, spatial_region], I.f32)
                output[batches, channel, spatial_region] = I.cast(
                    (values - carried_mean)
                    * carried_rstd
                    * weight[channel]
                    + bias[channel],
                    I.f16,
                )
                output_stream.yield_(carried_mean, carried_rstd)
        saved_mean[channel] = mean
        saved_rstd[channel] = rstd
