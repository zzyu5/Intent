import intent
import intent.language as I


BATCH = 32
CHANNELS = 64
SPATIAL = 4096


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
    spatial = I.domain(0, S)
    for channel in I.parallel(I.domain(0, C)):
        total_count = I.cast(B * S, I.f32)
        inverse_count = 1.0 / total_count
        statistics = I.state_stream(
            spatial,
            extent=I.auto("N_TILE"),
            init=(
                I.cast(0.0, I.f32),
                I.cast(0.0, I.f32),
                I.cast(0.0, I.f32),
            ),
        )
        with statistics:
            for spatial_region, (count, mean, m2) in statistics:
                values = I.cast(x[batches, channel, spatial_region], I.f32)
                spatial_count = I.reduce.sum(
                    I.full((spatial_region,), 1.0, dtype=I.f32),
                    axis=0,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                chunk_count = I.cast(B, I.f32) * spatial_count
                batch_sums = I.reduce.sum(
                    values,
                    axis=1,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                chunk_mean = (
                    I.reduce.sum(
                        batch_sums,
                        axis=0,
                        identity=0.0,
                        acc_dtype=I.f32,
                    )
                    / chunk_count
                )
                chunk_centered = values - chunk_mean
                batch_m2 = I.reduce.sum(
                    chunk_centered * chunk_centered,
                    axis=1,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                chunk_m2 = I.reduce.sum(
                    batch_m2,
                    axis=0,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                next_count = count + chunk_count
                delta = chunk_mean - mean
                next_mean = mean + delta * chunk_count / next_count
                next_m2 = (
                    m2
                    + chunk_m2
                    + delta * delta * count * chunk_count / next_count
                )
                statistics.yield_(next_count, next_mean, next_m2)
        _count, mean, m2 = statistics.result
        variance = m2 * inverse_count
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
        writeback = I.state_stream(
            spatial,
            extent=I.auto("N_TILE"),
            init=(mean, rstd),
        )
        with writeback:
            for spatial_region, (final_mean, final_rstd) in writeback:
                values = I.cast(x[batches, channel, spatial_region], I.f32)
                normalized = (values - final_mean) * final_rstd
                output[batches, channel, spatial_region] = I.cast(
                    normalized * weight[channel] + bias[channel],
                    I.f16,
                )
                writeback.yield_(final_mean, final_rstd)
        saved_mean[channel] = mean
        saved_rstd[channel] = rstd
