import intent
import intent.language as I


BATCH = 32
CHANNELS = 64
SPATIAL = 4096


@intent.kernel
def batch_norm_training(
    x: I.In[I.f16, (BATCH, CHANNELS, SPATIAL)],
    weight: I.In[I.f32, (CHANNELS,)],
    bias: I.In[I.f32, (CHANNELS,)],
    output: I.Out[I.f16, (BATCH, CHANNELS, SPATIAL)],
    saved_mean: I.Out[I.f32, (CHANNELS,)],
    saved_rstd: I.Out[I.f32, (CHANNELS,)],
    epsilon: I.f32,
):
    batches = I.domain(0, BATCH)
    spatial = I.domain(0, SPATIAL)
    inverse_count = 1.0 / (BATCH * SPATIAL)
    for channel in I.parallel(I.domain(0, CHANNELS)):
        values = I.cast(x[batches, channel, spatial], I.f32)
        batch_sums = I.reduce.sum(
            values,
            axis=1,
            identity=0.0,
            acc_dtype=I.f32,
        )
        mean = I.reduce.sum(
            batch_sums,
            axis=0,
            identity=0.0,
            acc_dtype=I.f32,
        ) * inverse_count
        centered = values - mean
        batch_square_sums = I.reduce.sum(
            centered * centered,
            axis=1,
            identity=0.0,
            acc_dtype=I.f32,
        )
        variance = I.reduce.sum(
            batch_square_sums,
            axis=0,
            identity=0.0,
            acc_dtype=I.f32,
        ) * inverse_count
        rstd = I.rsqrt(variance + epsilon)
        normalized = centered * rstd
        output[batches, channel, spatial] = I.cast(
            normalized * weight[channel] + bias[channel],
            I.f16,
        )
        saved_mean[channel] = mean
        saved_rstd[channel] = rstd
