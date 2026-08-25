import intent
import intent.language as I


SAMPLES = 8 * 1024 * 1024
BINS = 256


@intent.kernel
def histogram_256(
    samples: I.In[I.f32, ("N",)],
    histogram: I.Out[I.f32, (BINS,)],
):
    N = samples.shape[0]
    sample_axis = I.domain(0, N)
    bin_index = I.cast(samples[sample_axis], I.i32)
    valid = (bin_index >= 0) & (bin_index < BINS)
    counts = I.histogram(
        bin_index,
        bins=BINS,
        valid=valid,
        count_dtype=I.u32,
    )
    bins = I.domain(0, BINS)
    histogram[bins] = I.cast(counts, I.f32)
