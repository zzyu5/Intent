import intent
import intent.language as I


SAMPLES = 8 * 1024 * 1024
BINS = 256


@intent.kernel
def histogram_256(
    samples: I.In[I.f32, ("N",)],
    histogram: I.InOut[I.f32, (BINS,)],
):
    N = samples.shape[0]
    sample_axis = I.domain(0, N)
    bin_index = I.cast(samples[sample_axis], I.i32)
    I.assume_in_bounds(bin_index, histogram, axis=0)
    I.atomic_add(
        histogram,
        index=(bin_index,),
        value=I.full((sample_axis,), 1.0, dtype=I.f32),
    )
