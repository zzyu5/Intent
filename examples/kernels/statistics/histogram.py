import intent
import intent.language as I


SAMPLES = 8 * 1024 * 1024
BINS = 256


@intent.kernel
def histogram_256(
    samples: I.In[I.u8, ("N",)],
    histogram: I.InOut[I.i32, (BINS,)],
):
    N = samples.shape[0]
    for index in I.parallel(I.domain(0, N)):
        bin_index = I.cast(samples[index], I.i32)
        I.assume_in_bounds(bin_index, histogram, axis=0)
        I.atomic_add(
            histogram,
            index=(bin_index,),
            value=I.cast(1, I.i32),
        )
