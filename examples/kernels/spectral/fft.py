import intent
import intent.language as I


BATCH = 1024
FFT_SIZE = 1024
LOG_FFT_SIZE = 10


@intent.kernel
def radix2_fft(
    input_real: I.In[I.f32, (BATCH, FFT_SIZE)],
    input_imag: I.In[I.f32, (BATCH, FFT_SIZE)],
    twiddle_real: I.In[I.f32, (LOG_FFT_SIZE, FFT_SIZE // 2)],
    twiddle_imag: I.In[I.f32, (LOG_FFT_SIZE, FFT_SIZE // 2)],
    output_real: I.Out[I.f32, (BATCH, FFT_SIZE)],
    output_imag: I.Out[I.f32, (BATCH, FFT_SIZE)],
):
    for batch in I.parallel(I.domain(0, BATCH)):
        real = I.buffer((FFT_SIZE,), I.f32, init=0.0)
        imag = I.buffer((FFT_SIZE,), I.f32, init=0.0)

        for index in range(FFT_SIZE):
            source = index
            reversed_index = 0
            for _ in range(LOG_FFT_SIZE):
                reversed_index = (reversed_index << 1) | (source & 1)
                source = source >> 1
            I.store(real, index, input_real[batch, reversed_index])
            I.store(imag, index, input_imag[batch, reversed_index])

        span = 2
        for stage in range(LOG_FFT_SIZE):
            half = span // 2
            for butterfly in range(FFT_SIZE // 2):
                group = butterfly // half
                lane = butterfly % half
                even = group * span + lane
                odd = even + half
                even_real = I.mutable_load(real, even)
                even_imag = I.mutable_load(imag, even)
                odd_real = I.mutable_load(real, odd)
                odd_imag = I.mutable_load(imag, odd)
                weight_real = twiddle_real[stage, lane]
                weight_imag = twiddle_imag[stage, lane]
                rotated_real = odd_real * weight_real - odd_imag * weight_imag
                rotated_imag = odd_real * weight_imag + odd_imag * weight_real
                I.store(real, even, even_real + rotated_real)
                I.store(imag, even, even_imag + rotated_imag)
                I.store(real, odd, even_real - rotated_real)
                I.store(imag, odd, even_imag - rotated_imag)
            span = span * 2

        for index in range(FFT_SIZE):
            output_real[batch, index] = I.mutable_load(real, index)
            output_imag[batch, index] = I.mutable_load(imag, index)
