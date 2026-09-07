import intent
import intent.language as I


@intent.fn
def encode_e2m1(value):
    magnitude = I.maximum(value, -value)
    code = I.select(magnitude > 0.25, I.cast(1, I.u8), I.cast(0, I.u8))
    code = I.select(magnitude >= 0.75, I.cast(2, I.u8), code)
    code = I.select(magnitude > 1.25, I.cast(3, I.u8), code)
    code = I.select(magnitude >= 1.75, I.cast(4, I.u8), code)
    code = I.select(magnitude > 2.5, I.cast(5, I.u8), code)
    code = I.select(magnitude >= 3.5, I.cast(6, I.u8), code)
    code = I.select(magnitude > 5.0, I.cast(7, I.u8), code)
    sign = I.cast(I.bitcast(value, I.u32) >> I.cast(31, I.u32), I.u8)
    return code | (sign << I.cast(3, I.u8))


@intent.kernel
def nvfp4_quantize(
    x: I.In[I.bf16, ("M", "N")],
    global_scale: I.In[I.f32, (1,)],
    packed: I.InOut[I.u8, ("M", "P")],
    scales: I.InOut[I.u8, ("MT", "NT", 32, 4, 4)],
):
    M, N = x.shape
    pairs = I.indices(I.domain(0, 8))
    encoding_scale = global_scale[0]
    for row in I.parallel(I.domain(0, M)):
        for group in I.parallel(I.domain(0, N // 16)):
            columns = group * 16 + pairs * 2
            even = I.cast(x[row, columns], I.f32)
            odd = I.cast(x[row, columns + 1], I.f32)
            absolute = I.maximum(I.maximum(even, -even), I.maximum(odd, -odd))
            maximum = I.reduce.max(absolute, axis=0)
            scale = I.cast(
                I.maximum((maximum / 6.0) * encoding_scale, 1.5258789e-05),
                I.f8e4m3fn,
            )
            scales[row // 128, group // 4, row % 32, (row % 128) // 32, group % 4] = I.bitcast(scale, I.u8)
            multiplier = encoding_scale / I.cast(scale, I.f32)
            low = encode_e2m1(even * multiplier)
            high = encode_e2m1(odd * multiplier)
            packed[row, group * 8 + pairs] = low | (high << I.cast(4, I.u8))
