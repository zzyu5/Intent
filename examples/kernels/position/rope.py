import intent
import intent.language as I


@intent.kernel
def rotary_embedding_flat(
    values: I.In[I.f16, ("R", "D")],
    cosine: I.In[I.f16, ("U", "D_HALF")],
    sine: I.In[I.f16, ("U", "D_HALF")],
    output: I.Out[I.f16, ("R", "D")],
    HEADS: I.Constexpr[int],
):
    R, D = values.shape
    _, half_dimension = cosine.shape
    dimensions = I.domain(0, D)
    for row in I.parallel(I.domain(0, R)):
        token = row // HEADS
        dimension_index = I.indices(dimensions)
        paired_dimension = (dimension_index - half_dimension) % D
        phase_dimension = dimension_index % half_dimension
        rotate_sign = I.cast(
            I.cast(dimension_index >= half_dimension, I.i32) * 2 - 1,
            I.f16,
        )
        output[row, dimensions] = (
            values[row, dimensions] * cosine[token, phase_dimension]
            + values[row, paired_dimension]
            * sine[token, phase_dimension]
            * rotate_sign
        )
