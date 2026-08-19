import intent
import intent.language as I


BATCH = 32
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_DIMENSION = 128
CACHE_LENGTH = 8192
HALF_DIMENSION = HEAD_DIMENSION // 2


@intent.kernel
def padded_rope_cache_update(
    query: I.In[I.f16, ("B", QUERY_HEADS, HEAD_DIMENSION)],
    key: I.In[I.f16, ("B", KV_HEADS, HEAD_DIMENSION)],
    value: I.In[I.f16, ("B", KV_HEADS, HEAD_DIMENSION)],
    sequence_lengths: I.In[I.i32, ("B",)],
    query_output: I.Out[I.f16, ("B", QUERY_HEADS, HEAD_DIMENSION)],
    key_cache: I.InOut[
        I.f16, ("B", CACHE_LENGTH + 1, KV_HEADS, HEAD_DIMENSION)
    ],
    value_cache: I.InOut[
        I.f16, ("B", CACHE_LENGTH + 1, KV_HEADS, HEAD_DIMENSION)
    ],
    theta: I.f32,
    linear_scale: I.f32,
):
    B = query.shape[0]
    halves = I.domain(0, HALF_DIMENSION)
    for batch, head in I.parallel(
        (
            I.domain(0, B),
            I.domain(0, QUERY_HEADS + 2 * KV_HEADS),
        )
    ):
        position = I.cast(sequence_lengths[batch], I.index) - 1
        powers = I.cast(I.indices(halves) * 2, I.f32)
        frequencies = theta ** (powers / I.cast(-HEAD_DIMENSION, I.f32))
        angle = I.cast(position, I.f32) * frequencies / linear_scale
        cos = I.cos(angle)
        sin = I.sin(angle)
        if head < QUERY_HEADS:
            first = I.cast(query[batch, head, halves], I.f32)
            second = I.cast(
                query[batch, head, I.indices(halves) + HALF_DIMENSION],
                I.f32,
            )
            query_output[batch, head, halves] = I.cast(
                first * cos - second * sin, I.f16
            )
            query_output[
                batch, head, I.indices(halves) + HALF_DIMENSION
            ] = I.cast(second * cos + first * sin, I.f16)
        elif head < QUERY_HEADS + KV_HEADS:
            key_head = head - QUERY_HEADS
            first = I.cast(key[batch, key_head, halves], I.f32)
            second = I.cast(
                key[
                    batch,
                    key_head,
                    I.indices(halves) + HALF_DIMENSION,
                ],
                I.f32,
            )
            key_cache[batch, position, key_head, halves] = I.cast(
                first * cos - second * sin, I.f16
            )
            key_cache[
                batch,
                position,
                key_head,
                I.indices(halves) + HALF_DIMENSION,
            ] = I.cast(second * cos + first * sin, I.f16)
        else:
            value_head = head - QUERY_HEADS - KV_HEADS
            value_cache[batch, position, value_head, :] = value[
                batch, value_head, :
            ]
