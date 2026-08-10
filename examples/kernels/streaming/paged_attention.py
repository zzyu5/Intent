import math

import intent
import intent.language as I


BATCH = 8
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_GROUP = QUERY_HEADS // KV_HEADS
PAGE_SIZE = 64
HEAD_DIMENSION = 128
SEQUENCE_LENGTHS = (8191, 7937, 7683, 7429, 7175, 6921, 6667, 6413)
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


@intent.kernel
def paged_gqa_decode_attention(
    q: I.In[I.f16, ("B", "HQ", "D")],
    key_cache: I.In[I.f16, ("P", "PS", "HK", "D")],
    value_cache: I.In[I.f16, ("P", "PS", "HK", "DV")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    output: I.Out[I.f16, ("B", "HQ", "DV")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, D = q.shape
    _, PS, _, _ = key_cache.shape
    DV = value_cache.shape[-1]
    page_slots = page_indices.shape[0]
    pages = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, page_slots),
        offsets=page_offsets,
        indices=page_indices,
    )
    page_tokens = I.domain(0, PS)
    for batch in I.parallel(pages.outer):
        I.assume_in_bounds(batch, page_offsets, axis=0)
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = q[batch, query_head, :][None, :]
            page_axis = pages[batch]
            page_stream = I.state_stream(
                page_axis,
                extent=1,
                init=(
                    I.full((1,), -I.inf, dtype=I.f32),
                    I.zeros((1,), dtype=I.f32),
                    I.zeros((1, DV), dtype=I.f32),
                ),
                stop=I.end(page_axis),
            )
            with page_stream:
                for page_region, (maximum, denominator, accumulator) in page_stream:
                    physical_page = I.members(page_region)
                    I.assume_in_bounds(physical_page, key_cache, axis=0)
                    I.assume_in_bounds(physical_page, value_cache, axis=0)
                    page_ordinal = I.indices(page_region) - page_begin
                    token_stream = I.state_stream(
                        page_tokens,
                        extent=I.auto("K_TILE"),
                        init=(maximum, denominator, accumulator),
                        stop=I.end(page_tokens),
                    )
                    with token_stream:
                        for token_region, (
                            token_maximum,
                            token_denominator,
                            token_accumulator,
                        ) in token_stream:
                            token_index = I.indices(token_region)
                            I.assume_in_bounds(key_head, key_cache, axis=2)
                            I.assume_in_bounds(key_head, value_cache, axis=2)
                            logical_token = I.reshape(
                                page_ordinal[:, None] * PAGE_SIZE
                                + token_index[None, :],
                                (token_region,),
                            )
                            token_valid = logical_token < sequence_length
                            key = I.reshape(
                                key_cache[
                                    physical_page,
                                    token_region,
                                    key_head,
                                    :,
                                ],
                                (token_region, D),
                            )
                            value = I.reshape(
                                value_cache[
                                    physical_page,
                                    token_region,
                                    key_head,
                                    :,
                                ],
                                (token_region, DV),
                            )
                            scores = I.contract(
                                query,
                                key,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            scores = scores * (scale * I.LOG2E)
                            scores = I.mask(
                                scores,
                                valid=token_valid[None, :],
                                fill=-I.inf,
                            )
                            local_maximum = I.reduce.max(
                                scores,
                                axis=1,
                                identity=-I.inf,
                            )
                            next_maximum = I.maximum(
                                token_maximum,
                                local_maximum,
                            )
                            normalization_maximum = I.mask(
                                next_maximum,
                                valid=next_maximum != -I.inf,
                                fill=0.0,
                            )
                            alpha = I.exp2(
                                token_maximum - normalization_maximum
                            )
                            probability = I.exp2(
                                scores - normalization_maximum[:, None]
                            )
                            next_denominator = (
                                alpha * token_denominator
                                + I.reduce.sum(
                                    probability,
                                    axis=1,
                                    identity=0.0,
                                )
                            )
                            next_accumulator = (
                                alpha[:, None] * token_accumulator
                                + I.contract(
                                    I.cast(probability, I.f16),
                                    value,
                                    reduce=((1, 0),),
                                    acc_dtype=I.f32,
                                )
                            )
                            token_stream.yield_(
                                next_maximum,
                                next_denominator,
                                next_accumulator,
                            )
                    inner_maximum, inner_denominator, inner_accumulator = (
                        token_stream.result
                    )
                    page_stream.yield_(
                        inner_maximum,
                        inner_denominator,
                        inner_accumulator,
                    )
            _, denominator, accumulator = page_stream.result
            safe_denominator = I.mask(
                denominator,
                valid=denominator > 0.0,
                fill=1.0,
            )
            output[batch, query_head, :] = I.reshape(
                I.cast(
                    accumulator / safe_denominator[:, None],
                    I.f16,
                ),
                (DV,),
            )
