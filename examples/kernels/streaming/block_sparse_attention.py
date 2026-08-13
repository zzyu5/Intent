import math

import intent
import intent.language as I


BATCH = 8
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_GROUP = QUERY_HEADS // KV_HEADS
SEQUENCE = 4096
HEAD_DIMENSION = 128
BLOCK_SIZE = 64
SELECTED_BLOCKS = 32
SPLITS = 4
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


@intent.kernel
def block_sparse_gqa_decode_partials(
    q: I.In[I.f16, ("B", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "D")],
    block_indices: I.In[I.i32, ("B", "HK", "S")],
    cache_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("P_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "P")],
    partial_output: I.Out[I.f32, ("B", "HQ", "P", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    BLOCK_SIZE: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    KV_HEADS = k.shape[2]
    K = k.shape[1]
    S = block_indices.shape[2]
    split_ranges = I.ragged(
        outer=I.domain(0, SPLITS),
        members=I.domain(0, S),
        offsets=split_offsets,
    )
    block_tokens = I.domain(0, BLOCK_SIZE)
    local_query_head_axis = I.domain(0, HEAD_GROUP)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, KV_HEADS)):
            I.assume_in_bounds(key_head, block_indices, axis=1)
            for query_region in I.parallel(
                I.partition(local_query_head_axis, extent=HEAD_GROUP)
            ):
                query_head_indices = (
                    key_head * HEAD_GROUP + I.indices(query_region)
                )
                I.assume_in_bounds(query_head_indices, q, axis=1)
                query = I.gather(
                    q,
                    index=(batch, query_head_indices, slice(None)),
                )
                for split in I.parallel(split_ranges.outer):
                    selected = split_ranges[split]
                    stream = I.state_stream(
                        selected,
                        extent=1,
                        init=(
                            I.full((query_region,), -I.inf, dtype=I.f32),
                            I.zeros((query_region,), dtype=I.f32),
                            I.zeros((query_region, D), dtype=I.f32),
                        ),
                        stop=I.end(selected),
                    )
                    with stream:
                        for selected_region, (
                            maximum,
                            denominator,
                            accumulator,
                        ) in stream:
                            selected_index = I.indices(selected_region)
                            I.assume_in_bounds(
                                selected_index, block_indices, axis=2
                            )
                            selected_block = I.reshape(
                                block_indices[batch, key_head, selected_index],
                                (),
                            )
                            selected_block = I.cast(selected_block, I.index)
                            token_index = (
                                I.mask(
                                    selected_block,
                                    valid=selected_block >= 0,
                                    fill=I.cast(0, I.index),
                                )
                                * BLOCK_SIZE
                                + I.indices(block_tokens)
                            )
                            valid_token = (
                                (selected_block >= 0)
                                and (token_index < K)
                                and (
                                    token_index
                                    < I.cast(cache_lengths[batch], I.index)
                                )
                            )
                            key_block = I.gather(
                                k,
                                index=(
                                    batch,
                                    token_index,
                                    key_head,
                                    slice(None),
                                ),
                            )
                            value_block = I.gather(
                                v,
                                index=(
                                    batch,
                                    token_index,
                                    key_head,
                                    slice(None),
                                ),
                            )
                            scores = I.contract(
                                query,
                                key_block,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            scores = I.mask(
                                scores * (scale * I.LOG2E),
                                valid=valid_token,
                                fill=-I.inf,
                            )
                            local_maximum = I.reduce.max(
                                scores,
                                axis=1,
                                identity=-I.inf,
                            )
                            next_maximum = I.maximum(
                                maximum, local_maximum
                            )
                            safe_maximum = I.mask(
                                next_maximum,
                                valid=next_maximum != -I.inf,
                                fill=0.0,
                            )
                            old_scale = I.exp2(maximum - safe_maximum)
                            probability = I.exp2(
                                scores - safe_maximum[:, None]
                            )
                            local_denominator = I.reduce.sum(
                                probability,
                                axis=1,
                                identity=0.0,
                                acc_dtype=I.f32,
                            )
                            contribution = I.contract(
                                I.cast(probability, I.f16),
                                value_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                            stream.yield_(
                                next_maximum,
                                denominator * old_scale
                                + local_denominator,
                                accumulator * old_scale[:, None]
                                + contribution,
                            )
                    maximum, denominator, accumulator = stream.result
                    safe_denominator = I.mask(
                        denominator,
                        valid=denominator > 0.0,
                        fill=1.0,
                    )
                    I.scatter_unique(
                        partial_lse,
                        index=(batch, query_head_indices, split),
                        value=maximum
                        + I.log(safe_denominator) * I.LOG2E,
                    )
                    I.scatter_unique(
                        partial_output,
                        index=(
                            batch,
                            query_head_indices,
                            split,
                            slice(None),
                        ),
                        value=accumulator / safe_denominator[:, None],
                    )


@intent.kernel
def block_sparse_gqa_decode_combine(
    partial_lse: I.In[I.f32, ("B", "HQ", "P")],
    partial_output: I.In[I.f32, ("B", "HQ", "P", "D")],
    output: I.Out[I.f16, ("B", "HQ", "D")],
):
    B, HQ, P, D = partial_output.shape
    splits = I.domain(0, P)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            lse = partial_lse[batch, query_head, splits]
            maximum = I.reduce.max(lse, axis=0, identity=-I.inf)
            safe_maximum = I.mask(
                maximum,
                valid=maximum != -I.inf,
                fill=0.0,
            )
            weights = I.exp2(lse - safe_maximum)
            denominator = I.reduce.sum(
                weights, axis=0, identity=0.0, acc_dtype=I.f32
            )
            safe_denominator = I.mask(
                denominator,
                valid=denominator > 0.0,
                fill=1.0,
            )
            for dimension_region in I.parallel(
                I.partition(dimensions, extent=I.auto("D_TILE"))
            ):
                numerator = I.reshape(
                    I.reduce.sum(
                        weights[:, None]
                        * partial_output[
                            batch, query_head, splits, dimension_region
                        ],
                        axis=0,
                        identity=0.0,
                        acc_dtype=I.f32,
                    ),
                    (dimension_region,),
                )
                output[batch, query_head, dimension_region] = I.cast(
                    numerator / safe_denominator, I.f16
                )
