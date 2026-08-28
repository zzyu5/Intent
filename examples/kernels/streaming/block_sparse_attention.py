import math

import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import summarize_masked_attention_chunk


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
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.f32, ("B", "HQ", "SPLITS", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    BLOCK_SIZE: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
):
    B, HQ, D = q.shape
    KV_HEADS = k.shape[2]
    K = k.shape[1]
    local_query_head_axis = I.domain(0, HEAD_GROUP)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, KV_HEADS)):
            I.assume_in_bounds(key_head, block_indices, axis=1)
            query_head_indices = (
                key_head * HEAD_GROUP + I.indices(local_query_head_axis)
            )
            I.assume_in_bounds(query_head_indices, q, axis=1)
            query = I.gather(
                q,
                index=(batch, query_head_indices, slice(None)),
            )
            for split in I.parallel(I.domain(0, SPLITS)):
                selected_begin = I.cast(split_offsets[split], I.index)
                selected_count = I.cast(
                    split_offsets[split + 1] - split_offsets[split], I.index
                )
                token_count = selected_count * BLOCK_SIZE
                flat_tokens = I.domain(0, token_count)
                flat_token = I.indices(flat_tokens)
                selected_ordinal = flat_token // BLOCK_SIZE
                token_offset = flat_token % BLOCK_SIZE
                selected_index = selected_begin + selected_ordinal
                I.assume_in_bounds(selected_index, block_indices, axis=2)
                selected_block = I.cast(
                    block_indices[batch, key_head, selected_index],
                    I.index,
                )
                safe_block = I.mask(
                    selected_block,
                    valid=selected_block >= 0,
                    fill=I.cast(0, I.index),
                )
                token_index = safe_block * BLOCK_SIZE + token_offset
                valid_token = (
                    (selected_block >= 0)
                    & (token_index < K)
                    & (token_index < I.cast(cache_lengths[batch], I.index))
                )
                safe_token_index = I.select(valid_token, token_index, 0)
                key_block = I.gather(
                    k,
                    index=(
                        batch,
                        safe_token_index,
                        key_head,
                        slice(None),
                    ),
                )
                value_block = I.gather(
                    v,
                    index=(
                        batch,
                        safe_token_index,
                        key_head,
                        slice(None),
                    ),
                )
                flat_valid = valid_token
                summary = I.region_fold(
                    source=(
                        key_block,
                        value_block,
                        token_index,
                        flat_valid,
                    ),
                    axis=0,
                    summarize=summarize_masked_attention_chunk,
                    combine=merge_attention_summaries,
                    identity=empty_attention_summary(local_query_head_axis, D),
                    operands=(
                        query,
                        query_head_indices,
                        scale,
                    ),
                )
                safe_denominator = I.select(
                    summary.valid,
                    summary.denominator,
                    1.0,
                )
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_head_indices, split),
                    value=I.select(
                        summary.valid,
                        summary.maximum + I.log(safe_denominator) * I.LOG2E,
                        -I.inf,
                    ),
                )
                I.scatter_unique(
                    partial_output,
                    index=(
                        batch,
                        query_head_indices,
                        split,
                        slice(None),
                    ),
                    value=I.select(
                        summary.valid[:, None],
                        summary.accumulator / safe_denominator[:, None],
                        0.0,
                    ),
                )


@intent.kernel
def block_sparse_gqa_decode_combine(
    partial_lse: I.In[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.In[I.f32, ("B", "HQ", "SPLITS", "D")],
    output: I.Out[I.f16, ("B", "HQ", "D")],
    SPLITS: I.Constexpr[int],
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
                weights, axis=0, identity=0.0
            )
            safe_denominator = I.mask(
                denominator,
                valid=denominator > 0.0,
                fill=1.0,
            )
            numerator = I.reshape(
                I.reduce.sum(
                    weights[:, None]
                    * partial_output[
                        batch, query_head, splits, dimensions
                    ],
                    axis=0,
                    identity=0.0,
                ),
                (D,),
            )
            output[batch, query_head, dimensions] = I.cast(
                numerator / safe_denominator, I.f16
            )
