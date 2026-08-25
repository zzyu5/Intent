import math

import intent
import intent.language as I

from kernels.streaming.attention import empty_attention_summary
from kernels.streaming.attention import merge_attention_summaries
from kernels.streaming.attention import normalize_attention_summary
from kernels.streaming.attention import summarize_attention_chunk_f16
from kernels.streaming.attention import summarize_masked_attention_chunk


BATCH = 8
QUERY_HEADS = 32
KV_HEADS = 8
HEAD_GROUP = QUERY_HEADS // KV_HEADS
PAGE_SIZE = 64
HEAD_DIMENSION = 128
SEQUENCE_LENGTHS = (8191, 7937, 7683, 7429, 7175, 6921, 6667, 6413)
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
SPLITS = 16


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
    DV = value_cache.shape[3]
    for batch in I.parallel(I.domain(0, B)):
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        logical_tokens = I.domain(0, sequence_length)
        token_coordinates = I.indices(logical_tokens)
        page_slot = I.cast(page_offsets[batch], I.index) + (
            token_coordinates // PAGE_SIZE
        )
        I.assume_in_bounds(page_slot, page_indices, axis=0)
        physical_page = I.cast(page_indices[page_slot], I.index)
        I.assume_in_bounds(physical_page, key_cache, axis=0)
        I.assume_in_bounds(physical_page, value_cache, axis=0)
        page_offset = token_coordinates % PAGE_SIZE
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = I.reshape(q[batch, query_head, :], (1, D))
            summary = I.region_fold(
                source=(
                    key_cache[physical_page, page_offset, key_head, :],
                    value_cache[physical_page, page_offset, key_head, :],
                    token_coordinates,
                ),
                axis=0,
                summarize=summarize_attention_chunk_f16,
                combine=merge_attention_summaries,
                identity=empty_attention_summary(1, DV),
                operands=(
                    query,
                    I.full((1,), fill=0, dtype=I.index),
                    scale,
                    False,
                ),
            )
            output[batch, query_head, :] = I.reshape(
                I.cast(normalize_attention_summary(summary), I.f16),
                (DV,),
            )


@intent.kernel
def paged_gqa_decode_partials(
    q: I.In[I.f16, ("B", "HQ", "D")],
    key_cache: I.In[I.f16, ("P", "PS", "HK", "D")],
    value_cache: I.In[I.f16, ("P", "PS", "HK", "DV")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("SP_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.bf16, ("B", "HQ", "SPLITS", "DV")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
    BATCH_SIZE: I.Constexpr[int],
):
    B, HQ, D = q.shape
    PS = key_cache.shape[1]
    HK = key_cache.shape[2]
    DV = value_cache.shape[3]
    page_slots = page_indices.shape[0]
    split_pages = I.ragged(
        outer=I.domain(0, BATCH_SIZE * SPLITS),
        members=I.domain(0, page_slots),
        offsets=split_offsets,
        indices=page_indices,
    )
    page_tokens = I.domain(0, PS)
    local_query_heads = I.domain(0, HEAD_GROUP)
    for job in I.parallel(split_pages.outer):
        batch = job // SPLITS
        split = job % SPLITS
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        selected_pages = split_pages[job]
        page_count = split_offsets[job + 1] - split_offsets[job]
        token_count = page_count * PS
        page_positions = I.indices(selected_pages)
        physical_pages = I.cast(I.members(selected_pages), I.index)
        I.assume_in_bounds(physical_pages, key_cache, axis=0)
        I.assume_in_bounds(physical_pages, value_cache, axis=0)
        token_offsets = I.indices(page_tokens)
        logical_tokens = (
            (page_positions - page_begin)[:, None] * PAGE_SIZE
            + token_offsets[None, :]
        )
        active = I.reshape(logical_tokens < sequence_length, (token_count,))
        for key_head in I.parallel(I.domain(0, HK)):
            query_heads = key_head * HEAD_GROUP + I.indices(local_query_heads)
            query = I.gather(
                q,
                index=(batch, query_heads, slice(None)),
            )
            keys = I.reshape(
                key_cache[
                    physical_pages[:, None],
                    token_offsets[None, :],
                    key_head,
                    :,
                ],
                (token_count, D),
            )
            values = I.reshape(
                value_cache[
                    physical_pages[:, None],
                    token_offsets[None, :],
                    key_head,
                    :,
                ],
                (token_count, DV),
            )
            summary = summarize_masked_attention_chunk(
                keys,
                values,
                I.reshape(logical_tokens, (token_count,)),
                active,
                query,
                I.full((HEAD_GROUP,), fill=0, dtype=I.index),
                scale,
            )
            safe_denominator = I.select(
                summary.valid,
                summary.denominator,
                1.0,
            )
            I.scatter_unique(
                partial_lse,
                index=(batch, query_heads, split),
                value=I.select(
                    summary.valid,
                    summary.maximum + I.log(safe_denominator) * I.LOG2E,
                    -I.inf,
                ),
            )
            I.scatter_unique(
                partial_output,
                index=(batch, query_heads, split, slice(None)),
                value=I.cast(normalize_attention_summary(summary), I.bf16),
            )
