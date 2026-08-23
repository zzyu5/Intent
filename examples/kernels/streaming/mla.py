import math

import intent
import intent.language as I

from kernels.streaming.attention import online_attention_accumulate


ABSORBED_MLA_BATCH = 1
ABSORBED_MLA_SEQUENCE = 512
ABSORBED_MLA_HEADS = 8
ABSORBED_MLA_NOPE_DIMENSION = 128
ABSORBED_MLA_ROPE_DIMENSION = 64
ABSORBED_MLA_LATENT_DIMENSION = 512
ABSORBED_MLA_VALUE_DIMENSION = 128
ABSORBED_MLA_SCALE = 1.0 / math.sqrt(
    ABSORBED_MLA_NOPE_DIMENSION + ABSORBED_MLA_ROPE_DIMENSION
)

SPARSE_MLA_QUERIES = 64
SPARSE_MLA_HEADS = 8
SPARSE_MLA_KEYS = 4096
SPARSE_MLA_SELECTED_KEYS = 256
SPARSE_MLA_LATENT_DIMENSION = 512
SPARSE_MLA_ROPE_DIMENSION = 64
SPARSE_MLA_SCALE = 1.0 / math.sqrt(
    SPARSE_MLA_LATENT_DIMENSION + SPARSE_MLA_ROPE_DIMENSION
)

PAGED_MLA_BATCH = 8
PAGED_MLA_QUERY_HEADS = 32
PAGED_MLA_KV_HEADS = 1
PAGED_MLA_HEAD_GROUP = PAGED_MLA_QUERY_HEADS // PAGED_MLA_KV_HEADS
PAGED_MLA_PAGE_SIZE = 64
PAGED_MLA_LATENT_DIMENSION = 128
PAGED_MLA_ROPE_DIMENSION = 64
PAGED_MLA_SEQUENCE_LENGTHS = (4096, 4093, 4087, 4081, 4079, 4073, 4069, 4063)
PAGED_MLA_SCALE = 1.0 / math.sqrt(
    PAGED_MLA_LATENT_DIMENSION + PAGED_MLA_ROPE_DIMENSION
)
MLA_DECODE_BATCH = 8
MLA_DECODE_HEADS = 64
MLA_DECODE_SEQUENCE = 8192
MLA_DECODE_LATENT_DIMENSION = 512
MLA_DECODE_ROPE_DIMENSION = 64
MLA_DECODE_SPLITS = 16


@intent.fn
def online_sparse_mla_accumulate(
    maximum,
    normalization_maximum,
    denominator,
    accumulator,
    scores,
    value_block,
):
    alpha = I.exp2(maximum - normalization_maximum)
    probability = I.exp2(scores - normalization_maximum[:, :, None])
    next_denominator = alpha * denominator + I.reduce.sum(
        probability,
        axis=2,
        identity=0.0,
    )
    next_accumulator = alpha[:, :, None] * accumulator + I.contract(
        I.cast(probability, I.bf16),
        value_block,
        reduce=((2, 1),),
        batch=((0, 0),),
        acc_dtype=I.f32,
    )
    return next_denominator, next_accumulator


@intent.kernel
def absorbed_mla_prefill(
    q_latent: I.In[I.f16, ("B", "Q", "H", "C")],
    q_rope: I.In[I.f16, ("B", "Q", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    output: I.Out[I.f16, ("B", "Q", "H", "C")],
    scale: I.f32,
):
    B, Q, H, C = q_latent.shape
    K = latent_cache.shape[1]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            latent_query = q_latent[batch, query_axis, head, :]
            position_query = q_rope[batch, query_axis, head, :]
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_axis,), -I.inf, dtype=I.f32),
                    I.zeros((query_axis,), dtype=I.f32),
                    I.zeros((query_axis, C), dtype=I.f32),
                ),
                stop=I.end(query_axis),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    latent_block = latent_cache[batch, key_region, :]
                    content_scores = I.contract(
                        latent_query,
                        latent_block,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    position_scores = I.contract(
                        position_query,
                        rope_cache[batch, key_region, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    scores = (content_scores + position_scores) * (
                        scale * I.LOG2E
                    )
                    valid = I.indices(query_axis)[:, None] >= I.indices(
                        key_region
                    )[None, :]
                    scores = I.mask(scores, valid=valid, fill=-I.inf)
                    local_maximum = I.reduce.max(
                        scores, axis=1, identity=-I.inf
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    next_denominator, next_accumulator = (
                        online_attention_accumulate(
                            maximum,
                            next_maximum,
                            denominator,
                            accumulator,
                            scores,
                            latent_block,
                        )
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, query_axis, head, :] = I.cast(
                accumulator / denominator[:, None],
                I.f16,
            )


@intent.kernel
def token_sparse_mla_prefill(
    q_latent: I.In[I.bf16, ("Q", "H", "C")],
    q_rope: I.In[I.bf16, ("Q", "H", "DR")],
    latent_cache: I.In[I.bf16, ("K", "C")],
    rope_cache: I.In[I.bf16, ("K", "DR")],
    selected_tokens: I.In[I.i32, ("Q", "T")],
    output: I.Out[I.bf16, ("Q", "H", "C")],
    maximum_output: I.Out[I.f32, ("Q", "H")],
    lse_output: I.Out[I.f32, ("Q", "H")],
    scale: I.f32,
):
    Q, H, C = q_latent.shape
    K = latent_cache.shape[0]
    T = selected_tokens.shape[1]
    query_axis = I.domain(0, Q)
    selection_axis = I.domain(0, T)
    latent_query = q_latent[query_axis, :, :]
    position_query = q_rope[query_axis, :, :]
    stream = I.state_stream(
        selection_axis,
        extent=I.auto("K_TILE"),
        init=(
            I.full((query_axis, H), -I.inf, dtype=I.f32),
            I.zeros((query_axis, H), dtype=I.f32),
            I.zeros((query_axis, H, C), dtype=I.f32),
        ),
    )
    with stream:
        for selection_region, (maximum, denominator, accumulator) in stream:
            token = selected_tokens[query_axis, selection_region]
            token_index = I.cast(token, I.index)
            valid_token = (token_index >= 0) and (token_index < K)
            safe_token = I.mask(
                token_index,
                valid=valid_token,
                fill=I.cast(0, I.index),
            )
            latent_block = I.gather(
                latent_cache,
                index=(safe_token, slice(None)),
                valid=valid_token[:, :, None],
                fill=I.cast(0.0, I.bf16),
            )
            rope_block = I.gather(
                rope_cache,
                index=(safe_token, slice(None)),
                valid=valid_token[:, :, None],
                fill=I.cast(0.0, I.bf16),
            )
            content_scores = I.contract(
                latent_query,
                latent_block,
                reduce=((2, 2),),
                batch=((0, 0),),
                acc_dtype=I.f32,
            )
            position_scores = I.contract(
                position_query,
                rope_block,
                reduce=((2, 2),),
                batch=((0, 0),),
                acc_dtype=I.f32,
            )
            scores = (content_scores + position_scores) * (scale * I.LOG2E)
            scores = I.mask(
                scores,
                valid=valid_token[:, None, :],
                fill=-I.inf,
            )
            local_maximum = I.reduce.max(
                scores, axis=2, identity=-I.inf
            )
            next_maximum = I.maximum(maximum, local_maximum)
            normalization_maximum = I.mask(
                next_maximum,
                valid=next_maximum != -I.inf,
                fill=0.0,
            )
            next_denominator, next_accumulator = online_sparse_mla_accumulate(
                maximum,
                normalization_maximum,
                denominator,
                accumulator,
                scores,
                latent_block,
            )
            stream.yield_(
                next_maximum,
                next_denominator,
                next_accumulator,
            )
    maximum, denominator, accumulator = stream.result
    safe_denominator = I.mask(
        denominator,
        valid=denominator > 0.0,
        fill=1.0,
    )
    output[query_axis, :, :] = I.cast(
        accumulator / safe_denominator[:, :, None],
        I.bf16,
    )
    maximum_output[query_axis, :] = maximum
    lse_output[query_axis, :] = maximum + I.log(safe_denominator) * I.LOG2E


@intent.kernel
def token_sparse_mla_value_prefill(
    query_content: I.In[I.bf16, ("Q", "H", "D")],
    query_rope: I.In[I.bf16, ("Q", "H", "DR")],
    key_content: I.In[I.bf16, ("K", "D")],
    value_cache: I.In[I.bf16, ("K", "DV")],
    key_rope: I.In[I.bf16, ("K", "DR")],
    selected_tokens: I.In[I.i32, ("Q", "T")],
    output: I.Out[I.bf16, ("Q", "H", "DV")],
    scale: I.f32,
):
    Q, H, D = query_content.shape
    K = key_content.shape[0]
    DV = value_cache.shape[1]
    T = selected_tokens.shape[1]
    query_axis = I.domain(0, Q)
    selection_axis = I.domain(0, T)
    content_query = query_content[query_axis, :, :]
    position_query = query_rope[query_axis, :, :]
    stream = I.state_stream(
        selection_axis,
        extent=I.auto("K_TILE"),
        init=(
            I.full((query_axis, H), -I.inf, dtype=I.f32),
            I.zeros((query_axis, H), dtype=I.f32),
            I.zeros((query_axis, H, DV), dtype=I.f32),
        ),
    )
    with stream:
        for selection_region, (maximum, denominator, accumulator) in stream:
            token = selected_tokens[query_axis, selection_region]
            token_index = I.cast(token, I.index)
            valid_token = (token_index >= 0) and (token_index < K)
            safe_token = I.mask(
                token_index,
                valid=valid_token,
                fill=I.cast(0, I.index),
            )
            key_block = I.gather(
                key_content,
                index=(safe_token, slice(None)),
                valid=valid_token[:, :, None],
                fill=I.cast(0.0, I.bf16),
            )
            value_block = I.gather(
                value_cache,
                index=(safe_token, slice(None)),
                valid=valid_token[:, :, None],
                fill=I.cast(0.0, I.bf16),
            )
            rope_block = I.gather(
                key_rope,
                index=(safe_token, slice(None)),
                valid=valid_token[:, :, None],
                fill=I.cast(0.0, I.bf16),
            )
            content_scores = I.contract(
                content_query,
                key_block,
                reduce=((2, 2),),
                batch=((0, 0),),
                acc_dtype=I.f32,
            )
            position_scores = I.contract(
                position_query,
                rope_block,
                reduce=((2, 2),),
                batch=((0, 0),),
                acc_dtype=I.f32,
            )
            scores = (content_scores + position_scores) * (scale * I.LOG2E)
            scores = I.mask(
                scores,
                valid=valid_token[:, None, :],
                fill=-I.inf,
            )
            local_maximum = I.reduce.max(scores, axis=2, identity=-I.inf)
            next_maximum = I.maximum(maximum, local_maximum)
            normalization_maximum = I.mask(
                next_maximum,
                valid=next_maximum != -I.inf,
                fill=0.0,
            )
            next_denominator, next_accumulator = online_sparse_mla_accumulate(
                maximum,
                normalization_maximum,
                denominator,
                accumulator,
                scores,
                value_block,
            )
            stream.yield_(
                next_maximum,
                next_denominator,
                next_accumulator,
            )
    _, denominator, accumulator = stream.result
    safe_denominator = I.mask(
        denominator,
        valid=denominator > 0.0,
        fill=1.0,
    )
    output[query_axis, :, :] = I.cast(
        accumulator / safe_denominator[:, :, None],
        I.bf16,
    )


@intent.kernel
def paged_mla_decode(
    q_latent: I.In[I.f16, ("B", "HQ", "C")],
    q_rope: I.In[I.f16, ("B", "HQ", "DR")],
    latent_cache: I.In[I.f16, ("P", "PS", "HK", "C")],
    rope_cache: I.In[I.f16, ("P", "PS", "HK", "DR")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    output: I.Out[I.f16, ("B", "HQ", "C")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, C = q_latent.shape
    DR = q_rope.shape[-1]
    _, PS, _, _ = latent_cache.shape
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
            latent_query = q_latent[batch, query_head, :][None, :]
            rope_query = q_rope[batch, query_head, :][None, :]
            page_axis = pages[batch]
            page_stream = I.state_stream(
                page_axis,
                extent=1,
                init=(
                    I.full((1,), -I.inf, dtype=I.f32),
                    I.zeros((1,), dtype=I.f32),
                    I.zeros((1, C), dtype=I.f32),
                ),
                stop=I.end(page_axis),
            )
            with page_stream:
                for page_region, (maximum, denominator, accumulator) in page_stream:
                    physical_page = I.members(page_region)
                    I.assume_in_bounds(physical_page, latent_cache, axis=0)
                    I.assume_in_bounds(physical_page, rope_cache, axis=0)
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
                            I.assume_in_bounds(key_head, latent_cache, axis=2)
                            I.assume_in_bounds(key_head, rope_cache, axis=2)
                            logical_token = I.reshape(
                                page_ordinal[:, None] * PAGE_SIZE
                                + token_index[None, :],
                                (token_region,),
                            )
                            token_valid = logical_token < sequence_length
                            latent_block = I.reshape(
                                latent_cache[
                                    physical_page, token_region, key_head, :
                                ],
                                (token_region, C),
                            )
                            rope_block = I.reshape(
                                rope_cache[
                                    physical_page, token_region, key_head, :
                                ],
                                (token_region, DR),
                            )
                            content_scores = I.contract(
                                latent_query,
                                latent_block,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            position_scores = I.contract(
                                rope_query,
                                rope_block,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            scores = I.mask(
                                (content_scores + position_scores)
                                * (scale * I.LOG2E),
                                valid=token_valid[None, :],
                                fill=-I.inf,
                            )
                            local_maximum = I.reduce.max(
                                scores, axis=1, identity=-I.inf
                            )
                            next_maximum = I.maximum(
                                token_maximum, local_maximum
                            )
                            safe_maximum = I.mask(
                                next_maximum,
                                valid=next_maximum != -I.inf,
                                fill=0.0,
                            )
                            old_scale = I.exp2(
                                token_maximum - safe_maximum
                            )
                            probability = I.exp2(
                                scores - safe_maximum[:, None]
                            )
                            next_denominator = (
                                old_scale * token_denominator
                                + I.reduce.sum(
                                    probability,
                                    axis=1,
                                    identity=0.0,
                                )
                            )
                            next_accumulator = (
                                old_scale[:, None] * token_accumulator
                                + I.contract(
                                    I.cast(probability, I.f16),
                                    latent_block,
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
                (C,),
            )


@intent.kernel
def paged_mla_decode_partials(
    q_latent: I.In[I.f16, ("B", "HQ", "C")],
    q_rope: I.In[I.f16, ("B", "HQ", "DR")],
    latent_cache: I.In[I.f16, ("PT", "HK", "C")],
    rope_cache: I.In[I.f16, ("PT", "HK", "DR")],
    page_offsets: I.In[I.i32, ("B_PLUS_1",)],
    page_indices: I.In[I.i32, ("S",)],
    sequence_lengths: I.In[I.i32, ("B",)],
    split_offsets: I.In[I.i32, ("SP_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "HQ", "SPLITS")],
    partial_output: I.Out[I.f32, ("B", "HQ", "SPLITS", "C")],
    scale: I.f32,
    PAGE_SIZE: I.Constexpr[int],
    HEAD_GROUP: I.Constexpr[int],
    HEAD_TILE: I.Constexpr[int],
    SPLITS: I.Constexpr[int],
    BATCH_SIZE: I.Constexpr[int],
):
    B, HQ, C = q_latent.shape
    DR = q_rope.shape[-1]
    cache_tokens, HK, _ = latent_cache.shape
    split_tokens = I.ragged(
        outer=I.domain(0, BATCH_SIZE * SPLITS),
        members=I.domain(0, cache_tokens),
        offsets=split_offsets,
    )
    local_query_heads = I.domain(0, HEAD_GROUP)
    for job in I.parallel(split_tokens.outer):
        batch = job // SPLITS
        split = job % SPLITS
        I.assume_in_bounds(batch, q_latent, axis=0)
        I.assume_in_bounds(batch, q_rope, axis=0)
        I.assume_in_bounds(batch, page_offsets, axis=0)
        I.assume_in_bounds(batch, sequence_lengths, axis=0)
        page_begin = I.cast(page_offsets[batch], I.index)
        sequence_length = I.cast(sequence_lengths[batch], I.index)
        batch_token_begin = I.cast(split_offsets[batch * SPLITS], I.index)
        for key_head in I.parallel(I.domain(0, HK)):
            I.assume_in_bounds(key_head, latent_cache, axis=1)
            I.assume_in_bounds(key_head, rope_cache, axis=1)
            for query_region in I.parallel(
                I.partition(local_query_heads, extent=HEAD_TILE)
            ):
                query_heads = key_head * HEAD_GROUP + I.indices(query_region)
                I.assume_in_bounds(query_heads, q_latent, axis=1)
                I.assume_in_bounds(query_heads, q_rope, axis=1)
                latent_query = I.gather(
                    q_latent,
                    index=(batch, query_heads, slice(None)),
                )
                rope_query = I.gather(
                    q_rope,
                    index=(batch, query_heads, slice(None)),
                )
                selected_tokens = split_tokens[job]
                stream = I.state_stream(
                    selected_tokens,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((query_region,), -I.inf, dtype=I.f32),
                        I.zeros((query_region,), dtype=I.f32),
                        I.zeros((query_region, C), dtype=I.f32),
                    ),
                    stop=I.end(selected_tokens),
                )
                with stream:
                    for token_region, (
                        maximum,
                        denominator,
                        accumulator,
                    ) in stream:
                        logical_token = I.indices(token_region) - batch_token_begin
                        token_valid = logical_token < sequence_length
                        page_slot = page_begin + logical_token // PAGE_SIZE
                        I.assume_in_bounds(page_slot, page_indices, axis=0)
                        physical_page = I.cast(page_indices[page_slot], I.index)
                        page_token = logical_token % PAGE_SIZE
                        physical_token = physical_page * PAGE_SIZE + page_token
                        I.assume_in_bounds(physical_token, latent_cache, axis=0)
                        I.assume_in_bounds(physical_token, rope_cache, axis=0)
                        latent_block = I.gather(
                            latent_cache,
                            index=(
                                physical_token,
                                key_head,
                                slice(None),
                            ),
                        )
                        rope_block = I.gather(
                            rope_cache,
                            index=(
                                physical_token,
                                key_head,
                                slice(None),
                            ),
                        )
                        scores = I.mask(
                            (
                                I.contract(
                                    latent_query,
                                    latent_block,
                                    reduce=((1, 1),),
                                    acc_dtype=I.f32,
                                )
                                + I.contract(
                                    rope_query,
                                    rope_block,
                                    reduce=((1, 1),),
                                    acc_dtype=I.f32,
                                )
                            )
                            * scale,
                            valid=token_valid[None, :],
                            fill=-I.inf,
                        )
                        local_maximum = I.reduce.max(
                            scores, axis=1, identity=-I.inf
                        )
                        next_maximum = I.maximum(maximum, local_maximum)
                        safe_maximum = I.mask(
                            next_maximum,
                            valid=next_maximum != -I.inf,
                            fill=0.0,
                        )
                        old_scale = I.exp(maximum - safe_maximum)
                        probability = I.exp(scores - safe_maximum[:, None])
                        next_denominator = (
                            old_scale * denominator
                            + I.reduce.sum(
                                probability,
                                axis=1,
                                identity=0.0,
                            )
                        )
                        next_accumulator = (
                            old_scale[:, None] * accumulator
                            + I.contract(
                                I.cast(probability, I.f16),
                                latent_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                maximum, denominator, accumulator = stream.result
                safe_denominator = I.mask(
                    denominator,
                    valid=denominator > 0.0,
                    fill=1.0,
                )
                I.scatter_unique(
                    partial_lse,
                    index=(batch, query_heads, split),
                    value=maximum + I.log(safe_denominator),
                )
                I.scatter_unique(
                    partial_output,
                    index=(batch, query_heads, split, slice(None)),
                    value=accumulator / safe_denominator[:, None],
                )


@intent.kernel
def absorbed_mla_decode(
    q_latent: I.In[I.f16, ("B", "H", "C")],
    q_rope: I.In[I.f16, ("B", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    output: I.Out[I.f16, ("B", "H", "C")],
    scale: I.f32,
):
    B, H, C = q_latent.shape
    K = latent_cache.shape[1]
    DR = q_rope.shape[2]
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            query = I.reshape(q_latent[batch, head, :], (1, C))
            query_position = I.reshape(q_rope[batch, head, :], (1, DR))
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((1,), -I.inf, dtype=I.f32),
                    I.zeros((1,), dtype=I.f32),
                    I.zeros((1, C), dtype=I.f32),
                ),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    latent = latent_cache[batch, key_region, :]
                    scores = (
                        I.contract(
                            query,
                            latent,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        + I.contract(
                            query_position,
                            rope_cache[batch, key_region, :],
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                    ) * (scale * I.LOG2E)
                    local_maximum = I.reduce.max(
                        scores, axis=1, identity=-I.inf
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    next_denominator, next_accumulator = online_attention_accumulate(
                        maximum,
                        next_maximum,
                        denominator,
                        accumulator,
                        scores,
                        latent,
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, head, :] = I.reshape(
                I.cast(accumulator / denominator[:, None], I.f16),
                (C,),
            )


@intent.kernel
def splitk_mla_decode_partials(
    q_latent: I.In[I.f16, ("B", "H", "C")],
    q_rope: I.In[I.f16, ("B", "H", "DR")],
    latent_cache: I.In[I.f16, ("B", "K", "C")],
    rope_cache: I.In[I.f16, ("B", "K", "DR")],
    split_offsets: I.In[I.i32, ("SP_PLUS_1",)],
    partial_lse: I.Out[I.f32, ("B", "H", "SPLITS")],
    partial_output: I.Out[I.f16, ("B", "H", "SPLITS", "C")],
    scale: I.f32,
    SPLITS: I.Constexpr[int],
):
    B, H, C = q_latent.shape
    K = latent_cache.shape[1]
    DR = q_rope.shape[2]
    split_keys = I.ragged(
        outer=I.domain(0, SPLITS),
        members=I.domain(0, K),
        offsets=split_offsets,
    )
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            query = I.reshape(q_latent[batch, head, :], (1, C))
            query_position = I.reshape(q_rope[batch, head, :], (1, DR))
            for split in I.parallel(split_keys.outer):
                stream = I.state_stream(
                    split_keys[split],
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((1,), -I.inf, dtype=I.f32),
                        I.zeros((1,), dtype=I.f32),
                        I.zeros((1, C), dtype=I.f32),
                    ),
                )
                with stream:
                    for key_region, (maximum, denominator, accumulator) in stream:
                        latent = latent_cache[batch, key_region, :]
                        scores = (
                            I.contract(
                                query,
                                latent,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            + I.contract(
                                query_position,
                                rope_cache[batch, key_region, :],
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                        ) * (scale * I.LOG2E)
                        local_maximum = I.reduce.max(
                            scores, axis=1, identity=-I.inf
                        )
                        next_maximum = I.maximum(maximum, local_maximum)
                        next_denominator, next_accumulator = online_attention_accumulate(
                            maximum,
                            next_maximum,
                            denominator,
                            accumulator,
                            scores,
                            latent,
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                maximum, denominator, accumulator = stream.result
                partial_lse[batch, head, split] = maximum[0] + I.log(
                    denominator[0]
                ) * I.LOG2E
                partial_output[batch, head, split, :] = I.cast(
                    I.reshape(
                        accumulator / denominator[:, None],
                        (C,),
                    ),
                    I.f16,
                )
