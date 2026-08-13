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
    probability = I.exp2(scores - normalization_maximum[:, None])
    next_denominator = alpha * denominator + I.reduce.sum(
        probability,
        axis=1,
        identity=0.0,
    )
    weighted_value = I.cast(probability[:, :, None], I.f32) * I.cast(
        value_block,
        I.f32,
    )
    next_accumulator = alpha[:, None] * accumulator + I.reduce.sum(
        weighted_value,
        axis=1,
        identity=0.0,
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
            for query_region in I.parallel(
                I.partition(query_axis, extent=I.auto("Q_TILE"))
            ):
                latent_query = q_latent[batch, query_region, head, :]
                position_query = q_rope[batch, query_region, head, :]
                stream = I.state_stream(
                    key_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((query_region,), -I.inf, dtype=I.f32),
                        I.zeros((query_region,), dtype=I.f32),
                        I.zeros((query_region, C), dtype=I.f32),
                    ),
                    stop=I.end(query_region),
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
                        valid = I.indices(query_region)[:, None] >= I.indices(
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
                output[batch, query_region, head, :] = I.cast(
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
    for head in I.parallel(I.domain(0, H)):
        for query_region in I.parallel(
            I.partition(query_axis, extent=I.auto("Q_TILE"))
        ):
            latent_query = q_latent[query_region, head, :]
            position_query = q_rope[query_region, head, :]
            stream = I.state_stream(
                selection_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_region,), -I.inf, dtype=I.f32),
                    I.zeros((query_region,), dtype=I.f32),
                    I.zeros((query_region, C), dtype=I.f32),
                ),
            )
            with stream:
                for selection_region, (maximum, denominator, accumulator) in stream:
                    token = selected_tokens[query_region, selection_region]
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
                    content_scores = I.reduce.sum(
                        I.cast(latent_query[:, None, :], I.f32)
                        * I.cast(latent_block, I.f32),
                        axis=2,
                        identity=0.0,
                    )
                    position_scores = I.reduce.sum(
                        I.cast(position_query[:, None, :], I.f32)
                        * I.cast(rope_block, I.f32),
                        axis=2,
                        identity=0.0,
                    )
                    scores = (content_scores + position_scores) * (
                        scale * I.LOG2E
                    )
                    scores = I.mask(
                        scores,
                        valid=valid_token,
                        fill=-I.inf,
                    )
                    local_maximum = I.reduce.max(
                        scores, axis=1, identity=-I.inf
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    normalization_maximum = I.mask(
                        next_maximum,
                        valid=next_maximum != -I.inf,
                        fill=0.0,
                    )
                    next_denominator, next_accumulator = (
                        online_sparse_mla_accumulate(
                            maximum,
                            normalization_maximum,
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
            maximum, denominator, accumulator = stream.result
            safe_denominator = I.mask(
                denominator,
                valid=denominator > 0.0,
                fill=1.0,
            )
            output[query_region, head, :] = I.cast(
                accumulator / safe_denominator[:, None],
                I.bf16,
            )
            maximum_output[query_region, head] = maximum
            lse_output[query_region, head] = (
                maximum + I.log(safe_denominator) * I.LOG2E
            )
