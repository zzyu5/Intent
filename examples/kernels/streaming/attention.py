import math

import intent
import intent.language as I


BATCH = 4
HEADS = 32
SEQUENCE = 4096
HEAD_DIMENSION = 128
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)
VARLEN_BATCH = 8
VARLEN_TOTAL_TOKENS = 29114
VARLEN_GQA_QUERY_HEADS = 32
VARLEN_GQA_KV_HEADS = 8
VARLEN_GQA_HEAD_GROUP = VARLEN_GQA_QUERY_HEADS // VARLEN_GQA_KV_HEADS
VARLEN_GQA_TOTAL_TOKENS = 29184
GQA_DECODE_BATCH = 32
GQA_DECODE_QUERY_HEADS = 32
GQA_DECODE_KV_HEADS = 8
GQA_DECODE_SEQUENCE = 8192
GQA_DECODE_HEAD_DIMENSION = 128
GQA_DECODE_HEAD_GROUP = GQA_DECODE_QUERY_HEADS // GQA_DECODE_KV_HEADS
GQA_DECODE_SCALE = 1.0 / math.sqrt(GQA_DECODE_HEAD_DIMENSION)
MLA_PREFILL_BATCH = 1
MLA_PREFILL_QUERY_HEADS = 8
MLA_PREFILL_KV_HEADS = 2
MLA_PREFILL_SEQUENCE = 1024
MLA_PREFILL_CONTENT_DIMENSION = 128
MLA_PREFILL_POSITION_DIMENSION = 64
MLA_PREFILL_HEAD_GROUP = MLA_PREFILL_QUERY_HEADS // MLA_PREFILL_KV_HEADS
MLA_PREFILL_SCALE = 1.0 / math.sqrt(
    MLA_PREFILL_CONTENT_DIMENSION + MLA_PREFILL_POSITION_DIMENSION
)


@intent.fn
def online_attention_accumulate(
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
        probability, axis=1, identity=0.0
    )
    next_accumulator = alpha[:, None] * accumulator + I.contract(
        I.cast(probability, I.f16),
        value_block,
        reduce=((1, 0),),
        acc_dtype=I.f32,
    )
    return next_denominator, next_accumulator


@intent.kernel
def flash_attention_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, D = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for q_region in I.parallel(
                I.partition(q_axis, extent=I.auto("Q_TILE"))
            ):
                q_block = q[batch, head, q_region, :]
                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((q_region,), -I.inf, dtype=I.f32),
                        I.zeros((q_region,), dtype=I.f32),
                        I.zeros((q_region, DV), dtype=I.f32),
                    ),
                    stop=I.end(q_region) if CAUSAL else I.end(k_axis),
                )
                with stream:
                    for k_region, (maximum, denominator, accumulator) in stream:
                        k_block = k[batch, head, k_region, :]
                        v_block = v[batch, head, k_region, :]
                        scores = I.contract(
                            q_block,
                            k_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        scores = scores * (scale * I.LOG2E)
                        if CAUSAL:
                            q_index = I.indices(q_region)
                            k_index = I.indices(k_region)
                            valid = q_index[:, None] >= k_index[None, :]
                            scores = I.mask(scores, valid=valid, fill=-I.inf)
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
                            v_block,
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                _, denominator, accumulator = stream.result
                output[batch, head, q_region, :] = I.cast(
                    accumulator / denominator[:, None], I.f16
                )


@intent.kernel
def continuous_gqa_decode(
    q: I.In[I.f16, ("B", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "DV")],
    valid_mask: I.In[I.u8, ("B", "K", "HK")],
    output: I.Out[I.f16, ("B", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, D = q.shape
    K = k.shape[1]
    DV = v.shape[-1]
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = q[batch, query_head, :][None, :]
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((1,), -I.inf, dtype=I.f32),
                    I.zeros((1,), dtype=I.f32),
                    I.zeros((1, DV), dtype=I.f32),
                ),
                stop=I.end(key_axis),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    key = k[batch, key_region, key_head, :]
                    value = v[batch, key_region, key_head, :]
                    mask_values = valid_mask[batch, key_region, key_head]
                    mask_values = I.mask(
                        mask_values,
                        valid=I.indices(key_region) < K,
                        fill=I.cast(0, I.u8),
                    )
                    mask = mask_values != 0
                    scores = I.contract(
                        query,
                        key,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    scores = I.mask(
                        scores * (scale * I.LOG2E),
                        valid=mask[None, :],
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
                    next_denominator, next_accumulator = online_attention_accumulate(
                        maximum,
                        normalization_maximum,
                        denominator,
                        accumulator,
                        scores,
                        value,
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            safe_denominator = I.mask(
                denominator, valid=denominator > 0.0, fill=1.0
            )
            output[batch, query_head, :] = I.reshape(
                I.cast(accumulator / safe_denominator[:, None], I.f16),
                (DV,),
            )


@intent.kernel
def mla_prefill(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    qpe: I.In[I.f16, ("B", "H", "Q", "P")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    kpe: I.In[I.f16, ("B", 1, "K", "P")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    output: I.Out[I.f16, ("B", "H", "Q", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, H, Q, D = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, H)):
            key_head = query_head // HEAD_GROUP
            for query_region in I.parallel(
                I.partition(query_axis, extent=I.auto("Q_TILE"))
            ):
                query = q[batch, query_head, query_region, :]
                query_position = qpe[batch, query_head, query_region, :]
                stream = I.state_stream(
                    key_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((query_region,), -I.inf, dtype=I.f32),
                        I.full((query_region,), 1.0, dtype=I.f32),
                        I.zeros((query_region, D), dtype=I.f32),
                    ),
                    stop=I.end(query_region),
                )
                with stream:
                    for key_region, (maximum, denominator, accumulator) in stream:
                        content_scores = I.contract(
                            query,
                            k[batch, key_head, key_region, :],
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        position_scores = I.contract(
                            query_position,
                            kpe[batch, 0, key_region, :],
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
                                v[batch, key_head, key_region, :],
                            )
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                _, denominator, accumulator = stream.result
                output[batch, query_head, query_region, :] = I.cast(
                    accumulator / denominator[:, None], I.f16
                )


@intent.kernel
def flash_attention_bias_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    bias: I.In[I.f32, ("B", "H", "K")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
):
    B, H, Q, _ = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for q_region in I.parallel(
                I.partition(q_axis, extent=I.auto("Q_TILE"))
            ):
                q_block = q[batch, head, q_region, :]
                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((q_region,), -I.inf, dtype=I.f32),
                        I.zeros((q_region,), dtype=I.f32),
                        I.zeros((q_region, DV), dtype=I.f32),
                    ),
                    stop=I.end(k_axis),
                )
                with stream:
                    for k_region, (maximum, denominator, accumulator) in stream:
                        k_block = k[batch, head, k_region, :]
                        v_block = v[batch, head, k_region, :]
                        bias_block = bias[batch, head, k_region]
                        scores = I.contract(
                            q_block,
                            k_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        scores = (
                            scores * scale + bias_block[None, :]
                        ) * I.LOG2E
                        local_maximum = I.reduce.max(
                            scores, axis=1, identity=-I.inf
                        )
                        next_maximum = I.maximum(maximum, local_maximum)
                        normalization_maximum = I.mask(
                            next_maximum,
                            valid=next_maximum != -I.inf,
                            fill=0.0,
                        )
                        next_denominator, next_accumulator = online_attention_accumulate(
                            maximum,
                            normalization_maximum,
                            denominator,
                            accumulator,
                            scores,
                            v_block,
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
                output[batch, head, q_region, :] = I.cast(
                    accumulator / safe_denominator[:, None], I.f16
                )


@intent.kernel
def flash_varlen_attention_fwd(
    q: I.In[I.f16, ("U", "D")],
    k: I.In[I.f16, ("U", "D")],
    v: I.In[I.f16, ("U", "DV")],
    sequence_lengths: I.In[I.i32, ("B",)],
    cu_seqlens: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    U, _ = q.shape
    B = sequence_lengths.shape[0]
    DV = v.shape[-1]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=cu_seqlens,
    )
    for sequence in I.parallel(sequences.outer):
        for q_region in I.parallel(
            I.partition(sequences[sequence], extent=I.auto("Q_TILE"))
        ):
            q_block = q[q_region, :]
            k_axis = sequences[sequence]
            stream = I.state_stream(
                k_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((q_region,), -I.inf, dtype=I.f32),
                    I.zeros((q_region,), dtype=I.f32),
                    I.zeros((q_region, DV), dtype=I.f32),
                ),
                stop=I.end(q_region) if CAUSAL else I.end(k_axis),
            )
            with stream:
                for k_region, (maximum, denominator, accumulator) in stream:
                    k_block = k[k_region, :]
                    v_block = v[k_region, :]
                    scores = I.contract(
                        q_block,
                        k_block,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    scores = scores * (scale * I.LOG2E)
                    if CAUSAL:
                        q_index = I.indices(q_region)
                        k_index = I.indices(k_region)
                        valid = q_index[:, None] >= k_index[None, :]
                        scores = I.mask(scores, valid=valid, fill=-I.inf)
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
                        v_block,
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[q_region, :] = I.cast(
                accumulator / denominator[:, None], I.f16
            )


@intent.kernel
def flash_varlen_gqa_prefill(
    q: I.In[I.f16, ("U", "HQ", "D")],
    k: I.In[I.f16, ("U", "HK", "D")],
    v: I.In[I.f16, ("U", "HK", "DV")],
    sequence_lengths: I.In[I.i32, ("B",)],
    cu_seqlens: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    U, HQ, _ = q.shape
    B = sequence_lengths.shape[0]
    DV = v.shape[-1]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=cu_seqlens,
    )
    for sequence in I.parallel(sequences.outer):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            for q_region in I.parallel(
                I.partition(sequences[sequence], extent=I.auto("Q_TILE"))
            ):
                q_block = q[q_region, query_head, :]
                k_axis = sequences[sequence]
                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((q_region,), -I.inf, dtype=I.f32),
                        I.zeros((q_region,), dtype=I.f32),
                        I.zeros((q_region, DV), dtype=I.f32),
                    ),
                    stop=I.end(q_region),
                )
                with stream:
                    for k_region, (maximum, denominator, accumulator) in stream:
                        k_block = k[k_region, key_head, :]
                        v_block = v[k_region, key_head, :]
                        scores = I.contract(
                            q_block,
                            k_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        scores = scores * (scale * I.LOG2E)
                        q_index = I.indices(q_region)
                        k_index = I.indices(k_region)
                        valid = q_index[:, None] >= k_index[None, :]
                        scores = I.mask(scores, valid=valid, fill=-I.inf)
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
                            v_block,
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                _, denominator, accumulator = stream.result
                output[q_region, query_head, :] = I.cast(
                    accumulator / denominator[:, None], I.f16
                )
