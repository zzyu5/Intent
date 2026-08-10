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
                        alpha = I.exp2(maximum - next_maximum)
                        probability = I.exp2(scores - next_maximum[:, None])
                        next_denominator = alpha * denominator + I.reduce.sum(
                            probability, axis=1, identity=0.0
                        )
                        low_probability = I.cast(probability, I.f16)
                        next_accumulator = alpha[:, None] * accumulator + I.contract(
                            low_probability,
                            v_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
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
                        alpha = I.exp2(maximum - normalization_maximum)
                        probability = I.exp2(
                            scores - normalization_maximum[:, None]
                        )
                        next_denominator = alpha * denominator + I.reduce.sum(
                            probability, axis=1, identity=0.0
                        )
                        next_accumulator = alpha[:, None] * accumulator + I.contract(
                            I.cast(probability, I.f16),
                            v_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
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
                    alpha = I.exp2(maximum - next_maximum)
                    probability = I.exp2(scores - next_maximum[:, None])
                    next_denominator = alpha * denominator + I.reduce.sum(
                        probability, axis=1, identity=0.0
                    )
                    next_accumulator = alpha[:, None] * accumulator + I.contract(
                        I.cast(probability, I.f16),
                        v_block,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
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
                        alpha = I.exp2(maximum - next_maximum)
                        probability = I.exp2(scores - next_maximum[:, None])
                        next_denominator = alpha * denominator + I.reduce.sum(
                            probability, axis=1, identity=0.0
                        )
                        next_accumulator = alpha[:, None] * accumulator + I.contract(
                            I.cast(probability, I.f16),
                            v_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
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
