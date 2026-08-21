import intent
import intent.language as I

from kernels.activation.pointwise import tanh_value
from kernels.streaming.attention import online_attention_accumulate
from kernels.streaming.attention import online_attention_accumulate_bf16


SINK_BATCH = 1
SINK_SEQUENCE = 4096
SINK_QUERY_HEADS = 32
SINK_KV_HEADS = 8
SINK_HEAD_DIMENSION = 128
GEMMA_BATCH = 2
GEMMA_SEQUENCE = 4096
GEMMA_QUERY_HEADS = 32
GEMMA_KV_HEADS = 8
GEMMA_HEAD_DIMENSION = 128
GEMMA_WINDOW = 1024
GEMMA_SOFT_CAP = 50.0
SWA_BATCH = 2
SWA_SEQUENCE = 4096
SWA_QUERY_HEADS = 32
SWA_KV_HEADS = 8
SWA_HEAD_DIMENSION = 128
SWA_WINDOW = 1024
BLOCK_CAUSAL_BATCH = 2
BLOCK_CAUSAL_SEQUENCE = 4096
BLOCK_CAUSAL_HEADS = 16
BLOCK_CAUSAL_DIMENSION = 128
BLOCK_CAUSAL_GROUP = 64
NATIVE_SPARSE_FORWARD_BATCH = 2
NATIVE_SPARSE_FORWARD_SEQUENCE = 4096
NATIVE_SPARSE_DECODE_BATCH = 8
NATIVE_SPARSE_DECODE_SEQUENCE = 8192
NATIVE_SPARSE_QUERY_HEADS = 32
NATIVE_SPARSE_KV_HEADS = 4
NATIVE_SPARSE_DIMENSION = 128
NATIVE_SPARSE_SELECTED_BLOCKS = 64
NATIVE_SPARSE_BLOCK = 64


@intent.kernel
def attention_sink_prefill(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    sinks: I.In[I.bf16, ("HQ",)],
    output: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, HQ, Q, D = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = q[batch, query_head, query_axis, :]
            sink_log2 = I.cast(sinks[query_head], I.f32) * I.LOG2E
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_axis,), sink_log2, dtype=I.f32),
                    I.zeros((query_axis,), dtype=I.f32),
                    I.zeros((query_axis, DV), dtype=I.f32),
                ),
                stop=I.end(query_axis),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    scores = I.contract(
                        query,
                        k[batch, key_head, key_region, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    ) * (scale * I.LOG2E)
                    scores = I.mask(
                        scores,
                        valid=I.indices(query_axis)[:, None]
                        >= I.indices(key_region)[None, :],
                        fill=-I.inf,
                    )
                    local_maximum = I.reduce.max(
                        scores, axis=1, identity=-I.inf
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    next_denominator, next_accumulator = online_attention_accumulate_bf16(
                        maximum,
                        next_maximum,
                        denominator,
                        accumulator,
                        scores,
                        v[batch, key_head, key_region, :],
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            maximum, denominator, accumulator = stream.result
            sink_weight = I.exp2(sink_log2 - maximum)
            output[batch, query_head, query_axis, :] = I.cast(
                accumulator / (denominator + sink_weight)[:, None],
                I.bf16,
            )


@intent.kernel
def gemma_gqa_prefill(
    q: I.In[I.bf16, ("B", "HQ", "Q", "D")],
    k: I.In[I.bf16, ("B", "HK", "K", "D")],
    v: I.In[I.bf16, ("B", "HK", "K", "DV")],
    output: I.Out[I.bf16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
    SOFT_CAP: I.Constexpr[float],
):
    B, HQ, Q, D = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = q[batch, query_head, query_axis, :]
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_axis,), -I.inf, dtype=I.f32),
                    I.zeros((query_axis,), dtype=I.f32),
                    I.zeros((query_axis, DV), dtype=I.f32),
                ),
                stop=I.end(query_axis),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    raw_scores = I.contract(
                        query,
                        k[batch, key_head, key_region, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    ) * scale
                    scores = SOFT_CAP * tanh_value(raw_scores / SOFT_CAP)
                    query_index = I.indices(query_axis)
                    key_index = I.indices(key_region)
                    valid = (key_index[None, :] <= query_index[:, None]) and (
                        key_index[None, :] >= query_index[:, None] - WINDOW
                    )
                    scores = I.mask(
                        scores * I.LOG2E,
                        valid=valid,
                        fill=-I.inf,
                    )
                    local_maximum = I.reduce.max(
                        scores, axis=1, identity=-I.inf
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    next_denominator, next_accumulator = online_attention_accumulate_bf16(
                        maximum,
                        next_maximum,
                        denominator,
                        accumulator,
                        scores,
                        v[batch, key_head, key_region, :],
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, query_head, query_axis, :] = I.cast(
                accumulator / denominator[:, None], I.bf16
            )


@intent.kernel
def sliding_window_gqa_prefill(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "DV")],
    output: I.Out[I.f16, ("B", "HQ", "Q", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    WINDOW: I.Constexpr[int],
):
    B, HQ, Q, D = q.shape
    K = k.shape[2]
    DV = v.shape[3]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query = q[batch, query_head, query_axis, :]
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_axis,), -I.inf, dtype=I.f32),
                    I.zeros((query_axis,), dtype=I.f32),
                    I.zeros((query_axis, DV), dtype=I.f32),
                ),
                stop=I.end(query_axis),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    scores = I.contract(
                        query,
                        k[batch, key_head, key_region, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    ) * (scale * I.LOG2E)
                    query_index = I.indices(query_axis)
                    key_index = I.indices(key_region)
                    valid = (key_index[None, :] <= query_index[:, None]) and (
                        key_index[None, :] > query_index[:, None] - WINDOW
                    )
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
                        v[batch, key_head, key_region, :],
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, query_head, query_axis, :] = I.cast(
                accumulator / I.maximum(denominator, 1e-6)[:, None], I.f16
            )


@intent.kernel
def block_causal_attention_fwd(
    q: I.In[I.f16, ("B", "Q", "H", "D")],
    k: I.In[I.f16, ("B", "K", "H", "D")],
    v: I.In[I.f16, ("B", "K", "H", "DV")],
    output: I.Out[I.f16, ("B", "Q", "H", "DV")],
    scale: I.f32,
    BLOCK: I.Constexpr[int],
):
    B, Q, H, D = q.shape
    K = k.shape[1]
    DV = v.shape[3]
    half = Q // 2
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            query = q[batch, query_axis, head, :]
            stream = I.state_stream(
                key_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((query_axis,), -I.inf, dtype=I.f32),
                    I.zeros((query_axis,), dtype=I.f32),
                    I.zeros((query_axis, DV), dtype=I.f32),
                ),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    scores = I.contract(
                        query,
                        k[batch, key_region, head, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    ) * (scale * I.LOG2E)
                    query_index = I.indices(query_axis)
                    key_index = I.indices(key_region)
                    query_clean = query_index >= half
                    key_clean = key_index >= half
                    query_local = I.mask(
                        query_index - half,
                        valid=query_clean,
                        fill=query_index,
                    )
                    key_local = I.mask(
                        key_index - half,
                        valid=key_clean,
                        fill=key_index,
                    )
                    query_block = query_local // BLOCK
                    key_block = key_local // BLOCK
                    noisy_diagonal = (
                        (query_block[:, None] == key_block[None, :])
                        and (query_clean[:, None] == False)
                        and (key_clean[None, :] == False)
                    )
                    offset_causal = (
                        (query_block[:, None] > key_block[None, :])
                        and (query_clean[:, None] == False)
                        and key_clean[None, :]
                    )
                    clean_causal = (
                        (query_block[:, None] >= key_block[None, :])
                        and query_clean[:, None]
                        and key_clean[None, :]
                    )
                    scores = I.mask(
                        scores,
                        valid=noisy_diagonal or offset_causal or clean_causal,
                        fill=-I.inf,
                    )
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
                        v[batch, key_region, head, :],
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, query_axis, head, :] = I.cast(
                accumulator / denominator[:, None], I.f16
            )


@intent.kernel
def native_sparse_attention_fwd(
    q: I.In[I.f16, ("B", "Q", "HQ", "D")],
    k: I.In[I.f16, ("B", "K", "HK", "D")],
    v: I.In[I.f16, ("B", "K", "HK", "DV")],
    selected_blocks: I.In[I.i32, ("B", "Q", "HK", "S")],
    output: I.Out[I.f16, ("B", "Q", "HQ", "DV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    BLOCK: I.Constexpr[int],
):
    B, Q, HQ, D = q.shape
    K = k.shape[1]
    DV = v.shape[3]
    slots = selected_blocks.shape[3]
    block_members = I.domain(0, BLOCK)
    for batch in I.parallel(I.domain(0, B)):
        for query_index in I.parallel(I.domain(0, Q)):
            for query_head in I.parallel(I.domain(0, HQ)):
                key_head = query_head // HEAD_GROUP
                query = I.reshape(q[batch, query_index, query_head, :], (1, D))
                stream = I.state_stream(
                    I.domain(0, slots),
                    extent=1,
                    init=(
                        I.full((1,), -I.inf, dtype=I.f32),
                        I.zeros((1,), dtype=I.f32),
                        I.zeros((1, DV), dtype=I.f32),
                    ),
                )
                with stream:
                    for slot_region, (maximum, denominator, accumulator) in stream:
                        block_id = selected_blocks[
                            batch,
                            query_index,
                            key_head,
                            I.indices(slot_region),
                        ]
                        key_indices = (
                            I.cast(block_id, I.index)[:, None] * BLOCK
                            + I.indices(block_members)[None, :]
                        )
                        I.assume_in_bounds(key_indices, k, axis=1)
                        keys = I.gather(
                            k,
                            index=(
                                batch,
                                key_indices,
                                key_head,
                                slice(None),
                            ),
                        )
                        values = I.gather(
                            v,
                            index=(
                                batch,
                                key_indices,
                                key_head,
                                slice(None),
                            ),
                        )
                        keys = I.reshape(keys, (BLOCK, D))
                        values = I.reshape(values, (BLOCK, DV))
                        scores = I.contract(
                            query,
                            keys,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        ) * (scale * I.LOG2E)
                        absolute_keys = I.reshape(key_indices, (BLOCK,))
                        scores = I.mask(
                            scores,
                            valid=absolute_keys[None, :] <= query_index,
                            fill=-I.inf,
                        )
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
                            values,
                        )
                        stream.yield_(
                            next_maximum,
                            next_denominator,
                            next_accumulator,
                        )
                _, denominator, accumulator = stream.result
                output[batch, query_index, query_head, :] = I.reshape(
                    I.cast(accumulator / denominator[:, None], I.f16),
                    (DV,),
                )


@intent.kernel
def varlen_block_causal_attention_fwd(
    q: I.In[I.f16, ("U", "H", "D")],
    k: I.In[I.f16, ("U", "H", "D")],
    v: I.In[I.f16, ("U", "H", "DV")],
    sequence_offsets: I.In[I.i32, ("B_PLUS_1",)],
    output: I.Out[I.f16, ("U", "H", "DV")],
    scale: I.f32,
    BLOCK: I.Constexpr[int],
):
    U, H, D = q.shape
    B = sequence_offsets.shape[0] - 1
    DV = v.shape[2]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=sequence_offsets,
    )
    for sequence in I.parallel(sequences.outer):
        members = sequences[sequence]
        sequence_start = I.indices(members)[0]
        half = (I.end(members) - sequence_start) // 2
        for head in I.parallel(I.domain(0, H)):
            query = q[members, head, :]
            stream = I.state_stream(
                members,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((members,), -I.inf, dtype=I.f32),
                    I.zeros((members,), dtype=I.f32),
                    I.zeros((members, DV), dtype=I.f32),
                ),
            )
            with stream:
                for key_region, (maximum, denominator, accumulator) in stream:
                    scores = I.contract(
                        query,
                        k[key_region, head, :],
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    ) * (scale * I.LOG2E)
                    query_index = I.indices(members) - sequence_start
                    key_index = I.indices(key_region) - sequence_start
                    query_clean = query_index >= half
                    key_clean = key_index >= half
                    query_local = I.mask(
                        query_index - half,
                        valid=query_clean,
                        fill=query_index,
                    )
                    key_local = I.mask(
                        key_index - half,
                        valid=key_clean,
                        fill=key_index,
                    )
                    query_block = query_local // BLOCK
                    key_block = key_local // BLOCK
                    valid = (
                        (query_block[:, None] == key_block[None, :])
                        and (query_clean[:, None] == False)
                        and (key_clean[None, :] == False)
                    ) or (
                        (query_block[:, None] > key_block[None, :])
                        and (query_clean[:, None] == False)
                        and key_clean[None, :]
                    ) or (
                        (query_block[:, None] >= key_block[None, :])
                        and query_clean[:, None]
                        and key_clean[None, :]
                    )
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
                        v[key_region, head, :],
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[members, head, :] = I.cast(
                accumulator / denominator[:, None], I.f16
            )
