import math

import intent
import intent.language as I


BATCH = 2
QUERY_HEADS = 8
KV_HEADS = 2
HEAD_GROUP = QUERY_HEADS // KV_HEADS
SEQUENCE = 1024
HEAD_DIMENSION = 64
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


@intent.kernel
def attention_backward_delta(
    output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    delta: I.Out[I.f32, ("B", "HQ", "Q")],
):
    B, HQ, Q, _ = output.shape
    query_axis = I.domain(0, Q)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            output_block = I.cast(output[batch, query_head, query_axis, :], I.f32)
            grad_output_block = I.cast(
                grad_output[batch, query_head, query_axis, :], I.f32
            )
            delta[batch, query_head, query_axis] = I.reduce.sum(
                output_block * grad_output_block,
                axis=1,
            )


@intent.fn
def summarize_key_value_gradients(
    queries, grad_outputs, lse, delta, query_coordinates,
    keys, values, key_coordinates, scale, causal,
):
    scores = I.matmul(keys, queries, transpose_rhs=True, acc_dtype=I.f32)
    probability = I.exp2(
        scores * (scale * I.LOG2E) - lse[None, :] * I.LOG2E,
        approximate=True, flush_to_zero=True,
    )
    if causal:
        probability = I.mask(
            probability,
            valid=key_coordinates[:, None] <= query_coordinates[None, :],
            fill=0.0,
        )
    grad_values = I.matmul(
        I.cast(probability, I.f16), grad_outputs, acc_dtype=I.f32
    )
    grad_probability = I.matmul(
        values, grad_outputs, transpose_rhs=True, acc_dtype=I.f32
    )
    grad_scores = probability * (grad_probability - delta[None, :])
    grad_keys = I.matmul(
        I.cast(grad_scores, I.f16), queries, acc_dtype=I.f32
    )
    return I.record(grad_k=grad_keys, grad_v=grad_values)


@intent.fn
def merge_key_value_gradients(lhs, rhs):
    return I.record(grad_k=lhs.grad_k + rhs.grad_k, grad_v=lhs.grad_v + rhs.grad_v)


@intent.kernel
def attention_backward_dkdv(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    lse: I.In[I.f32, ("B", "HQ", "Q")],
    delta: I.In[I.f32, ("B", "HQ", "Q")],
    grad_k: I.Out[I.f16, ("B", "HK", "K", "D")],
    grad_v: I.Out[I.f16, ("B", "HK", "K", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, D = q.shape
    _, HK, K, _ = k.shape
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for key_head in I.parallel(I.domain(0, HK)):
            key_block = k[batch, key_head, key_axis, :]
            value_block = v[batch, key_head, key_axis, :]
            q_index = I.indices(query_axis)
            k_index = I.indices(key_axis)
            grad_k_value = I.zeros((K, D), dtype=I.f32)
            grad_v_value = I.zeros((K, D), dtype=I.f32)
            for query_head_offset in range(HEAD_GROUP):
                query_head = key_head * HEAD_GROUP + query_head_offset
                query_block = q[batch, query_head, query_axis, :]
                grad_output_block = grad_output[
                    batch, query_head, query_axis, :
                ]
                summary = I.region_fold(
                    source=(
                        query_block, grad_output_block,
                        lse[batch, query_head, query_axis],
                        delta[batch, query_head, query_axis], q_index,
                    ),
                    axis=0,
                    summarize=summarize_key_value_gradients,
                    combine=merge_key_value_gradients,
                    identity=I.record(
                        grad_k=I.zeros((K, D), dtype=I.f32),
                        grad_v=I.zeros((K, D), dtype=I.f32),
                    ),
                    operands=(key_block, value_block, k_index, scale, CAUSAL),
                )
                grad_k_value = grad_k_value + summary.grad_k
                grad_v_value = grad_v_value + summary.grad_v
            grad_k[batch, key_head, key_axis, :] = I.cast(
                grad_k_value * scale, I.f16
            )
            grad_v[batch, key_head, key_axis, :] = I.cast(
                grad_v_value, I.f16
            )


@intent.kernel
def attention_backward_dq(
    q: I.In[I.f16, ("B", "HQ", "Q", "D")],
    k: I.In[I.f16, ("B", "HK", "K", "D")],
    v: I.In[I.f16, ("B", "HK", "K", "D")],
    grad_output: I.In[I.f16, ("B", "HQ", "Q", "D")],
    lse: I.In[I.f32, ("B", "HQ", "Q")],
    delta: I.In[I.f32, ("B", "HQ", "Q")],
    grad_q: I.Out[I.f16, ("B", "HQ", "Q", "D")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
    CAUSAL: I.Constexpr[bool],
):
    B, HQ, Q, D = q.shape
    K = k.shape[2]
    query_axis = I.domain(0, Q)
    key_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for query_head in I.parallel(I.domain(0, HQ)):
            key_head = query_head // HEAD_GROUP
            query_block = q[batch, query_head, query_axis, :]
            grad_output_block = grad_output[
                batch, query_head, query_axis, :
            ]
            query_lse = lse[batch, query_head, query_axis][:, None]
            query_delta = delta[batch, query_head, query_axis][:, None]
            key_block = k[batch, key_head, key_axis, :]
            value_block = v[batch, key_head, key_axis, :]
            scores = I.matmul(
                query_block,
                key_block,
                transpose_rhs=True,
                acc_dtype=I.f32,
            )
            probability = I.exp2(
                scores * (scale * I.LOG2E) - query_lse * I.LOG2E,
                approximate=True, flush_to_zero=True,
            )
            if CAUSAL:
                q_index = I.indices(query_axis)
                k_index = I.indices(key_axis)
                probability = I.mask(
                    probability,
                    valid=q_index[:, None] >= k_index[None, :],
                    fill=0.0,
                )
            grad_probability = I.matmul(
                grad_output_block,
                value_block,
                transpose_rhs=True,
                acc_dtype=I.f32,
            )
            grad_scores = probability * (grad_probability - query_delta)
            grad_q_value = I.matmul(
                I.cast(grad_scores, I.f16),
                key_block,
                acc_dtype=I.f32,
            )
            grad_q[batch, query_head, query_axis, :] = I.cast(
                grad_q_value * scale, I.f16
            )
