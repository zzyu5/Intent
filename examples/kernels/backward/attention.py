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
            for query_region in I.parallel(
                I.partition(query_axis, extent=I.auto("Q_TILE"))
            ):
                output_block = I.cast(output[batch, query_head, query_region, :], I.f32)
                grad_output_block = I.cast(
                    grad_output[batch, query_head, query_region, :], I.f32
                )
                delta[batch, query_head, query_region] = I.reduce.sum(
                    output_block * grad_output_block,
                    axis=1,
                    identity=0.0,
                )


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
            for key_region in I.parallel(
                I.partition(key_axis, extent=I.auto("K_TILE"))
            ):
                key_block = k[batch, key_head, key_region, :]
                value_block = v[batch, key_head, key_region, :]
                accumulation = I.state_stream(
                    query_axis,
                    extent=I.auto("Q_TILE"),
                    init=(
                        I.zeros((key_region, D), dtype=I.f32),
                        I.zeros((key_region, D), dtype=I.f32),
                    ),
                    stop=I.end(query_axis),
                )
                with accumulation:
                    for query_region, (grad_k_value, grad_v_value) in accumulation:
                        q_index = I.indices(query_region)
                        k_index = I.indices(key_region)
                        next_grad_k = grad_k_value
                        next_grad_v = grad_v_value
                        for query_head_offset in range(HEAD_GROUP):
                            query_head = key_head * HEAD_GROUP + query_head_offset
                            query_block = q[
                                batch, query_head, query_region, :
                            ]
                            grad_output_block = grad_output[
                                batch, query_head, query_region, :
                            ]
                            scores = I.contract(
                                key_block,
                                query_block,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            probability = I.exp2(
                                scores * (scale * I.LOG2E)
                                - lse[batch, query_head, query_region][None, :]
                                * I.LOG2E
                            )
                            if CAUSAL:
                                probability = I.mask(
                                    probability,
                                    valid=k_index[:, None] <= q_index[None, :],
                                    fill=0.0,
                                )
                            next_grad_v = next_grad_v + I.contract(
                                I.cast(probability, I.f16),
                                grad_output_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                            grad_probability = I.contract(
                                value_block,
                                grad_output_block,
                                reduce=((1, 1),),
                                acc_dtype=I.f32,
                            )
                            grad_scores = probability * (
                                grad_probability
                                - delta[
                                    batch, query_head, query_region
                                ][None, :]
                            )
                            next_grad_k = next_grad_k + I.contract(
                                I.cast(grad_scores, I.f16),
                                query_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                        accumulation.yield_(next_grad_k, next_grad_v)
                grad_k_value, grad_v_value = accumulation.result
                grad_k[batch, key_head, key_region, :] = I.cast(
                    grad_k_value * scale, I.f16
                )
                grad_v[batch, key_head, key_region, :] = I.cast(
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
            for query_region in I.parallel(
                I.partition(query_axis, extent=I.auto("Q_TILE"))
            ):
                query_block = q[batch, query_head, query_region, :]
                grad_output_block = grad_output[
                    batch, query_head, query_region, :
                ]
                query_lse = lse[batch, query_head, query_region][:, None]
                query_delta = delta[batch, query_head, query_region][:, None]
                accumulation = I.state_stream(
                    key_axis,
                    extent=I.auto("K_TILE"),
                    init=(I.zeros((query_region, D), dtype=I.f32),),
                    stop=I.end(query_region) if CAUSAL else I.end(key_axis),
                )
                with accumulation:
                    for key_region, grad_q_value in accumulation:
                        key_block = k[batch, key_head, key_region, :]
                        value_block = v[batch, key_head, key_region, :]
                        scores = I.contract(
                            query_block,
                            key_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        probability = I.exp2(
                            scores * (scale * I.LOG2E) - query_lse * I.LOG2E
                        )
                        if CAUSAL:
                            q_index = I.indices(query_region)
                            k_index = I.indices(key_region)
                            probability = I.mask(
                                probability,
                                valid=q_index[:, None] >= k_index[None, :],
                                fill=0.0,
                            )
                        grad_probability = I.contract(
                            grad_output_block,
                            value_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )
                        grad_scores = probability * (
                            grad_probability - query_delta
                        )
                        accumulation.yield_(
                            grad_q_value
                            + I.contract(
                                I.cast(grad_scores, I.f16),
                                key_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                        )
                grad_q[batch, query_head, query_region, :] = I.cast(
                    accumulation.result * scale, I.f16
                )
