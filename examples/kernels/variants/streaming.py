import math

import intent
import intent.language as I

from kernels.streaming.attention import online_attention_accumulate


@intent.kernel
def streamed_online_softmax_inline(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        statistics = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(I.cast(-I.inf, I.f32), I.cast(0.0, I.f32)),
        )
        with statistics:
            for region, (maximum, denominator) in statistics:
                values = x[row, region]
                local_maximum = I.reduce.max(values, axis=0, identity=-I.inf)
                next_maximum = I.maximum(maximum, local_maximum)
                statistics.yield_(
                    next_maximum,
                    I.exp(maximum - next_maximum) * denominator
                    + I.reduce.sum(
                        I.exp(values - next_maximum),
                        axis=0,
                        identity=0.0,
                    ),
                )
        maximum, denominator = statistics.result
        output = I.state_stream(
            columns,
            extent=I.auto("N_TILE"),
            init=(maximum, denominator),
        )
        with output:
            for region, (final_maximum, final_denominator) in output:
                y[row, region] = (
                    I.exp(x[row, region] - final_maximum) / final_denominator
                )
                output.yield_(final_maximum, final_denominator)


@intent.kernel
def flash_attention_inline_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            q_block = q[batch, head, q_axis, :]
            stream = I.state_stream(
                k_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((q_axis,), -I.inf, dtype=I.f32),
                    I.zeros((q_axis,), dtype=I.f32),
                    I.zeros((q_axis, DV), dtype=I.f32),
                ),
                stop=I.end(q_axis) if CAUSAL else I.end(k_axis),
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
                    ) * (scale * I.LOG2E)
                    if CAUSAL:
                        q_index = I.indices(q_axis)
                        k_index = I.indices(k_region)
                        scores = I.mask(
                            scores,
                            valid=q_index[:, None] >= k_index[None, :],
                            fill=-I.inf,
                        )
                    local_maximum = I.reduce.max(
                        scores,
                        axis=1,
                        identity=-I.inf,
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    alpha = I.exp2(maximum - next_maximum)
                    probability = I.exp2(scores - next_maximum[:, None])
                    next_denominator = alpha * denominator + I.reduce.sum(
                        probability,
                        axis=1,
                        identity=0.0,
                    )
                    next_accumulator = (
                        alpha[:, None] * accumulator
                        + I.contract(
                            I.cast(probability, I.f16),
                            v_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        )
                    )
                    stream.yield_(
                        next_maximum,
                        next_denominator,
                        next_accumulator,
                    )
            _, denominator, accumulator = stream.result
            output[batch, head, q_axis, :] = I.cast(
                accumulator / denominator[:, None],
                I.f16,
            )


@intent.kernel
def flash_attention_select_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            q_block = q[batch, head, q_axis, :]
            stream = I.state_stream(
                k_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((q_axis,), -I.inf, dtype=I.f32),
                    I.zeros((q_axis,), dtype=I.f32),
                    I.zeros((q_axis, DV), dtype=I.f32),
                ),
                stop=I.end(q_axis) if CAUSAL else I.end(k_axis),
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
                    ) * (scale * I.LOG2E)
                    if CAUSAL:
                        q_index = I.indices(q_axis)
                        k_index = I.indices(k_region)
                        valid = q_index[:, None] >= k_index[None, :]
                        scores = (
                            scores
                            if valid
                            else I.full(
                                (q_axis, k_region),
                                -I.inf,
                                dtype=I.f32,
                            )
                        )
                    local_maximum = I.reduce.max(
                        scores,
                        axis=1,
                        identity=-I.inf,
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
            output[batch, head, q_axis, :] = I.cast(
                accumulator / denominator[:, None],
                I.f16,
            )


@intent.kernel
def flash_attention_full_causal_stream_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    output: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, _ = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]
    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            q_block = q[batch, head, q_axis, :]
            stream = I.state_stream(
                k_axis,
                extent=I.auto("K_TILE"),
                init=(
                    I.full((q_axis,), -I.inf, dtype=I.f32),
                    I.zeros((q_axis,), dtype=I.f32),
                    I.zeros((q_axis, DV), dtype=I.f32),
                ),
                stop=I.end(k_axis),
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
                    ) * (scale * I.LOG2E)
                    if CAUSAL:
                        q_index = I.indices(q_axis)
                        k_index = I.indices(k_region)
                        scores = I.mask(
                            scores,
                            valid=q_index[:, None] >= k_index[None, :],
                            fill=-I.inf,
                        )
                    local_maximum = I.reduce.max(
                        scores,
                        axis=1,
                        identity=-I.inf,
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
            output[batch, head, q_axis, :] = I.cast(
                accumulator / denominator[:, None],
                I.f16,
            )
