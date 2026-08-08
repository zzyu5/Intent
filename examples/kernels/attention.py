import math

import intent
import intent.language as I


BATCH = 4
HEADS = 32
SEQUENCE = 4096
HEAD_DIMENSION = 128
SCALE = 1.0 / math.sqrt(HEAD_DIMENSION)


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
                            if not I.any(valid):
                                stream.yield_(maximum, denominator, accumulator)
                                continue
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
