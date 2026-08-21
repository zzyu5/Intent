import math

import intent
import intent.language as I


BATCH = 2
SEQUENCE = 2048
HEADS = 8
KEY_DIMENSION = 128
VALUE_DIMENSION = 128
SCALE = 1.0 / math.sqrt(KEY_DIMENSION)


@intent.kernel
def recurrent_gated_delta_fwd(
    query: I.In[I.bf16, ("B", "T", "H", "K")],
    key: I.In[I.bf16, ("B", "T", "H", "K")],
    value: I.In[I.bf16, ("B", "T", "HV", "V")],
    gate: I.In[I.bf16, ("B", "T", "HV")],
    beta: I.In[I.bf16, ("B", "T", "HV")],
    output: I.Out[I.bf16, ("B", "T", "HV", "V")],
    final_state: I.Out[I.f32, ("B", "HV", "K", "V")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, T, _, K = query.shape
    HV = value.shape[2]
    V = value.shape[3]
    positions = I.domain(0, T)
    key_features = I.domain(0, K)
    value_features = I.domain(0, V)
    for batch in I.parallel(I.domain(0, B)):
        for value_head in I.parallel(I.domain(0, HV)):
            query_head = value_head // HEAD_GROUP
            stream = I.state_stream(
                positions,
                extent=1,
                init=(I.zeros((K, value_features), dtype=I.f32),),
            )
            with stream:
                for position_region, state in stream:
                    position = I.indices(position_region)
                    query_vector = I.reshape(
                        I.cast(
                            query[
                                batch,
                                position_region,
                                query_head,
                                key_features,
                            ],
                            I.f32,
                        ),
                        (K,),
                    ) * scale
                    key_vector = I.reshape(
                        I.cast(
                            key[
                                batch,
                                position_region,
                                query_head,
                                key_features,
                            ],
                            I.f32,
                        ),
                        (K,),
                    )
                    value_vector = I.reshape(
                        I.cast(
                            value[
                                batch,
                                position_region,
                                value_head,
                                value_features,
                            ],
                            I.f32,
                        ),
                        (value_features,),
                    )
                    decay = I.exp(
                        I.reshape(
                            I.cast(
                                gate[batch, position, value_head],
                                I.f32,
                            ),
                            (),
                        )
                    )
                    decayed_state = state * decay
                    remembered = I.reduce.sum(
                        decayed_state * key_vector[:, None],
                        axis=0,
                        identity=0.0,
                    )
                    update = (
                        value_vector - remembered
                    ) * I.reshape(
                        I.cast(
                            beta[batch, position, value_head],
                            I.f32,
                        ),
                        (),
                    )
                    next_state = decayed_state + key_vector[:, None] * update[None, :]
                    output[
                        batch,
                        position_region,
                        value_head,
                        value_features,
                    ] = I.cast(
                        I.reduce.sum(
                            next_state * query_vector[:, None],
                            axis=0,
                            identity=0.0,
                        )[None, :],
                        I.bf16,
                    )
                    stream.yield_(next_state)
            final_state[
                batch,
                value_head,
                key_features,
                value_features,
            ] = stream.result
