import intent
import intent.language as I


BATCH = 1
SEQUENCE = 2048
HEADS = 32
GROUPS = 8
HEAD_DIMENSION = 64
STATE_DIMENSION = 128
CHUNK_SIZE = 256
CHUNKS = SEQUENCE // CHUNK_SIZE
HEADS_PER_GROUP = HEADS // GROUPS
MAMBA3_QK_HEADS = 4
MAMBA3_HEADS = 16
MAMBA3_QK_DIMENSION = 32
MAMBA3_VALUE_DIMENSION = 64
MAMBA3_ANGLE_DIMENSION = 16


@intent.kernel
def mamba_chunk_state_fwd(
    state_basis: I.In[I.f16, ("B", "L", "G", "N")],
    x: I.In[I.f16, ("B", "L", "H", "P")],
    dt: I.In[I.f16, ("B", "H", "C", "S")],
    cumulative_decay: I.In[I.f16, ("B", "H", "C", "S")],
    states: I.Out[I.f16, ("B", "C", "H", "P", "N")],
    HEAD_GROUP: I.Constexpr[int],
):
    B, L, H, P = x.shape
    C = dt.shape[2]
    S = dt.shape[3]
    N = state_basis.shape[3]
    chunk_axis = I.domain(0, S)
    state_axis = I.domain(0, N)
    dimensions = I.domain(0, P)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEAD_GROUP
                for dimension_region in I.parallel(
                    I.partition(dimensions, extent=I.auto("M_TILE"))
                ):
                    for state_region in I.parallel(
                        I.partition(state_axis, extent=I.auto("N_TILE"))
                    ):
                        positions = chunk * S + I.indices(chunk_axis)
                        last_decay = I.cast(
                            cumulative_decay[batch, head, chunk, S - 1], I.f32
                        )
                        scale = I.exp(
                            I.minimum(
                                last_decay
                                - I.cast(
                                    cumulative_decay[batch, head, chunk, chunk_axis],
                                    I.f32,
                                ),
                                0.0,
                            )
                        ) * I.cast(dt[batch, head, chunk, chunk_axis], I.f32)
                        lhs = I.cast(
                            x[batch, positions, head, dimension_region], I.f16
                        )
                        rhs = I.cast(
                            state_basis[batch, positions, group, state_region]
                            * I.cast(scale[:, None], I.f16),
                            I.f16,
                        )
                        result = I.contract(
                            lhs,
                            rhs,
                            reduce=((0, 0),),
                            acc_dtype=I.f32,
                        )
                        states[batch, chunk, head, dimension_region, state_region] = (
                            I.cast(result, I.f16)
                        )


@intent.kernel
def mamba_chunk_state_bf16_fwd(
    state_basis: I.In[I.bf16, ("B", "L", "G", "N")],
    x: I.In[I.bf16, ("B", "L", "H", "P")],
    dt: I.In[I.f32, ("B", "H", "C", "S")],
    cumulative_decay: I.In[I.f32, ("B", "H", "C", "S")],
    states: I.Out[I.f32, ("B", "C", "H", "P", "N")],
    HEAD_GROUP: I.Constexpr[int],
):
    B, L, H, P = x.shape
    C = dt.shape[2]
    S = dt.shape[3]
    N = state_basis.shape[3]
    chunk_axis = I.domain(0, S)
    state_axis = I.domain(0, N)
    dimensions = I.domain(0, P)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEAD_GROUP
                for dimension_region in I.parallel(
                    I.partition(dimensions, extent=I.auto("M_TILE"))
                ):
                    for state_region in I.parallel(
                        I.partition(state_axis, extent=I.auto("N_TILE"))
                    ):
                        positions = chunk * S + I.indices(chunk_axis)
                        last_decay = cumulative_decay[batch, head, chunk, S - 1]
                        scale = I.exp(
                            I.minimum(
                                last_decay
                                - cumulative_decay[batch, head, chunk, chunk_axis],
                                0.0,
                            )
                        ) * dt[batch, head, chunk, chunk_axis]
                        lhs = x[batch, positions, head, dimension_region]
                        rhs = I.cast(
                            I.cast(
                                state_basis[
                                    batch, positions, group, state_region
                                ],
                                I.f32,
                            )
                            * scale[:, None],
                            I.bf16,
                        )
                        states[
                            batch,
                            chunk,
                            head,
                            dimension_region,
                            state_region,
                        ] = I.contract(
                            lhs,
                            rhs,
                            reduce=((0, 0),),
                            acc_dtype=I.f32,
                        )


@intent.kernel
def mamba_state_passing_fwd(
    chunk_states: I.In[I.f32, ("B", "C", "H", "D")],
    chunk_decay: I.In[I.f32, ("B", "H", "C")],
    initial_states: I.In[I.f32, ("B", "H", "D")],
    states_before_chunk: I.Out[I.f32, ("B", "C", "H", "D")],
    final_states: I.Out[I.f32, ("B", "H", "D")],
):
    B, C, H, D = chunk_states.shape
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            state = I.cast(initial_states[batch, head, dimensions], I.f32)
            for chunk in I.ordered(I.domain(0, C)):
                states_before_chunk[batch, chunk, head, dimensions] = state
                state = (
                    I.exp(chunk_decay[batch, head, chunk]) * state
                    + chunk_states[batch, chunk, head, dimensions]
                )
            final_states[batch, head, dimensions] = state


@intent.kernel
def mamba3_siso_step(
    query: I.In[I.bf16, ("B", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    key: I.In[I.bf16, ("B", MAMBA3_QK_HEADS, MAMBA3_QK_DIMENSION)],
    value: I.In[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    adt: I.In[I.f32, ("B", MAMBA3_HEADS)],
    dt: I.In[I.f32, ("B", MAMBA3_HEADS)],
    trap: I.In[I.f32, ("B", MAMBA3_HEADS)],
    query_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    key_bias: I.In[I.bf16, (MAMBA3_HEADS, MAMBA3_QK_DIMENSION)],
    angles: I.In[I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)],
    residual_scale: I.In[I.f32, (MAMBA3_HEADS,)],
    gate: I.In[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    input_angle_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)
    ],
    input_ssm_state: I.In[
        I.f32,
        ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION, MAMBA3_QK_DIMENSION),
    ],
    input_key_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    input_value_state: I.In[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)
    ],
    output: I.Out[I.bf16, ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION)],
    output_angle_state: I.Out[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_ANGLE_DIMENSION)
    ],
    output_ssm_state: I.Out[
        I.f32,
        ("B", MAMBA3_HEADS, MAMBA3_VALUE_DIMENSION, MAMBA3_QK_DIMENSION),
    ],
    output_key_state: I.Out[
        I.f32, ("B", MAMBA3_HEADS, MAMBA3_QK_DIMENSION)
    ],
    HEAD_GROUP: I.Constexpr[int],
):
    B = query.shape[0]
    value_dimensions = I.domain(0, MAMBA3_VALUE_DIMENSION)
    angle_dimensions = I.domain(0, MAMBA3_ANGLE_DIMENSION)
    for batch, head in I.parallel(
        (I.domain(0, B), I.domain(0, MAMBA3_HEADS))
    ):
        for angle_region in I.parallel(
            I.partition(angle_dimensions, extent=MAMBA3_ANGLE_DIMENSION)
        ):
            for value_region in I.parallel(
                I.partition(value_dimensions, extent=MAMBA3_VALUE_DIMENSION)
            ):
                pair_indices = I.indices(angle_region) * 2
                query_head = head // HEAD_GROUP
                query_first = I.cast(
                    query[batch, query_head, pair_indices], I.f32
                )
                query_second = I.cast(
                    query[batch, query_head, pair_indices + 1], I.f32
                )
                key_first = I.cast(
                    key[batch, query_head, pair_indices], I.f32
                )
                key_second = I.cast(
                    key[batch, query_head, pair_indices + 1], I.f32
                )
                query_first = query_first + I.cast(
                    query_bias[head, pair_indices], I.f32
                )
                query_second = query_second + I.cast(
                    query_bias[head, pair_indices + 1], I.f32
                )
                key_first = key_first + I.cast(
                    key_bias[head, pair_indices], I.f32
                )
                key_second = key_second + I.cast(
                    key_bias[head, pair_indices + 1], I.f32
                )

                angle_delta = (
                    2.0
                    * I.sigmoid(2.0 * angles[batch, head, angle_region])
                    - 1.0
                ) * 3.141592653589793 * dt[batch, head]
                angle = (
                    angle_delta
                    + input_angle_state[batch, head, angle_region]
                )
                angle = I.mask(
                    angle + 6.283185307179586,
                    valid=angle < 0.0,
                    fill=angle,
                )
                angle = I.mask(
                    angle - 6.283185307179586,
                    valid=angle >= 6.283185307179586,
                    fill=angle,
                )
                output_angle_state[batch, head, angle_region] = angle
                cosine = I.cos(angle)
                sine = I.sin(angle)

                rotated_query_first = I.cast(
                    query_first * cosine - query_second * sine, I.bf16
                )
                rotated_query_second = I.cast(
                    query_first * sine + query_second * cosine, I.bf16
                )
                rotated_key_first = I.cast(
                    key_first * cosine - key_second * sine, I.bf16
                )
                rotated_key_second = I.cast(
                    key_first * sine + key_second * cosine, I.bf16
                )
                I.scatter_unique(
                    output_key_state,
                    index=(batch, head, pair_indices),
                    value=I.cast(rotated_key_first, I.f32),
                )
                I.scatter_unique(
                    output_key_state,
                    index=(batch, head, pair_indices + 1),
                    value=I.cast(rotated_key_second, I.f32),
                )

                alpha = I.exp2(adt[batch, head] * I.LOG2E)
                trap_value = I.sigmoid(trap[batch, head])
                beta = alpha * dt[batch, head] * (1.0 - trap_value)
                gamma = trap_value * dt[batch, head]
                previous_key_first = input_key_state[
                    batch, head, pair_indices
                ]
                previous_key_second = input_key_state[
                    batch, head, pair_indices + 1
                ]
                previous_value = input_value_state[
                    batch, head, value_region
                ]
                current_value = I.cast(
                    value[batch, head, value_region], I.f32
                )
                state_first = (
                    input_ssm_state[
                        batch, head, value_region, pair_indices
                    ]
                    * alpha
                    + (beta * previous_value)[:, None]
                    * previous_key_first[None, :]
                    + (gamma * current_value)[:, None]
                    * I.cast(rotated_key_first, I.f32)[None, :]
                )
                state_second = (
                    input_ssm_state[
                        batch, head, value_region, pair_indices + 1
                    ]
                    * alpha
                    + (beta * previous_value)[:, None]
                    * previous_key_second[None, :]
                    + (gamma * current_value)[:, None]
                    * I.cast(rotated_key_second, I.f32)[None, :]
                )
                I.scatter_unique(
                    output_ssm_state,
                    index=(batch, head, value_region, pair_indices),
                    value=state_first,
                )
                I.scatter_unique(
                    output_ssm_state,
                    index=(batch, head, value_region, pair_indices + 1),
                    value=state_second,
                )
                projected_first = I.contract(
                    I.cast(state_first, I.bf16),
                    I.reshape(rotated_query_first, (angle_region, 1)),
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                )
                projected_second = I.contract(
                    I.cast(state_second, I.bf16),
                    I.reshape(rotated_query_second, (angle_region, 1)),
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                )
                projected = I.reshape(
                    projected_first + projected_second, (value_region,)
                )
                gate_value = I.cast(
                    gate[batch, head, value_region], I.f32
                )
                output[batch, head, value_region] = I.cast(
                    (projected + residual_scale[head] * current_value)
                    * gate_value
                    * I.sigmoid(gate_value),
                    I.bf16,
                )
