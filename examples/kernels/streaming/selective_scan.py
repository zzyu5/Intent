import intent
import intent.language as I


BATCH = 128
LENGTH = 4096
MAMBA_BATCH = 2
MAMBA_CHUNKS = 32
MAMBA_GROUPS = 2
MAMBA_CHUNK_SIZE = 64
MAMBA_HEADS = 8
MAMBA_HEAD_DIMENSION = 64
MAMBA_STATE_DIMENSION = 128


@intent.kernel
def selective_state_scan(
    x: I.In[I.f32, ("B", "L")],
    decay: I.In[I.f32, ("B", "L")],
    drive: I.In[I.f32, ("B", "L")],
    output: I.Out[I.f32, ("B", "L")],
):
    B, L = x.shape
    positions = I.domain(0, L)
    for batch in I.parallel(I.domain(0, B)):
        state = I.cast(0.0, I.f32)
        for position in positions:
            state = (
                decay[batch, position] * state
                + drive[batch, position] * x[batch, position]
            )
            output[batch, position] = state


@intent.kernel
def mamba_chunk_scan_fwd(
    cb: I.In[I.f16, ("B", "C", "G", "S", "S")],
    x: I.In[I.f16, ("B", "L", "H", "P")],
    dt: I.In[I.f16, ("B", "H", "C", "S")],
    dA_cumsum: I.In[I.f16, ("B", "H", "C", "S")],
    state_matrix: I.In[I.f16, ("B", "L", "G", "N")],
    previous_states: I.In[I.f16, ("B", "C", "H", "P", "N")],
    residual_scale: I.In[I.f16, ("H",)],
    output: I.Out[I.f16, ("B", "L", "H", "P")],
    HEADS_PER_GROUP: I.Constexpr[int],
):
    B, C, G, S, _ = cb.shape
    H = x.shape[2]
    P = x.shape[3]
    N = state_matrix.shape[3]
    rows = I.domain(0, S)
    columns = I.domain(0, P)
    state_axis = I.domain(0, N)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEADS_PER_GROUP
                I.assume_in_bounds(group, state_matrix, axis=2)
                I.assume_in_bounds(group, cb, axis=2)
                row_index = I.indices(rows)
                column_index = I.indices(columns)
                global_row = chunk * S + row_index
                state_term = I.contract(
                    state_matrix[
                        batch, global_row, group, state_axis
                    ],
                    previous_states[
                        batch, chunk, head, column_index, state_axis
                    ],
                    reduce=((1, 1),),
                    acc_dtype=I.f32,
                )
                state_term = state_term * I.exp2(
                    I.cast(
                        dA_cumsum[batch, head, chunk, rows],
                        I.f32,
                    )[:, None]
                    * I.LOG2E
                )
                scan = I.state_stream(
                    rows,
                    extent=I.auto("K_TILE"),
                    init=(state_term,),
                    stop=I.end(rows),
                )
                with scan:
                    for scan_region, accumulator in scan:
                        scan_index = I.indices(scan_region)
                        global_scan = chunk * S + scan_index
                        decay = I.exp2(
                            I.minimum(
                                I.cast(
                                    dA_cumsum[
                                        batch, head, chunk, rows
                                    ],
                                    I.f32,
                                )[:, None]
                                - I.cast(
                                    dA_cumsum[
                                        batch, head, chunk, scan_region
                                    ],
                                    I.f32,
                                )[None, :],
                                0.0,
                            )
                            * I.LOG2E
                        )
                        coefficient = (
                            I.cast(
                                cb[
                                    batch,
                                    chunk,
                                    group,
                                    rows,
                                    scan_region,
                                ],
                                I.f32,
                            )
                            * decay
                            * I.cast(
                                dt[batch, head, chunk, scan_region],
                                I.f32,
                            )[None, :]
                        )
                        coefficient = I.mask(
                            coefficient,
                            valid=global_row[:, None]
                            >= global_scan[None, :],
                            fill=0.0,
                        )
                        partial = I.contract(
                            I.cast(coefficient, I.f16),
                            x[
                                batch,
                                chunk * S + scan_index[:, None],
                                head,
                                column_index[None, :],
                            ],
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        )
                        scan.yield_(accumulator + partial)
                scan_term = scan.result
                residual = (
                    I.cast(
                        x[
                            batch,
                            global_row[:, None],
                            head,
                            column_index[None, :],
                        ],
                        I.f32,
                    )
                    * I.cast(residual_scale[head], I.f32)
                )
                I.scatter_unique(
                    output,
                    index=(
                        batch,
                        global_row[:, None],
                        head,
                        column_index[None, :],
                    ),
                    value=I.cast(scan_term + residual, I.f16),
                )


@intent.kernel
def mamba_chunk_scan_bf16_fwd(
    cb: I.In[I.bf16, ("B", "C", "G", "S", "S")],
    x: I.In[I.bf16, ("B", "L", "H", "P")],
    dt: I.In[I.f32, ("B", "H", "C", "S")],
    dA_cumsum: I.In[I.f32, ("B", "H", "C", "S")],
    state_matrix: I.In[I.bf16, ("B", "L", "G", "N")],
    previous_states: I.In[I.f32, ("B", "C", "H", "P", "N")],
    residual_scale: I.In[I.f32, ("H",)],
    output: I.Out[I.bf16, ("B", "L", "H", "P")],
    HEADS_PER_GROUP: I.Constexpr[int],
):
    B, C, G, S, _ = cb.shape
    H = x.shape[2]
    P = x.shape[3]
    N = state_matrix.shape[3]
    rows = I.domain(0, S)
    columns = I.domain(0, P)
    state_axis = I.domain(0, N)
    for batch in I.parallel(I.domain(0, B)):
        for chunk in I.parallel(I.domain(0, C)):
            for head in I.parallel(I.domain(0, H)):
                group = head // HEADS_PER_GROUP
                I.assume_in_bounds(group, state_matrix, axis=2)
                I.assume_in_bounds(group, cb, axis=2)
                row_index = I.indices(rows)
                column_index = I.indices(columns)
                global_row = chunk * S + row_index
                state_term = I.contract(
                    state_matrix[
                        batch, global_row, group, state_axis
                    ],
                    I.cast(
                        previous_states[
                            batch, chunk, head, column_index, state_axis
                        ],
                        I.bf16,
                    ),
                    reduce=((1, 1),),
                    acc_dtype=I.f32,
                )
                state_term = state_term * I.exp2(
                    dA_cumsum[batch, head, chunk, rows][:, None]
                    * I.LOG2E
                )
                scan = I.state_stream(
                    rows,
                    extent=I.auto("K_TILE"),
                    init=(state_term,),
                    stop=I.end(rows),
                )
                with scan:
                    for scan_region, accumulator in scan:
                        scan_index = I.indices(scan_region)
                        global_scan = chunk * S + scan_index
                        decay = I.exp2(
                            I.minimum(
                                dA_cumsum[
                                    batch, head, chunk, rows
                                ][:, None]
                                - dA_cumsum[
                                    batch, head, chunk, scan_region
                                ][None, :],
                                0.0,
                            )
                            * I.LOG2E
                        )
                        coefficient = (
                            I.cast(
                                cb[
                                    batch,
                                    chunk,
                                    group,
                                    rows,
                                    scan_region,
                                ],
                                I.f32,
                            )
                            * decay
                            * dt[batch, head, chunk, scan_region][None, :]
                        )
                        coefficient = I.mask(
                            coefficient,
                            valid=global_row[:, None]
                            >= global_scan[None, :],
                            fill=0.0,
                        )
                        partial = I.contract(
                            I.cast(coefficient, I.bf16),
                            x[
                                batch,
                                chunk * S + scan_index[:, None],
                                head,
                                column_index[None, :],
                            ],
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        )
                        scan.yield_(accumulator + partial)
                scan_term = scan.result
                residual = (
                    I.cast(
                        x[
                            batch,
                            global_row[:, None],
                            head,
                            column_index[None, :],
                        ],
                        I.f32,
                    )
                    * residual_scale[head]
                )
                I.scatter_unique(
                    output,
                    index=(
                        batch,
                        global_row[:, None],
                        head,
                        column_index[None, :],
                    ),
                    value=I.cast(scan_term + residual, I.bf16),
                )
