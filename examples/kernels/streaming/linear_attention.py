import intent
import intent.language as I


BATCH = 1
SEQUENCE = 2048
HEADS = 16
DIMENSION = 128
CHUNK_SIZE = 64


@intent.kernel
def fused_chunk_linear_attention_fwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    output: I.Out[I.f32, ("B", "S", "H", "DV")],
    final_state: I.Out[I.f32, ("B", "H", "D", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    key_dimensions = I.domain(0, D)
    value_dimensions = I.domain(0, DV)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            chunks = I.state_stream(
                positions,
                extent=CHUNK_SIZE,
                init=(I.zeros((D, DV), dtype=I.f32),),
            )
            with chunks:
                for chunk_region, state in chunks:
                    query = I.cast(
                        I.cast(q[batch, chunk_region, head, :], I.f32) * scale,
                        I.f16,
                    )
                    key = k[batch, chunk_region, head, :]
                    value = v[batch, chunk_region, head, :]
                    scores = I.contract(
                        query,
                        key,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    positions_in_chunk = I.indices(chunk_region)
                    scores = I.mask(
                        scores,
                        valid=positions_in_chunk[:, None]
                        >= positions_in_chunk[None, :],
                        fill=0.0,
                    )
                    intra = I.contract(
                        I.cast(scores, I.f16),
                        value,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    inter = I.contract(
                        query,
                        I.cast(state, I.f16),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    output[batch, chunk_region, head, :] = intra + inter
                    update = I.contract(
                        key,
                        value,
                        reduce=((0, 0),),
                        acc_dtype=I.f32,
                    )
                    chunks.yield_(state + update)
            final_state[batch, head, key_dimensions, value_dimensions] = chunks.result


@intent.kernel
def fused_chunk_linear_attention_bwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    grad_output: I.In[I.f16, ("B", "S", "H", "DV")],
    grad_q: I.Out[I.f32, ("B", "S", "H", "D")],
    grad_k: I.Out[I.f32, ("B", "S", "H", "D")],
    grad_v: I.Out[I.f32, ("B", "S", "H", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    key_dimensions = I.domain(0, D)
    value_dimensions = I.domain(0, DV)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            forward = I.state_stream(
                positions,
                extent=CHUNK_SIZE,
                init=(I.zeros((DV, D), dtype=I.f32),),
            )
            with forward:
                for chunk_region, state in forward:
                    key = k[batch, chunk_region, head, :]
                    value = v[batch, chunk_region, head, :]
                    grad = grad_output[batch, chunk_region, head, :]
                    local_positions = I.indices(chunk_region)
                    score_gradient = I.contract(
                        grad,
                        value,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    score_gradient = I.mask(
                        score_gradient,
                        valid=local_positions[:, None]
                        >= local_positions[None, :],
                        fill=0.0,
                    )
                    local_grad_q = I.contract(
                        I.cast(score_gradient, I.f16),
                        key,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    carried_grad_q = I.contract(
                        grad,
                        I.cast(state, I.f16),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    grad_q[batch, chunk_region, head, key_dimensions] = (
                        local_grad_q + carried_grad_q
                    ) * scale
                    state_update = I.contract(
                        value,
                        key,
                        reduce=((0, 0),),
                        acc_dtype=I.f32,
                    )
                    forward.yield_(state + state_update)

            reverse = I.state_stream(
                positions,
                extent=CHUNK_SIZE,
                init=(I.zeros((D, DV), dtype=I.f32),),
            )
            with reverse:
                for traversal_region, grad_state in reverse:
                    traversal = I.indices(traversal_region)
                    local = traversal - traversal[0]
                    reverse_start = S - CHUNK_SIZE - traversal[0]
                    reverse_positions = reverse_start + local
                    query = I.cast(
                        I.cast(
                            q[
                                batch,
                                reverse_positions,
                                head,
                                key_dimensions,
                            ],
                            I.f32,
                        )
                        * scale,
                        I.f16,
                    )
                    key = k[
                        batch,
                        reverse_positions,
                        head,
                        key_dimensions,
                    ]
                    value = v[
                        batch,
                        reverse_positions,
                        head,
                        value_dimensions,
                    ]
                    grad = grad_output[
                        batch,
                        reverse_positions,
                        head,
                        value_dimensions,
                    ]
                    score_gradient = I.contract(
                        value,
                        grad,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    score_gradient = I.mask(
                        score_gradient,
                        valid=local[:, None] <= local[None, :],
                        fill=0.0,
                    )
                    local_grad_k = I.contract(
                        I.cast(score_gradient, I.f16),
                        query,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    carried_grad_k = I.contract(
                        value,
                        I.cast(grad_state, I.f16),
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    scores = I.contract(
                        key,
                        query,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    scores = I.mask(
                        scores,
                        valid=local[:, None] <= local[None, :],
                        fill=0.0,
                    )
                    local_grad_v = I.contract(
                        I.cast(scores, I.f16),
                        grad,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    carried_grad_v = I.contract(
                        key,
                        I.cast(grad_state, I.f16),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    I.scatter_unique(
                        grad_k,
                        index=(
                            batch,
                            reverse_positions,
                            head,
                            key_dimensions,
                        ),
                        value=local_grad_k + carried_grad_k,
                    )
                    I.scatter_unique(
                        grad_v,
                        index=(
                            batch,
                            reverse_positions,
                            head,
                            value_dimensions,
                        ),
                        value=local_grad_v + carried_grad_v,
                    )
                    state_update = I.contract(
                        query,
                        grad,
                        reduce=((0, 0),),
                        acc_dtype=I.f32,
                    )
                    reverse.yield_(grad_state + state_update)


@intent.kernel
def chunk_retention_fwd(
    q: I.In[I.f16, ("B", "S", "H", "D")],
    k: I.In[I.f16, ("B", "S", "H", "D")],
    v: I.In[I.f16, ("B", "S", "H", "DV")],
    output: I.Out[I.f16, ("B", "S", "H", "DV")],
    scale: I.f32,
):
    B, S, H, D = q.shape
    DV = v.shape[3]
    positions = I.domain(0, S)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            log_decay = I.log(1.0 - I.exp(-5.0 - I.cast(head, I.f32)))
            chunks = I.state_stream(
                positions,
                extent=CHUNK_SIZE,
                init=(I.zeros((D, DV), dtype=I.f32),),
            )
            with chunks:
                for chunk_region, state in chunks:
                    query = I.cast(
                        I.cast(q[batch, chunk_region, head, :], I.f32) * scale,
                        I.f16,
                    )
                    key = k[batch, chunk_region, head, :]
                    value = v[batch, chunk_region, head, :]
                    local = I.indices(chunk_region) - I.indices(chunk_region)[0]
                    score = I.contract(
                        query,
                        key,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    causal = local[:, None] >= local[None, :]
                    decay = I.exp(
                        I.cast(local[:, None] - local[None, :], I.f32)
                        * log_decay
                    )
                    score = I.mask(score * decay, valid=causal, fill=0.0)
                    intra = I.contract(
                        I.cast(score, I.f16),
                        value,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    inter = I.contract(
                        query,
                        I.cast(state, I.f16),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    ) * I.exp(I.cast(local + 1, I.f32) * log_decay)[:, None]
                    output[batch, chunk_region, head, :] = I.cast(
                        intra + inter, I.f16
                    )
                    weighted_value = value * I.cast(
                        I.exp(
                            I.cast(CHUNK_SIZE - local - 1, I.f32) * log_decay
                        )[:, None],
                        I.f16,
                    )
                    update = I.contract(
                        key,
                        weighted_value,
                        reduce=((0, 0),),
                        acc_dtype=I.f32,
                    )
                    chunks.yield_(
                        I.exp(I.cast(CHUNK_SIZE, I.f32) * log_decay) * state
                        + update
                    )
