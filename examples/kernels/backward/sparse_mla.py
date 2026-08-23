import intent
import intent.language as I


@intent.kernel
def sparse_mla_backward_delta(
    output: I.In[I.bf16, ("B", "S", "H", "D")],
    grad_output: I.In[I.bf16, ("B", "S", "H", "D")],
    delta: I.Out[I.f32, ("B", "S", "H")],
):
    B, S, H, D = output.shape
    features = I.domain(0, D)
    for batch, position, head in I.parallel(
        (I.domain(0, B), I.domain(0, S), I.domain(0, H))
    ):
        delta[batch, position, head] = I.reduce.sum(
            I.cast(output[batch, position, head, features], I.f32)
            * I.cast(grad_output[batch, position, head, features], I.f32),
            axis=0,
            identity=0.0,
            acc_dtype=I.f32,
        )


@intent.kernel
def sparse_mla_backward_main(
    query: I.In[I.bf16, ("B", "S", "H", "DQKV")],
    key_value: I.In[I.bf16, ("B", "SKV", "G", "DQKV")],
    grad_output: I.In[I.bf16, ("B", "S", "H", "DV")],
    selected_indices: I.In[I.i32, ("B", "S", "G", "TOPK")],
    lse: I.In[I.f32, ("B", "S", "H")],
    delta: I.In[I.f32, ("B", "S", "H")],
    grad_key_value: I.InOut[I.f32, ("B", "SKV", "G", "DQKV")],
    grad_query: I.Out[I.bf16, ("B", "S", "H", "DQKV")],
    scale: I.f32,
    HEAD_GROUP: I.Constexpr[int],
):
    B, S, H, DQKV = query.shape
    SKV = key_value.shape[1]
    G = key_value.shape[2]
    DV = grad_output.shape[3]
    TOPK = selected_indices.shape[3]
    local_heads = I.domain(0, HEAD_GROUP)
    value_features = I.domain(0, DV)
    tail_features = I.domain(0, DQKV - DV)
    all_features = I.domain(0, DQKV)
    selected_axis = I.domain(0, TOPK)

    for batch, position, key_value_group in I.parallel(
        (I.domain(0, B), I.domain(0, S), I.domain(0, G))
    ):
        head_indices = key_value_group * HEAD_GROUP + I.indices(local_heads)
        I.assume_in_bounds(head_indices, query, axis=2)
        query_block = I.gather(
            query,
            index=(batch, position, head_indices, slice(None)),
        )
        grad_output_block = I.gather(
            grad_output,
            index=(batch, position, head_indices, slice(None)),
        )
        lse_block = lse[batch, position, head_indices]
        delta_block = delta[batch, position, head_indices]
        accumulation = I.state_stream(
            selected_axis,
            extent=I.auto("TOPK_TILE"),
            init=(I.zeros((local_heads, DQKV), dtype=I.f32),),
            stop=I.end(selected_axis),
        )
        with accumulation:
            for selected_region, grad_query_accumulator in accumulation:
                raw_index = I.cast(
                    selected_indices[
                        batch,
                        position,
                        key_value_group,
                        selected_region,
                    ],
                    I.index,
                )
                valid_index = (
                    (raw_index >= 0)
                    and (raw_index <= position)
                    and (raw_index < SKV)
                )
                safe_index = I.mask(
                    raw_index,
                    valid=valid_index,
                    fill=I.cast(0, I.index),
                )
                I.assume_in_bounds(safe_index, key_value, axis=1)
                selected_key_value = I.gather(
                    key_value,
                    index=(
                        batch,
                        safe_index,
                        key_value_group,
                        slice(None),
                    ),
                )
                key_value_main = selected_key_value[:, value_features]
                key_value_tail = selected_key_value[
                    :, DV + I.indices(tail_features)
                ]
                query_main = query_block[:, value_features]
                query_tail = query_block[:, DV + I.indices(tail_features)]
                probability = I.contract(
                    query_block,
                    selected_key_value,
                    reduce=((1, 1),),
                    acc_dtype=I.f32,
                )
                probability = I.exp2(
                    probability * (scale * I.LOG2E) - lse_block[:, None]
                )
                probability = I.mask(
                    probability,
                    valid=valid_index[None, :],
                    fill=0.0,
                )
                grad_probability = I.contract(
                    grad_output_block,
                    key_value_main,
                    reduce=((1, 1),),
                    acc_dtype=I.f32,
                )
                grad_score = (
                    probability
                    * (grad_probability - delta_block[:, None])
                    * scale
                )
                grad_query_update = I.contract(
                    I.cast(grad_score, I.bf16),
                    selected_key_value,
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                )
                grad_key_value_main = I.contract(
                    I.cast(grad_score, I.bf16),
                    query_main,
                    reduce=((0, 0),),
                    acc_dtype=I.f32,
                ) + I.contract(
                    I.cast(probability, I.bf16),
                    grad_output_block,
                    reduce=((0, 0),),
                    acc_dtype=I.f32,
                )
                grad_key_value_tail = I.contract(
                    I.cast(grad_score, I.bf16),
                    query_tail,
                    reduce=((0, 0),),
                    acc_dtype=I.f32,
                )
                I.atomic_add(
                    grad_key_value,
                    index=(
                        batch,
                        safe_index,
                        key_value_group,
                        value_features,
                    ),
                    value=I.mask(
                        grad_key_value_main,
                        valid=valid_index[:, None],
                        fill=0.0,
                    ),
                )
                I.atomic_add(
                    grad_key_value,
                    index=(
                        batch,
                        safe_index[:, None],
                        key_value_group,
                        DV + I.indices(tail_features)[None, :],
                    ),
                    value=I.mask(
                        grad_key_value_tail,
                        valid=valid_index[:, None],
                        fill=0.0,
                    ),
                )
                accumulation.yield_(
                    grad_query_accumulator + grad_query_update
                )
        I.scatter_unique(
            grad_query,
            index=(batch, position, head_indices, all_features),
            value=I.cast(accumulation.result, I.bf16),
        )


@intent.kernel
def sparse_mla_grad_kv_cast(
    grad_key_value: I.In[I.f32, ("B", "SKV", "G", "D")],
    output: I.Out[I.bf16, ("B", "SKV", "G", "D")],
):
    B, SKV, G, D = grad_key_value.shape
    positions = I.domain(0, SKV)
    features = I.domain(0, D)
    for batch, group in I.parallel((I.domain(0, B), I.domain(0, G))):
        output[batch, positions, group, features] = I.cast(
            grad_key_value[batch, positions, group, features],
            I.bf16,
        )
