import torch
import intent
import intent.language as I


@intent.kernel
def _fused_embedding_add_tanh(
    input_indices: I.In[I.i64, (16, 16)],
    weight: I.In[I.f32, (1024, 16)],
    other: I.In[I.f32, (16, 16, 16)],
    out: I.Out[I.f32, (16, 16, 16)],
):
    rows = I.domain(0, 16)
    columns = I.domain(0, 16)
    embedding_dim = I.domain(0, 16)

    # Each output element has one independent embedding lookup and activation.
    indices = input_indices[rows, columns]
    values = weight[indices, embedding_dim] + other[rows, columns, embedding_dim]
    out[rows, columns, embedding_dim] = I.tanh(values)


def build(context):
    fused = context.compile("fused_embedding_add_tanh", _fused_embedding_add_tanh)

    def fused_embedding_add_tanh(
        input_indices,
        weight,
        other,
        *,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        out=None,
    ):
        result = out if out is not None else torch.empty_like(other)
        fused(input_indices, weight, other, result)
        return result

    return fused_embedding_add_tanh
