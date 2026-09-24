import torch
import intent
import intent.language as I


@intent.kernel
def _cosine_loss_rows(
    input1: I.In[I.f32, ("M", "N")],
    input2: I.In[I.f32, ("M", "N")],
    target: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M",)],
    margin: I.f32,
):
    M, N = input1.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    x1 = input1[rows, columns]
    x2 = input2[rows, columns]
    norm1_sq = I.reduce.sum(x1 * x1, axis=1, acc_dtype=I.f32)
    norm2_sq = I.reduce.sum(x2 * x2, axis=1, acc_dtype=I.f32)

    eps = I.cast(1.0e-12, I.f32)
    norm1 = I.maximum(I.sqrt(norm1_sq), eps)
    norm2 = I.maximum(I.sqrt(norm2_sq), eps)
    norm1 = I.reshape(norm1, (M, 1))
    norm2 = I.reshape(norm2, (M, 1))

    cosine = I.reduce.sum((x1 / norm1) * (x2 / norm2), axis=1, acc_dtype=I.f32)
    target_value = target[rows]
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    base_loss = I.maximum(one - cosine * target_value, zero)
    margin_loss = margin - cosine
    loss = I.select(margin > zero, I.maximum(base_loss, margin_loss), base_loss)
    output[rows] = loss


@intent.kernel
def _reduce_loss(
    values: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ()],
    scale: I.f32,
):
    M, = values.shape
    rows = I.domain(0, M)
    total = I.reduce.sum(values[rows], axis=0, acc_dtype=I.f32)
    output[()] = total * scale


def build(context):
    row_kernel = context.compile("cosine_loss_rows", _cosine_loss_rows)
    reduce_kernel = context.compile("cosine_loss_reduce", _reduce_loss)

    def fused_cosine_embedding_loss_with_normalization(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction not in ("none", "sum", "mean"):
            raise ValueError("reduction must be one of 'none', 'sum', or 'mean'")

        row_losses = torch.empty(
            (input1.shape[0],), device=input1.device, dtype=input1.dtype
        )
        row_kernel(input1, input2, target, row_losses, margin)
        if reduction == "none":
            return row_losses

        result = torch.empty((), device=input1.device, dtype=input1.dtype)
        scale = 1.0 if reduction == "sum" else 1.0 / input1.shape[0]
        reduce_kernel(row_losses, result, scale)
        return result

    return fused_cosine_embedding_loss_with_normalization
