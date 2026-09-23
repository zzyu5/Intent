import torch
import intent
import intent.language as I


@intent.fn
def _cosine_losses(input1, input2, target, margin):
    m, _ = input1.shape
    eps = I.cast(1.0e-12, I.f32)

    norm1_sq = I.reduce.sum(input1 * input1, axis=1, acc_dtype=I.f32)
    norm2_sq = I.reduce.sum(input2 * input2, axis=1, acc_dtype=I.f32)
    norm1 = I.maximum(I.sqrt(norm1_sq), eps)
    norm2 = I.maximum(I.sqrt(norm2_sq), eps)

    norm1 = I.reshape(norm1, (m, 1))
    norm2 = I.reshape(norm2, (m, 1))
    unit1 = input1 / norm1
    unit2 = input2 / norm2
    cosine = I.reduce.sum(unit1 * unit2, axis=1, acc_dtype=I.f32)

    base = I.cast(1.0, I.f32) - cosine * target
    losses = I.maximum(base, I.cast(0.0, I.f32))
    if margin > I.cast(0.0, I.f32):
        losses = I.maximum(losses, margin - cosine)
    return losses


@intent.kernel
def _loss_none(
    input1: I.In[I.f32, ("M", "K")],
    input2: I.In[I.f32, ("M", "K")],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    output: I.Out[I.f32, ("M",)],
):
    m, k = input1.shape
    rows = I.domain(0, m)
    columns = I.domain(0, k)
    losses = _cosine_losses(input1[rows, columns], input2[rows, columns], target[rows], margin)
    output[rows] = losses


@intent.kernel
def _loss_sum(
    input1: I.In[I.f32, ("M", "K")],
    input2: I.In[I.f32, ("M", "K")],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    output: I.Out[I.f32, ()],
):
    m, k = input1.shape
    rows = I.domain(0, m)
    columns = I.domain(0, k)
    losses = _cosine_losses(input1[rows, columns], input2[rows, columns], target[rows], margin)
    output[()] = I.reduce.sum(losses, axis=0, acc_dtype=I.f32)


@intent.kernel
def _loss_mean(
    input1: I.In[I.f32, ("M", "K")],
    input2: I.In[I.f32, ("M", "K")],
    target: I.In[I.f32, ("M",)],
    margin: I.f32,
    output: I.Out[I.f32, ()],
):
    m, k = input1.shape
    rows = I.domain(0, m)
    columns = I.domain(0, k)
    losses = _cosine_losses(input1[rows, columns], input2[rows, columns], target[rows], margin)
    total = I.reduce.sum(losses, axis=0, acc_dtype=I.f32)
    output[()] = total / I.cast(m, I.f32)


def build(context):
    none_kernel = context.compile("fused_cosine_embedding_loss_none", _loss_none)
    sum_kernel = context.compile("fused_cosine_embedding_loss_sum", _loss_sum)
    mean_kernel = context.compile("fused_cosine_embedding_loss_mean", _loss_mean)

    def fused_cosine_embedding_loss_with_normalization(
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        margin: float = 0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        if reduction == "none":
            return none_kernel.run(input1, input2, target, margin)
        if reduction == "sum":
            return sum_kernel.run(input1, input2, target, margin)
        return mean_kernel.run(input1, input2, target, margin)

    return fused_cosine_embedding_loss_with_normalization
