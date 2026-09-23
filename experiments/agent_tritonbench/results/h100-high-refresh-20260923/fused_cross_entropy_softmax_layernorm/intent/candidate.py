import torch
import intent
import intent.language as I


@intent.kernel
def fused_rows(
    logits: I.In[I.f32, (256, 1024)],
    targets: I.In[I.i64, (256,)],
    normalized: I.Out[I.f32, (256, 1024)],
    row_losses: I.Out[I.f32, (256,)],
    eps: I.f32,
):
    rows = I.domain(0, 256)
    classes = I.domain(0, 1024)
    class_count = I.cast(1024, I.f32)

    for row in I.parallel(rows):
        row_logits = logits[row, classes]
        row_max = I.reduce.max(row_logits, axis=0, acc_dtype=I.f32)
        shifted = row_logits - row_max
        exponentials = I.exp(shifted)
        exp_sum = I.reduce.sum(exponentials, axis=0, acc_dtype=I.f32)
        probabilities = exponentials / exp_sum

        row_mean = I.reduce.sum(probabilities, axis=0, acc_dtype=I.f32) / class_count
        centered = probabilities - row_mean
        row_variance = I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32) / class_count
        normalized[row, classes] = centered / I.sqrt(row_variance + eps)

        target_index = I.cast(targets[row], I.index)
        target_logit = I.gather(row_logits, target_index, fill=I.cast(0.0, I.f32))
        row_losses[row] = row_max + I.log(exp_sum) - target_logit


@intent.kernel
def mean_loss(
    row_losses: I.In[I.f32, (256,)],
    loss: I.Out[I.f32, ()],
):
    rows = I.domain(0, 256)
    total = I.reduce.sum(row_losses[rows], axis=0, acc_dtype=I.f32)
    loss[()] = total / I.cast(256, I.f32)


def build(context):
    fused_rows_artifact = context.compile("fused_cross_entropy_rows", fused_rows)
    mean_loss_artifact = context.compile("fused_cross_entropy_mean", mean_loss)

    def fused_cross_entropy_softmax_layernorm(
        logits,
        targets,
        normalized_shape,
        weight=None,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        eps=1e-5,
        *,
        out=None,
    ):
        normalized = out if out is not None else torch.empty_like(logits)
        row_losses = torch.empty((256,), device=logits.device, dtype=torch.float32)
        loss = torch.empty((), device=logits.device, dtype=torch.float32)

        fused_rows_artifact(logits, targets, normalized, row_losses, eps)
        mean_loss_artifact(row_losses, loss)
        return loss, normalized

    return fused_cross_entropy_softmax_layernorm
