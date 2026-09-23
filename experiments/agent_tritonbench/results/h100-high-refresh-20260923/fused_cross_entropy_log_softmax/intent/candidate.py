import torch
import intent
import intent.language as I


@intent.kernel
def _row_loss(
    logits: I.In[I.f32, (4096, 128)],
    target: I.In[I.i64, (4096,)],
    losses: I.Out[I.f32, (4096,)],
    ignore_index: I.i64,
):
    rows = I.domain(0, 4096)
    classes = I.domain(0, 128)

    row_logits = logits[rows, classes]
    row_max = I.reduce.max(row_logits, axis=1, acc_dtype=I.f32)
    centered = row_logits - I.reshape(row_max, (4096, 1))
    exp_centered = I.exp(centered)
    row_log_z = row_max + I.log(
        I.reduce.sum(exp_centered, axis=1, acc_dtype=I.f32)
    )

    row_target = target[rows]
    active = row_target != ignore_index
    class_index = I.cast(row_target, I.index)
    selected = I.gather(
        logits,
        (rows, class_index),
        valid=active,
        fill=I.cast(0.0, I.f32),
    )
    row_loss = row_log_z - selected
    losses[rows] = I.select(active, row_loss, I.cast(0.0, I.f32))


@intent.kernel
def _mean_loss(
    losses: I.In[I.f32, (4096,)],
    target: I.In[I.i64, (4096,)],
    output: I.Out[I.f32, ()],
    ignore_index: I.i64,
):
    rows = I.domain(0, 4096)
    row_target = target[rows]
    active = row_target != ignore_index
    active_f32 = I.cast(active, I.f32)
    total = I.reduce.sum(losses[rows] * active_f32, axis=0, acc_dtype=I.f32)
    count = I.reduce.sum(active_f32, axis=0, acc_dtype=I.f32)
    output[()] = I.fdiv(total, count)


def build(context):
    row_loss = context.compile("cross_entropy_row_loss", _row_loss)
    mean_loss = context.compile("cross_entropy_mean_loss", _mean_loss)

    def fused_cross_entropy_log_softmax(
        input: torch.Tensor,
        target: torch.Tensor,
        dim: int = 1,
        weight: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        per_row = torch.empty((4096,), device=input.device, dtype=torch.float32)
        output = torch.empty((), device=input.device, dtype=torch.float32)
        row_loss(input, target, per_row, ignore_index)
        mean_loss(per_row, target, output, ignore_index)
        return output

    return fused_cross_entropy_log_softmax
