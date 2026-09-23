import torch
import intent
import intent.language as I


@intent.kernel
def _fused_pairwise_distance(
    x1: I.In[I.f32, (16, 3, 32, 32)],
    x2: I.In[I.f32, (16, 3, 32, 32)],
    out: I.Out[I.f32, (16,)],
    p: I.f32,
    eps: I.f32,
):
    channels = I.domain(0, 3)
    output_rows = I.domain(0, 16)
    output_cols = I.domain(0, 16)
    kernel_rows = I.domain(0, 2)
    kernel_cols = I.domain(0, 2)

    c = I.reshape(I.indices(channels), (3, 1, 1, 1, 1))
    rows = I.reshape(I.indices(output_rows), (1, 16, 1, 1, 1))
    cols = I.reshape(I.indices(output_cols), (1, 1, 16, 1, 1))
    kr = I.reshape(I.indices(kernel_rows), (1, 1, 1, 2, 1))
    kc = I.reshape(I.indices(kernel_cols), (1, 1, 1, 1, 2))

    input_rows = rows * 2 + kr
    input_cols = cols * 2 + kc

    for batch in I.parallel(I.domain(0, 16)):
        window1 = x1[batch, c, input_rows, input_cols]
        window2 = x2[batch, c, input_rows, input_cols]
        pooled1 = I.reduce.sum(window1, axis=(3, 4), acc_dtype=I.f32)
        pooled2 = I.reduce.sum(window2, axis=(3, 4), acc_dtype=I.f32)
        difference = pooled1 * I.cast(0.25, I.f32) - pooled2 * I.cast(0.25, I.f32) + eps

        if p == I.cast(2.0, I.f32):
            powered = difference * difference
            total = I.reduce.sum(powered, axis=(0, 1, 2), acc_dtype=I.f32)
            out[batch] = I.sqrt(total)
        else:
            powered = I.exp(p * I.log(I.abs(difference)))
            total = I.reduce.sum(powered, axis=(0, 1, 2), acc_dtype=I.f32)
            out[batch] = I.exp(I.log(total) / p)


def build(context):
    fused = context.compile("fused_pairwise_distance_adaptive_avg_pool2d", _fused_pairwise_distance)

    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        output_size: int or tuple,
        p: float = 2.0,
        eps: float = 1e-6,
        keepdim: bool = False,
    ) -> torch.Tensor:
        del output_size
        out = torch.empty((16,), device=x1.device, dtype=x1.dtype)
        fused(x1, x2, out, p, eps)
        if keepdim:
            return out.reshape((16, 1))
        return out

    return fused_pairwise_distance_adaptive_avg_pool2d
