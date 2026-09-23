import torch
import intent
import intent.language as I


@intent.kernel
def _fused_mv_sigmoid_sub(
    input: I.In[I.f32, ("M", "K")],
    vec: I.In[I.f32, ("K",)],
    other: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, ("M",)],
):
    M, K = input.shape
    rows = I.domain(0, M)
    columns = I.domain(0, K)
    one = I.cast(1.0, I.f32)

    # Each row is independent; the dot is the only ordered/reduction work.
    for row in I.parallel(rows):
        row_values = input[row, columns]
        z = I.dot(row_values, vec[columns], acc_dtype=I.f32)
        sigmoid = I.fdiv(one, one + I.exp(-z))
        out[row] = sigmoid - alpha * other


def build(context):
    kernel = context.compile("fused_mv_sigmoid_sub", _fused_mv_sigmoid_sub)

    def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
        if out is None:
            out = torch.empty((input.shape[0],), device=input.device, dtype=input.dtype)
        kernel(input, vec, float(other), float(alpha), out)
        return out

    return fused_mv_sigmoid_sub
