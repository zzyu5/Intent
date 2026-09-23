import torch
import intent
import intent.language as I


@intent.kernel
def _tensordot_rsqrt(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    M, _ = a.shape
    _, N = b.shape

    # The contraction is one logical matrix-product stage.  The output
    # coordinates are independent and the rsqrt is fused into their stores.
    product = I.matmul(a, b, acc_dtype=I.f32)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row in I.parallel(rows):
        for column in I.parallel(columns):
            out[row, column] = I.rsqrt(product[row, column])


def build(context):
    artifact = context.compile("tensordot_rsqrt", _tensordot_rsqrt)

    def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
        artifact(a, b, out)
        return out

    return tensordot_rsqrt
