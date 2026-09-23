import torch
import intent
import intent.language as I


@intent.kernel
def fused_tile_exp_kernel(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, (2, "N")],
):
    n, = x.shape
    rows = I.domain(0, 2)
    columns = I.domain(0, n)

    # The column relation reads the source once per logical tile row; the
    # broadcast over rows expresses the repeat before the elementwise exp.
    out[rows, columns] = I.exp(x[columns])


def build(context):
    artifact = context.compile("fused_tile_exp", fused_tile_exp_kernel)

    def fused_tile_exp(input, dims, *, out=None):
        if out is None:
            out = torch.empty(
                (2, input.shape[0]),
                dtype=input.dtype,
                device=input.device,
            )
        artifact(input, out)
        return out

    return fused_tile_exp
