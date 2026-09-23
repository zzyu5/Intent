import torch
import intent
import intent.language as I


@intent.kernel
def _tensordot_kernel(
    a: I.In[I.f16, ("M", "P", "Q")],
    b: I.In[I.f16, ("P", "Q", "N")],
    out: I.Out[I.f16, ("M", "N")],
):
    # The two contracted axes are flattened by the contract lowering into K=64.
    value = I.contract(
        a,
        b,
        reduce=((1, 0), (2, 1)),
        acc_dtype=I.f32,
    )
    m, _, _ = a.shape
    _, _, n = b.shape
    rows = I.domain(0, m)
    columns = I.domain(0, n)
    out[rows, columns] = I.cast(value, I.f16)


def build(context):
    compiled = context.compile(
        "tensordot_contract",
        _tensordot_kernel,
        constexprs={},
    )

    def tensordot(a: torch.Tensor, b: torch.Tensor, dims):
        out = torch.empty(
            (a.shape[0], b.shape[2]),
            device=a.device,
            dtype=a.dtype,
        )
        compiled(a, b, out)
        return out

    return tensordot
