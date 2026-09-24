import torch
import intent
import intent.language as I


@intent.kernel
def _cholesky(
    A: I.In[I.f32, (256, 256)],
    L: I.Out[I.f32, (256, 256)],
):
    # Columns are the sequential factorization frontier.  Rows below the
    # frontier are independent once the current diagonal is available.
    for j in range(256):
        diagonal = A[j, j]
        if j > 0:
            previous = I.indices(I.domain(0, j))
            diagonal = diagonal - I.reduce.sum(
                L[j, previous] * L[j, previous], axis=0, acc_dtype=I.f32
            )
        L[j, j] = I.sqrt(diagonal)

        for i in I.parallel(I.domain(j + 1, 256)):
            value = A[i, j]
            if j > 0:
                previous = I.indices(I.domain(0, j))
                value = value - I.reduce.sum(
                    L[i, previous] * L[j, previous], axis=0, acc_dtype=I.f32
                )
            L[i, j] = value / L[j, j]


@intent.kernel
def _forward_substitution(
    L: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256, 1)],
    y: I.Out[I.f32, (256, 1)],
):
    for i in range(256):
        value = b[i, 0]
        if i > 0:
            previous = I.indices(I.domain(0, i))
            value = value - I.reduce.sum(
                L[i, previous] * y[previous, 0], axis=0, acc_dtype=I.f32
            )
        y[i, 0] = value / L[i, i]


@intent.kernel
def _backward_substitution(
    L: I.In[I.f32, (256, 256)],
    y: I.In[I.f32, (256, 1)],
    x: I.Out[I.f32, (256, 1)],
):
    for reverse_i in range(256):
        i = 255 - reverse_i
        value = y[i, 0]
        if i < 255:
            following = I.indices(I.domain(i + 1, 256))
            value = value - I.reduce.sum(
                L[following, i] * x[following, 0], axis=0, acc_dtype=I.f32
            )
        x[i, 0] = value / L[i, i]


def build(context):
    cholesky = context.compile("fused_cholesky_factor", _cholesky, constexprs={})
    forward = context.compile(
        "fused_cholesky_forward", _forward_substitution, constexprs={}
    )
    backward = context.compile(
        "fused_cholesky_backward", _backward_substitution, constexprs={}
    )

    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        L = torch.empty_like(A)
        y = torch.empty_like(b)
        x = torch.empty_like(b)

        cholesky(A, L)
        forward(L, b, y)
        backward(L, y, x)
        return x

    return fused_cholesky_solve
