import torch
import intent
import intent.language as I


@intent.kernel
def _solve_kernel(
    A: I.In[I.f32, (256, 256)],
    B: I.In[I.f32, (256, 1)],
    W: I.Out[I.f32, (256, 257)],
    X: I.Out[I.f32, (256, 1)],
):
    rows = I.domain(0, 256)
    columns = I.domain(0, 256)

    # Form the augmented matrix before any read of the output workspace.
    W[rows, columns] = A[rows, columns]
    W[rows, 256] = B[rows, 0]

    # Pivoted LU elimination.  The outer loop is ordered by pivot dependency;
    # each trailing rank-one update is independent over its two logical axes.
    for k in range(255):
        active = I.domain(k, 256)
        column = W[active, k]
        _, pivot_relative = I.arg_reduce.max(I.abs(column), axis=0)
        pivot = k + pivot_relative

        if pivot != k:
            whole_row = I.domain(0, 257)
            top = W[k, whole_row]
            bottom = W[pivot, whole_row]
            W[k, whole_row] = bottom
            W[pivot, whole_row] = top

        diagonal = W[k, k]
        below = I.domain(k + 1, 256)
        trailing = I.domain(k + 1, 257)
        factors = W[below, k] / diagonal
        W[below, k] = factors
        pivot_row = W[k, trailing]
        W[below, trailing] = W[below, trailing] - I.outer(factors, pivot_row)

    # The final row has no right-hand-side terms to subtract.
    last = W[255, 256] / W[255, 255]
    W[255, 256] = last

    # Back substitution is a strict recurrence over rows.  Each inner dot is
    # a reduction over the already solved suffix.
    for ordinal in range(255):
        i = 254 - ordinal
        suffix = I.domain(i + 1, 256)
        tail = I.reduce.sum(W[i, suffix] * W[suffix, 256], axis=0, acc_dtype=I.f32)
        value = (W[i, 256] - tail) / W[i, i]
        W[i, 256] = value

    X[rows, 0] = W[rows, 256]


def build(context):
    solve_artifact = context.compile("gaussian_solve", _solve_kernel)

    def solve(A, B):
        workspace = torch.empty((256, 257), device=A.device, dtype=A.dtype)
        output = torch.empty_like(B)
        solve_artifact(A, B, workspace, output)
        return output

    return solve
