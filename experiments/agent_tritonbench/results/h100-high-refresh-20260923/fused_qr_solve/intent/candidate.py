import torch
import intent
import intent.language as I


@intent.kernel
def qr_factor(
    A: I.In[I.f32, (512, 256)],
    b: I.In[I.f32, (512, 10)],
    R: I.Out[I.f32, (512, 256)],
    Y: I.Out[I.f32, (512, 10)],
):
    rows = I.domain(0, 512)
    cols = I.domain(0, 256)
    rhs_cols = I.domain(0, 10)

    # Materialize the inputs in the cross-kernel workspaces first.  The
    # reflector loop below has an ordered dependency through R and Y.
    R[rows, cols] = A[rows, cols]
    Y[rows, rhs_cols] = b[rows, rhs_cols]

    zero = I.cast(0.0, I.f32)
    two = I.cast(2.0, I.f32)
    for j in I.domain(0, 256):
        tail_rows = I.domain(j, 512)
        column = R[tail_rows, j]
        norm_sq = I.reduce.sum(column * column, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm_sq)
        leading = R[j, j]

        # Choosing the sign this way avoids cancellation in the first
        # Householder vector component.
        if leading >= zero:
            alpha = -norm
        else:
            alpha = norm
        R[j, j] = leading - alpha

        vector = R[tail_rows, j]
        vector_sq = I.reduce.sum(vector * vector, axis=0, acc_dtype=I.f32)
        beta = I.fdiv(two, vector_sq)

        # Each strict trailing matrix column is independent for this
        # reflector.  The current column now contains v, so it is finalized
        # explicitly after the update rather than being transformed again.
        for c in I.parallel(I.domain(j + 1, 256)):
            trailing_column = R[tail_rows, c]
            projection = I.reduce.sum(
                vector * trailing_column, axis=0, acc_dtype=I.f32
            )
            R[tail_rows, c] = trailing_column - (beta * projection) * vector

        R[j, j] = alpha
        R[I.domain(j + 1, 512), j] = zero

        # The same reflector is applied independently to every RHS column.
        for rhs in I.parallel(rhs_cols):
            rhs_column = Y[tail_rows, rhs]
            projection = I.reduce.sum(
                vector * rhs_column, axis=0, acc_dtype=I.f32
            )
            Y[tail_rows, rhs] = rhs_column - (beta * projection) * vector


@intent.kernel
def triangular_solve(
    R: I.In[I.f32, (512, 256)],
    Y: I.In[I.f32, (512, 10)],
    X: I.Out[I.f32, (256, 10)],
):
    zero = I.cast(0.0, I.f32)
    for rhs in I.parallel(I.domain(0, 10)):
        # Reverse ordinal expresses the true dependency: x[i] consumes
        # entries x[i+1:] that were produced by earlier iterations.
        for ordinal in I.domain(0, 256):
            i = 255 - ordinal
            if i + 1 < 256:
                tail = I.domain(i + 1, 256)
                row = R[i, tail]
                solved_tail = X[tail, rhs]
                correction = I.reduce.sum(
                    row * solved_tail, axis=0, acc_dtype=I.f32
                )
            else:
                correction = zero
            X[i, rhs] = I.fdiv(Y[i, rhs] - correction, R[i, i])


def build(context):
    factor_artifact = context.compile("qr_factor", qr_factor)
    solve_artifact = context.compile("triangular_solve", triangular_solve)

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        R = torch.empty((512, 256), device=A.device, dtype=A.dtype)
        Y = torch.empty((512, 10), device=b.device, dtype=b.dtype)
        X = torch.empty((256, 10), device=A.device, dtype=A.dtype)
        factor_artifact(A, b, R, Y)
        solve_artifact(R, Y, X)
        return X

    return fused_qr_solve
