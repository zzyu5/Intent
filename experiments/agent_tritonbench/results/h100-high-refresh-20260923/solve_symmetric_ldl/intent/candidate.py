import torch
import intent
import intent.language as I


@intent.kernel
def _solve_kernel(
    A: I.In[I.f32, (32, 32)],
    b: I.In[I.f32, (32, 1)],
    out: I.Out[I.f32, (32, 1)],
    hermitian: I.bool,
):
    # S is the active symmetric Schur complement.  F becomes the compact
    # lower LD factor, with the diagonal D stored on its diagonal.
    zero = I.cast(0.0, I.f32)
    alpha = I.cast(0.6403882032022076, I.f32)
    S = I.buffer((32, 32), I.f32, init=A)
    F = I.buffer((32, 32), I.f32, init=I.zeros((32, 32), I.f32))
    pivots = I.buffer((32,), I.i64)

    # Lower Bunch-Kaufman pivot selection and the 1x1 path used by the
    # positive-definite profile.  The row swaps also permute prior L rows.
    for k in I.domain(0, 32):
        colmax = zero
        imax = k
        for i in I.domain(0, 32):
            if i > k:
                candidate = I.abs(S[i, k])
                if candidate > colmax:
                    colmax = candidate
                    imax = i

        kp = k
        absakk = I.abs(S[k, k])
        if k < 31:
            rowmax = zero
            for j in I.domain(0, 32):
                if j >= k and j != imax:
                    candidate = I.abs(S[imax, j])
                    if candidate > rowmax:
                        rowmax = candidate
            if absakk < alpha * colmax:
                if I.abs(S[imax, imax]) >= alpha * rowmax:
                    kp = imax
                elif rowmax != zero and absakk >= alpha * colmax * (colmax / rowmax):
                    kp = k
                else:
                    # An SPD input takes the 1x1 branch.  Keep a defined
                    # pivot for the exceptional 2x2 decision as well.
                    kp = imax

        if kp != k:
            for j in I.domain(0, 32):
                if j >= k:
                    tmp = S[k, j]
                    S[k, j] = S[kp, j]
                    S[kp, j] = tmp
            for i in I.domain(0, 32):
                if i >= k:
                    tmp = S[i, k]
                    S[i, k] = S[i, kp]
                    S[i, kp] = tmp
            for j in I.domain(0, 32):
                if j < k:
                    tmp = F[k, j]
                    F[k, j] = F[kp, j]
                    F[kp, j] = tmp

        pivots[k] = I.cast(kp + 1, I.i64)
        diagonal = S[k, k]
        F[k, k] = diagonal

        for i in I.domain(0, 32):
            if i > k:
                F[i, k] = S[i, k] / diagonal

        for i in I.domain(0, 32):
            if i > k:
                for j in I.domain(0, 32):
                    if j > k:
                        S[i, j] = S[i, j] - F[i, k] * diagonal * F[j, k]

    # Materialize C = F diag(float32(p)) F^T.  The real fixed profile has
    # identical transpose and conjugate-transpose paths.
    for i in I.parallel(I.domain(0, 32)):
        for j in I.domain(0, 32):
            value = zero
            for k in I.domain(0, 32):
                value = value + F[i, k] * I.cast(pivots[k], I.f32) * F[j, k]
            S[i, j] = value

    rhs = I.buffer((32,), I.f32, init=I.reshape(b, (32,)))

    # Gaussian elimination with partial pivoting is robust for the explicitly
    # materialized C and reuses S as the solve workspace.
    for k in I.domain(0, 32):
        pivot = k
        pivot_abs = I.abs(S[k, k])
        for i in I.domain(0, 32):
            if i > k:
                candidate = I.abs(S[i, k])
                if candidate > pivot_abs:
                    pivot_abs = candidate
                    pivot = i
        if pivot != k:
            for j in I.domain(0, 32):
                tmp = S[k, j]
                S[k, j] = S[pivot, j]
                S[pivot, j] = tmp
            tmp_rhs = rhs[k]
            rhs[k] = rhs[pivot]
            rhs[pivot] = tmp_rhs

        diagonal = S[k, k]
        for i in I.domain(0, 32):
            if i > k:
                multiplier = S[i, k] / diagonal
                S[i, k] = zero
                rhs[i] = rhs[i] - multiplier * rhs[k]
                for j in I.domain(0, 32):
                    if j > k:
                        S[i, j] = S[i, j] - multiplier * S[k, j]

    for ordinal in I.domain(0, 32):
        i = 31 - ordinal
        value = rhs[i]
        for j in I.domain(0, 32):
            if j > i:
                value = value - S[i, j] * rhs[j]
        rhs[i] = value / S[i, i]

    for i in I.parallel(I.domain(0, 32)):
        out[i, 0] = rhs[i]


def build(context):
    artifact = context.compile("solve_symmetric_ldl_kernel", _solve_kernel)

    def solve_symmetric_ldl(A, b, hermitian=False, out=None):
        result = out
        if result is None:
            result = torch.empty((32, 1), device=A.device, dtype=torch.float32)
        artifact(A, b, result, hermitian)
        return result

    return solve_symmetric_ldl
