import torch
import intent
import intent.language as I


M = 512
N = 256
K = 10


@intent.kernel
def householder_factor(
    A: I.In[I.f32, (M, N)],
    R: I.Out[I.f32, (M, N)],
    V: I.Out[I.f32, (M, N)],
    tau: I.Out[I.f32, (N,)],
):
    rows = I.domain(0, M)
    cols = I.domain(0, N)

    # Materialize the input as the working R factor and initialize the
    # reflector storage before the ordered factorization recurrence.
    for row in I.parallel(rows):
        for col in I.parallel(cols):
            R[row, col] = A[row, col]
            V[row, col] = I.cast(0.0, I.f32)

    for j in I.domain(0, N):
        tail = rows[j:M]
        v_old = R[tail, j]
        norm_sq = I.reduce.sum(v_old * v_old, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm_sq)
        x0 = R[j, j]
        zero = I.cast(0.0, I.f32)
        neg_one = I.cast(-1.0, I.f32)
        pos_one = I.cast(1.0, I.f32)
        sign = I.select(x0 >= zero, neg_one, pos_one)
        alpha = sign * norm
        v0 = x0 - alpha
        v = I.select(I.indices(tail) == j, v0, v_old)

        R[j, j] = alpha
        V[tail, j] = v

        tau_j = zero
        if norm_sq != zero:
            tau_j = I.cast(2.0, I.f32) / I.reduce.sum(v * v, axis=0, acc_dtype=I.f32)
        tau[j] = tau_j

        if j + 1 < N:
            right = cols[j + 1:N]
            for col in I.parallel(right):
                column = R[tail, col]
                projection = I.reduce.sum(v * column, axis=0, acc_dtype=I.f32)
                R[tail, col] = column - (tau_j * projection) * v


@intent.kernel
def apply_qt(
    V: I.In[I.f32, (M, N)],
    tau: I.In[I.f32, (N,)],
    b: I.In[I.f32, (M, K)],
    Y: I.Out[I.f32, (M, K)],
):
    rows = I.domain(0, M)
    rhs = I.domain(0, K)

    for row in I.parallel(rows):
        for col in I.parallel(rhs):
            Y[row, col] = b[row, col]

    for j in I.domain(0, N):
        tail = rows[j:M]
        v = V[tail, j]
        tau_j = tau[j]
        for col in I.parallel(rhs):
            values = Y[tail, col]
            projection = I.reduce.sum(v * values, axis=0, acc_dtype=I.f32)
            Y[tail, col] = values - (tau_j * projection) * v


@intent.kernel
def back_substitute(
    R: I.In[I.f32, (M, N)],
    Y: I.In[I.f32, (M, K)],
    X: I.Out[I.f32, (N, K)],
):
    cols = I.domain(0, N)
    rhs = I.domain(0, K)

    for ordinal in I.domain(0, N):
        row = N - 1 - ordinal
        if row + 1 < N:
            right = cols[row + 1:N]
            for col in I.parallel(rhs):
                products = R[row, right] * X[right, col]
                tail_sum = I.reduce.sum(products, axis=0, acc_dtype=I.f32)
                X[row, col] = (Y[row, col] - tail_sum) / R[row, row]
        else:
            for col in I.parallel(rhs):
                X[row, col] = Y[row, col] / R[row, row]


def build(context):
    factor = context.compile("fused_qr_householder_factor", householder_factor)
    apply = context.compile("fused_qr_apply_qt", apply_qt)
    solve = context.compile("fused_qr_back_substitute", back_substitute)

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        R = torch.empty((M, N), device=A.device, dtype=A.dtype)
        V = torch.empty((M, N), device=A.device, dtype=A.dtype)
        tau = torch.empty((N,), device=A.device, dtype=A.dtype)
        Y = torch.empty((M, K), device=b.device, dtype=b.dtype)
        X = torch.empty((N, K), device=A.device, dtype=A.dtype)

        factor(A, R, V, tau)
        apply(V, tau, b, Y)
        solve(R, Y, X)
        return X

    return fused_qr_solve
