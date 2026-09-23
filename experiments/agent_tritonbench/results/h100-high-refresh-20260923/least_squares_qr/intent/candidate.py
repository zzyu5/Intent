import torch
import intent
import intent.language as I


def build(context):
    @intent.kernel
    def householder_qr(
        A: I.In[I.f32, (64, 16)],
        b: I.In[I.f32, (64, 1)],
        R: I.Out[I.f32, (64, 16)],
        qtb: I.Out[I.f32, (64, 1)],
    ):
        rows = I.domain(0, 64)
        cols = I.domain(0, 16)
        rhs = I.domain(0, 1)

        # Materialize the inputs first because the factorization updates R and qtb.
        for i in I.parallel(rows):
            for j in cols:
                R[i, j] = A[i, j]
        for i in I.parallel(rows):
            for j in rhs:
                qtb[i, j] = b[i, j]

        # Each reflector depends on the preceding reflector, so this outer loop is
        # deliberately ordered. The trailing-column and RHS updates are independent
        # logical work inside each reflector.
        for k in cols:
            active_rows = rows[k:]
            trailing_cols = cols[k + 1 :]

            xkk = R[k, k]
            norm_sq = I.cast(0.0, I.f32)
            for i in active_rows:
                v = R[i, k]
                norm_sq = norm_sq + v * v
            norm = I.sqrt(norm_sq)

            if xkk >= 0.0:
                alpha = -norm
            else:
                alpha = norm

            v0 = xkk - alpha
            R[k, k] = v0
            vv = v0 * v0
            for i in rows[k + 1 :]:
                vi = R[i, k]
                vv = vv + vi * vi

            tau = I.cast(0.0, I.f32)
            if vv > 0.0:
                tau = I.cast(2.0, I.f32) / vv

            if tau > 0.0:
                for j in I.parallel(trailing_cols):
                    dot = I.cast(0.0, I.f32)
                    for i in active_rows:
                        dot = dot + R[i, k] * R[i, j]
                    scale = tau * dot
                    for i in I.parallel(active_rows):
                        R[i, j] = R[i, j] - scale * R[i, k]

                dot_b = I.cast(0.0, I.f32)
                for i in active_rows:
                    dot_b = dot_b + R[i, k] * qtb[i, 0]
                scale_b = tau * dot_b
                for i in I.parallel(active_rows):
                    qtb[i, 0] = qtb[i, 0] - scale_b * R[i, k]

            # The reflector vector is no longer needed after all trailing updates.
            R[k, k] = alpha
            for i in I.parallel(rows[k + 1 :]):
                R[i, k] = I.cast(0.0, I.f32)

    @intent.kernel
    def triangular_solve(
        R: I.In[I.f32, (64, 16)],
        qtb: I.In[I.f32, (64, 1)],
        x: I.Out[I.f32, (16, 1)],
    ):
        cols = I.domain(0, 16)
        # Reverse ordinal traversal expresses the dependency x[i] <- x[i+1:].
        for ordinal in cols:
            i = 15 - ordinal
            total = I.cast(0.0, I.f32)
            for j in cols[i + 1 :]:
                total = total + R[i, j] * x[j, 0]
            x[i, 0] = (qtb[i, 0] - total) / R[i, i]

    qr_artifact = context.compile("least_squares_qr_householder", householder_qr)
    solve_artifact = context.compile("least_squares_qr_backsolve", triangular_solve)

    def least_squares_qr(
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        mode: str = "reduced",
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        del mode
        R = torch.empty_like(A)
        qtb = torch.empty_like(b)
        if out is None:
            result = torch.empty((16, 1), device=A.device, dtype=A.dtype)
        else:
            result = out

        qr_artifact(A, b, R, qtb)
        solve_artifact(R, qtb, result)
        return result

    return least_squares_qr
