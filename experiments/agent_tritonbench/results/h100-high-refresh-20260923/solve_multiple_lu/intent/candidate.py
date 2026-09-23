import torch
import intent
import intent.language as I


@intent.kernel
def _solve_lu(
    A: I.In[I.f32, (64, 64)],
    Bs: I.In[I.f32, (64, 64)],
    X: I.Out[I.f32, (64, 64)],
    PIVOT: I.Constexpr[bool],
):
    # The local snapshots make the documented out=Bs alias case independent
    # of the order in which solution columns are written.
    lu = I.buffer((64, 64), I.f32, init=A)
    rhs = I.buffer((64, 64), I.f32, init=Bs)
    sol = I.buffer((64, 64), I.f32)

    # Doolittle LU.  The outer factorization loop is ordered because every
    # panel depends on the preceding panel's updates.
    for k in I.domain(0, 64):
        if PIVOT:
            pivot = k
            best = I.abs(lu[k, k])
            for candidate in I.domain(k + 1, 64):
                magnitude = I.abs(lu[candidate, k])
                if magnitude > best:
                    best = magnitude
                    pivot = candidate

            # Apply each row interchange to both U/L storage and the RHS.
            for col in I.domain(0, 64):
                tmp = lu[k, col]
                lu[k, col] = lu[pivot, col]
                lu[pivot, col] = tmp
                tmp = rhs[k, col]
                rhs[k, col] = rhs[pivot, col]
                rhs[pivot, col] = tmp

        diagonal = lu[k, k]
        for row in I.domain(k + 1, 64):
            factor = lu[row, k] / diagonal
            lu[row, k] = factor
            for col in I.domain(k + 1, 64):
                lu[row, col] = lu[row, col] - factor * lu[k, col]

    # Each right-hand side is independent after factorization.  Within a
    # column, forward and backward substitution remain strictly ordered.
    for col in I.parallel(I.domain(0, 64)):
        for row in I.domain(0, 64):
            value = rhs[row, col]
            for j in I.domain(0, row):
                value = value - lu[row, j] * sol[j, col]
            sol[row, col] = value

        for ordinal in I.domain(0, 64):
            row = 63 - ordinal
            value = sol[row, col]
            for j in I.domain(row + 1, 64):
                value = value - lu[row, j] * sol[j, col]
            sol[row, col] = value / lu[row, row]

        for row in I.domain(0, 64):
            X[row, col] = sol[row, col]


def build(context):
    pivoted = context.compile(
        "solve_multiple_lu_pivoted",
        _solve_lu,
        constexprs={"PIVOT": True},
    )
    unpivoted = context.compile(
        "solve_multiple_lu_unpivoted",
        _solve_lu,
        constexprs={"PIVOT": False},
    )

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        if out is None:
            out = torch.empty_like(Bs)
        artifact = pivoted if pivot else unpivoted
        artifact(A, Bs, out)
        return out

    return solve_multiple_lu
