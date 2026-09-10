import torch
import intent
import intent.language as I


@intent.kernel
def solve_kernel(
    matrix: I.In[I.f32, ("N", "N")],
    rhs: I.In[I.f32, ("N", "K")],
    work: I.Out[I.f32, ("N", "N")],
    rhs_work: I.Out[I.f32, ("N", "K")],
    result: I.Out[I.f32, ("N", "K")],
):
    n = matrix.shape[0]
    rows = I.domain(0, n)
    cols = I.domain(0, n)
    rhs_cols = I.domain(0, rhs.shape[1])

    for row in rows:
        for col in cols:
            work[row, col] = matrix[row, col]
        for col in rhs_cols:
            rhs_work[row, col] = rhs[row, col]

    # Partial pivoting makes the factorization valid for any nonsingular input
    # whose values are representable in the requested dtype.
    for k in rows:
        pivot_row = k
        pivot_abs = 0.0
        for row in I.domain(k, n):
            candidate = work[row, k]
            candidate_abs = -candidate if candidate < 0.0 else candidate
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                pivot_row = row

        if pivot_row != k:
            for col in cols:
                saved = work[k, col]
                work[k, col] = work[pivot_row, col]
                work[pivot_row, col] = saved
            for col in rhs_cols:
                saved = rhs_work[k, col]
                rhs_work[k, col] = rhs_work[pivot_row, col]
                rhs_work[pivot_row, col] = saved

        pivot = work[k, k]
        for row in I.domain(k + 1, n):
            factor = work[row, k] / pivot
            work[row, k] = factor
            for col in I.domain(k + 1, n):
                work[row, col] = work[row, col] - factor * work[k, col]
            for col in rhs_cols:
                rhs_work[row, col] = rhs_work[row, col] - factor * rhs_work[k, col]

    for reverse in rows:
        k = n - 1 - reverse
        for col in rhs_cols:
            value = rhs_work[k, col]
            for upper_col in I.domain(k + 1, n):
                value = value - work[k, upper_col] * rhs_work[upper_col, col]
            rhs_work[k, col] = value / work[k, k]

    for row in rows:
        for col in rhs_cols:
            result[row, col] = rhs_work[row, col]


def build(context):
    compiled = context.compile("solve", solve_kernel)

    def wrapper(A, B, *, left=True, out=None):
        if left:
            matrix = A
            rhs = B
            output = out if out is not None else torch.empty_like(B)
            output_view = output
        else:
            matrix = A.transpose(-2, -1)
            rhs = B.transpose(-2, -1)
            output = out if out is not None else torch.empty_like(B)
            output_view = output.transpose(-2, -1)

        work = torch.empty_like(matrix)
        rhs_work = torch.empty_like(rhs)
        compiled(matrix, rhs, work, rhs_work, output_view)
        return output

    return wrapper
