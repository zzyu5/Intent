import torch
import intent
import intent.language as I


@intent.kernel
def solve_kernel(
    matrix: I.In[I.f32, ("N", "N")],
    rhs: I.In[I.f32, ("N", "K")],
    work: I.InOut[I.f32, ("N", "N")],
    rhs_work: I.InOut[I.f32, ("N", "K")],
    result: I.Out[I.f32, ("N", "K")],
):
    n = matrix.shape[0]
    rows = I.domain(0, n)
    cols = I.domain(0, n)
    rhs_cols = I.domain(0, rhs.shape[1])
    zero = I.cast(0.0, I.f32)

    for row in rows:
        for col in cols:
            work[row, col] = matrix[row, col]
        for col in rhs_cols:
            rhs_work[row, col] = rhs[row, col]

    # Eliminate both sides of each pivot.  Domains derived from k keep the
    # logical row coordinates compatible with the pivot coordinate.
    for k in rows:
        pivot_row = k
        pivot_abs = zero
        for row in I.domain(k, n):
            candidate = work[row, k]
            candidate_abs = -candidate if candidate < zero else candidate
            better = candidate_abs > pivot_abs
            pivot_abs = candidate_abs if better else pivot_abs
            pivot_row = row if better else pivot_row

        # Swapping a row with itself is harmless, so no conditional is needed.
        for col in cols:
            saved = work[k, col]
            work[k, col] = work[pivot_row, col]
            work[pivot_row, col] = saved
        for col in rhs_cols:
            saved = rhs_work[k, col]
            rhs_work[k, col] = rhs_work[pivot_row, col]
            rhs_work[pivot_row, col] = saved

        pivot = work[k, k]
        for col in cols:
            work[k, col] = work[k, col] / pivot
        for col in rhs_cols:
            rhs_work[k, col] = rhs_work[k, col] / pivot

        for row in I.domain(0, k):
            factor = work[row, k]
            for col in cols:
                work[row, col] = work[row, col] - factor * work[k, col]
            for col in rhs_cols:
                rhs_work[row, col] = rhs_work[row, col] - factor * rhs_work[k, col]

        for row in I.domain(k + 1, n):
            factor = work[row, k]
            for col in cols:
                work[row, col] = work[row, col] - factor * work[k, col]
            for col in rhs_cols:
                rhs_work[row, col] = rhs_work[row, col] - factor * rhs_work[k, col]

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
