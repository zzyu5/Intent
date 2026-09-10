import torch
import intent
import intent.language as I


@intent.kernel
def _copy_inputs(
    a: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N", "K")],
    work: I.Out[I.f32, ("N", "N")],
    rhs: I.Out[I.f32, ("N", "K")],
):
    a_rows = I.domain(0, a.shape[0])
    a_cols = I.domain(0, a.shape[1])
    b_rows = I.domain(0, b.shape[0])
    b_cols = I.domain(0, b.shape[1])
    work[a_rows, a_cols] = a[a_rows, a_cols]
    rhs[b_rows, b_cols] = b[b_rows, b_cols]


@intent.kernel
def _gauss_jordan(
    work: I.InOut[I.f32, ("N", "N")],
    rhs: I.InOut[I.f32, ("N", "K")],
):
    rows = I.domain(0, work.shape[0])
    columns = I.domain(0, work.shape[1])
    rhs_columns = I.domain(0, rhs.shape[1])

    for pivot in rows:
        swap_row = pivot
        pivot_abs = I.maximum(work[pivot, pivot], -work[pivot, pivot])
        for candidate in rows[pivot + 1 : work.shape[0]]:
            candidate_value = work[candidate, pivot]
            candidate_abs = I.maximum(candidate_value, -candidate_value)
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                swap_row = candidate

        if swap_row != pivot:
            for column in columns:
                saved = work[pivot, column]
                work[pivot, column] = work[swap_row, column]
                work[swap_row, column] = saved
            for column in rhs_columns:
                saved = rhs[pivot, column]
                rhs[pivot, column] = rhs[swap_row, column]
                rhs[swap_row, column] = saved

        pivot_value = work[pivot, pivot]
        work[pivot, columns] = work[pivot, columns] / pivot_value
        rhs[pivot, rhs_columns] = rhs[pivot, rhs_columns] / pivot_value

        for row in rows:
            if row != pivot:
                factor = work[row, pivot]
                work[row, columns] = work[row, columns] - factor * work[pivot, columns]
                rhs[row, rhs_columns] = rhs[row, rhs_columns] - factor * rhs[pivot, rhs_columns]


def build(context):
    copy_inputs = context.compile("solve_copy_inputs", _copy_inputs, constexprs={})
    gauss_jordan = context.compile("solve_gauss_jordan", _gauss_jordan, constexprs={})

    def wrapper(A, B, *, left=True, out=None):
        if left:
            a_view = A
            b_view = B
            result = out if out is not None else torch.empty_like(B)
            result_view = result
        else:
            a_view = A.transpose(-2, -1)
            b_view = B.transpose(-2, -1)
            result = out if out is not None else torch.empty_like(B)
            result_view = result.transpose(-2, -1)

        work = torch.empty_like(a_view)
        copy_inputs(a_view, b_view, work, result_view)
        gauss_jordan(work, result_view)
        return result

    return wrapper
