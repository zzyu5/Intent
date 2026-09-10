import torch
import intent
import intent.language as I


@intent.kernel
def _solve_system(
    A: I.In[I.f32, ("N", "N")],
    B: I.In[I.f32, ("N", "K")],
    work: I.Out[I.f32, ("W",)],
    X: I.Out[I.f32, ("N", "K")],
):
    n = A.shape[0]
    width = n + 1
    rows = I.domain(0, n)
    columns = I.domain(0, n)
    rhs_columns = I.domain(0, B.shape[1])

    # Materialize the augmented matrix before any read of the workspace.
    for row in rows:
        row_base = row * width
        for column in columns:
            work[row_base + column] = A[row, column]
        for column in rhs_columns:
            work[row_base + n + column] = B[row, column]

    for pivot in rows:
        pivot_base = pivot * width
        swap_row = pivot
        pivot_value = work[pivot_base + pivot]
        pivot_abs = I.maximum(pivot_value, -pivot_value)

        for candidate in rows[pivot + 1 : n]:
            candidate_value = work[candidate * width + pivot]
            candidate_abs = I.maximum(candidate_value, -candidate_value)
            if candidate_abs > pivot_abs:
                pivot_abs = candidate_abs
                swap_row = candidate

        if swap_row != pivot:
            swap_base = swap_row * width
            for column in columns:
                saved = work[pivot_base + column]
                work[pivot_base + column] = work[swap_base + column]
                work[swap_base + column] = saved
            for column in rhs_columns:
                saved = work[pivot_base + n + column]
                work[pivot_base + n + column] = work[swap_base + n + column]
                work[swap_base + n + column] = saved

        pivot_value = work[pivot_base + pivot]
        for column in columns:
            work[pivot_base + column] = work[pivot_base + column] / pivot_value
        for column in rhs_columns:
            work[pivot_base + n + column] = (
                work[pivot_base + n + column] / pivot_value
            )

        for row in rows:
            if row != pivot:
                row_base = row * width
                factor = work[row_base + pivot]
                for column in columns:
                    work[row_base + column] = (
                        work[row_base + column]
                        - factor * work[pivot_base + column]
                    )
                for column in rhs_columns:
                    work[row_base + n + column] = (
                        work[row_base + n + column]
                        - factor * work[pivot_base + n + column]
                    )

    for row in rows:
        row_base = row * width
        for column in rhs_columns:
            X[row, column] = work[row_base + n + column]


def build(context):
    compiled = context.compile("solve_system", _solve_system, constexprs={})

    def wrapper(A, B, *, left=True, out=None):
        if left:
            result = out if out is not None else torch.empty_like(B)
            work = torch.empty(
                (A.shape[-1] * (A.shape[-1] + 1),),
                device=A.device,
                dtype=A.dtype,
            )
            compiled(A, B, work, result)
            return result

        a_view = A.transpose(-2, -1)
        b_view = B.transpose(-2, -1)
        result = out if out is not None else torch.empty_like(B)
        result_view = result.transpose(-2, -1)
        work = torch.empty(
            (a_view.shape[-1] * (a_view.shape[-1] + 1),),
            device=A.device,
            dtype=A.dtype,
        )
        compiled(a_view, b_view, work, result_view)
        return result

    return wrapper
