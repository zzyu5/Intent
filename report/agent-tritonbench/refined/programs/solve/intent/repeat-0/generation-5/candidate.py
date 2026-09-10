import torch
import intent
import intent.language as I


@intent.kernel
def _solve_system(
    A: I.In[I.f32, ("N", "N")],
    B: I.In[I.f32, ("N", "K")],
    work: I.InOut[I.f32, ("W",)],
    X: I.Out[I.f32, ("N", "K")],
):
    n = A.shape[0]
    width = n + 1
    rows = I.domain(0, n)
    columns = I.domain(0, n)
    rhs_columns = I.domain(0, B.shape[1])

    # Build an augmented matrix in row-major logical storage.
    for row in rows:
        row_base = row * width
        for column in columns:
            work[row_base + column] = A[row, column]
        for column in rhs_columns:
            work[row_base + n + column] = B[row, column]

    # Doolittle elimination without pivot branches. The lower triangle holds
    # the unit-lower factors and the upper triangle holds U.
    for pivot in rows:
        pivot_base = pivot * width
        pivot_value = work[pivot_base + pivot]
        for row in rows[pivot + 1 : n]:
            row_base = row * width
            factor = work[row_base + pivot] / pivot_value
            work[row_base + pivot] = factor
            for column in columns[pivot + 1 : n]:
                work[row_base + column] = (
                    work[row_base + column]
                    - factor * work[pivot_base + column]
                )
            for column in rhs_columns:
                work[row_base + n + column] = (
                    work[row_base + n + column]
                    - factor * work[pivot_base + n + column]
                )

    # The elimination above has already formed y=L^-1*B. Solve U*x=y in
    # reverse row order, expressed with a forward ordinal.
    for reverse in rows:
        row = n - 1 - reverse
        row_base = row * width
        pivot_value = work[row_base + row]
        for column in rhs_columns:
            value = work[row_base + n + column] / pivot_value
            work[row_base + n + column] = value
        for previous in rows[0:row]:
            previous_base = previous * width
            for column in rhs_columns:
                work[previous_base + n + column] = (
                    work[previous_base + n + column]
                    - work[previous_base + row]
                    * work[row_base + n + column]
                )

    for row in rows:
        row_base = row * width
        for column in rhs_columns:
            X[row, column] = work[row_base + n + column]


def build(context):
    compiled = context.compile("solve_system_lu", _solve_system, constexprs={})

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
