import intent
import intent.language as I
import torch


@intent.fn
def _sqrt_positive(value):
    one = I.cast(1, I.f32)
    half = I.cast(1, I.f32) / I.cast(2, I.f32)
    guess = one
    if value > one:
        guess = value
    for _ in I.domain(0, 16):
        guess = half * (guess + I.fdiv(value, guess))
    return guess


@intent.kernel
def _householder_qr(
    matrix: I.In[I.f32, ("M", "N")],
    rhs: I.In[I.f32, ("M", "K")],
    work_matrix: I.InOut[I.f32, ("M", "N")],
    work_rhs: I.InOut[I.f32, ("M", "K")],
):
    m = matrix.shape[0]
    n = matrix.shape[1]
    k = rhs.shape[1]
    rows = I.domain(0, m)
    columns = I.domain(0, n)
    rhs_columns = I.domain(0, k)
    zero = I.cast(0, I.f32)
    two = I.cast(2, I.f32)

    work_matrix[rows, columns] = matrix[rows, columns]
    work_rhs[rows, rhs_columns] = rhs[rows, rhs_columns]

    for column in columns:
        norm_squared = zero
        for row in rows[column:m]:
            value = work_matrix[row, column]
            norm_squared = norm_squared + value * value

        if norm_squared > zero:
            norm = _sqrt_positive(norm_squared)
            leading = work_matrix[column, column]
            if leading >= zero:
                alpha = -norm
            else:
                alpha = norm
            first = leading - alpha

            reflector_norm = first * first
            for row in rows[column + 1:m]:
                value = work_matrix[row, column]
                reflector_norm = reflector_norm + value * value

            for target_column in columns[column + 1:n]:
                projection = first * work_matrix[column, target_column]
                for row in rows[column + 1:m]:
                    projection = projection + work_matrix[row, column] * work_matrix[row, target_column]
                factor = I.fdiv(two * projection, reflector_norm)
                work_matrix[column, target_column] = (
                    work_matrix[column, target_column] - factor * first
                )
                for row in rows[column + 1:m]:
                    work_matrix[row, target_column] = (
                        work_matrix[row, target_column]
                        - factor * work_matrix[row, column]
                    )

            for rhs_column in rhs_columns:
                projection = first * work_rhs[column, rhs_column]
                for row in rows[column + 1:m]:
                    projection = projection + work_matrix[row, column] * work_rhs[row, rhs_column]
                factor = I.fdiv(two * projection, reflector_norm)
                work_rhs[column, rhs_column] = (
                    work_rhs[column, rhs_column] - factor * first
                )
                for row in rows[column + 1:m]:
                    work_rhs[row, rhs_column] = (
                        work_rhs[row, rhs_column]
                        - factor * work_matrix[row, column]
                    )

            work_matrix[column, column] = alpha
            for row in rows[column + 1:m]:
                work_matrix[row, column] = zero
        else:
            work_matrix[column, column] = zero
            for row in rows[column + 1:m]:
                work_matrix[row, column] = zero


@intent.kernel
def _back_substitute(
    factor: I.In[I.f32, ("M", "N")],
    transformed_rhs: I.In[I.f32, ("M", "K")],
    solution: I.InOut[I.f32, ("N", "K")],
):
    n = factor.shape[1]
    k = transformed_rhs.shape[1]
    rows = I.domain(0, n)
    rhs_columns = I.domain(0, k)

    for reverse_row in rows:
        row = n - 1 - reverse_row
        for rhs_column in rhs_columns:
            value = transformed_rhs[row, rhs_column]
            for later_row in rows[row + 1:n]:
                value = value - factor[row, later_row] * solution[later_row, rhs_column]
            solution[row, rhs_column] = I.fdiv(value, factor[row, row])


def build(context):
    qr = context.compile("fused_qr_householder", _householder_qr)
    solve = context.compile("fused_qr_back_substitute", _back_substitute)

    def wrapper(A, b):
        work_matrix = torch.empty_like(A)
        work_rhs = torch.empty_like(b)
        solution = torch.empty(
            (A.shape[1], b.shape[1]),
            device=A.device,
            dtype=A.dtype,
        )
        qr(A, b, work_matrix, work_rhs)
        solve(work_matrix, work_rhs, solution)
        return solution

    return wrapper
