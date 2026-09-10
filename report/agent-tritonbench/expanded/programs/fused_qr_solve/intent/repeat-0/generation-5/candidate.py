import intent
import intent.language as I
import torch


@intent.kernel
def _normal_equations(
    matrix: I.In[I.f32, ("M", "N")],
    rhs: I.In[I.f32, ("M", "K")],
    gram: I.Out[I.f32, ("N", "N")],
    cross: I.Out[I.f32, ("N", "K")],
):
    rows = I.domain(0, gram.shape[0])
    columns = I.domain(0, gram.shape[1])
    rhs_columns = I.domain(0, cross.shape[1])

    gram[rows, columns] = I.matmul(
        matrix,
        matrix,
        acc_dtype=I.f32,
        transpose_lhs=True,
    )
    cross[rows, rhs_columns] = I.matmul(
        matrix,
        rhs,
        acc_dtype=I.f32,
        transpose_lhs=True,
    )


@intent.fn
def _sqrt_positive(value):
    zero = I.cast(0, I.f32)
    one = I.cast(1, I.f32)
    half = I.cast(1, I.f32) / I.cast(2, I.f32)

    guess = one
    if value == zero:
        guess = zero
    else:
        if value > one:
            guess = value
        for _ in I.domain(0, 16):
            guess = half * (guess + I.fdiv(value, guess))
    return guess


@intent.kernel
def _cholesky_solve(
    gram: I.InOut[I.f32, ("N", "N")],
    cross: I.InOut[I.f32, ("N", "K")],
    solution: I.InOut[I.f32, ("N", "K")],
):
    n = gram.shape[0]
    k = cross.shape[1]
    rows = I.domain(0, n)
    rhs_columns = I.domain(0, k)

    for column in rows:
        diagonal = gram[column, column]
        for previous in rows[0:column]:
            value = gram[column, previous]
            diagonal = diagonal - value * value
        gram[column, column] = _sqrt_positive(diagonal)

        for row in rows[column + 1:n]:
            value = gram[row, column]
            for previous in rows[0:column]:
                value = value - gram[row, previous] * gram[column, previous]
            gram[row, column] = I.fdiv(value, gram[column, column])

    for row in rows:
        for rhs_column in rhs_columns:
            value = cross[row, rhs_column]
            for previous in rows[0:row]:
                value = value - gram[row, previous] * cross[previous, rhs_column]
            cross[row, rhs_column] = I.fdiv(value, gram[row, row])

    for reverse_row in rows:
        row = n - 1 - reverse_row
        for rhs_column in rhs_columns:
            value = cross[row, rhs_column]
            for later in rows[row + 1:n]:
                value = value - gram[later, row] * solution[later, rhs_column]
            solution[row, rhs_column] = I.fdiv(value, gram[row, row])


def build(context):
    form = context.compile("fused_qr_normal_equations", _normal_equations)
    solve = context.compile("fused_qr_cholesky_solve", _cholesky_solve)

    def wrapper(A, b):
        gram = torch.empty((A.shape[1], A.shape[1]), device=A.device, dtype=A.dtype)
        cross = torch.empty((A.shape[1], b.shape[1]), device=b.device, dtype=b.dtype)
        solution = torch.empty((A.shape[1], b.shape[1]), device=b.device, dtype=b.dtype)
        form(A, b, gram, cross)
        solve(gram, cross, solution)
        return solution

    return wrapper
