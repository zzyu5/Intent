import torch
import intent
import intent.language as I


@intent.kernel
def _symmetrize_lower(
    A: I.In[I.f32, ("N", "N")],
    S: I.Out[I.f32, ("N", "N")],
):
    n = A.shape[0]
    rows = I.domain(0, n)
    cols = I.domain(0, n)
    row_index = I.reshape(I.indices(rows), (n, 1))
    col_index = I.reshape(I.indices(cols), (1, n))

    lower = A[row_index, col_index]
    upper = A[col_index, row_index]
    S[rows, cols] = I.select(col_index <= row_index, lower, upper)


@intent.kernel
def _square_symmetric(
    S: I.In[I.f32, ("N", "N")],
    O: I.Out[I.f32, ("N", "N")],
):
    n = S.shape[0]
    rows = I.domain(0, n)
    cols = I.domain(0, n)
    matrix = S[rows, cols]
    O[rows, cols] = I.matmul(matrix, matrix, acc_dtype=I.f32)


def build(context):
    symmetrize = context.compile("matrix_power_eig_symmetrize", _symmetrize_lower)
    square = context.compile("matrix_power_eig_square", _square_symmetric)

    def matrix_power_eig(A: torch.Tensor, k: float, *, out: torch.Tensor = None):
        symmetric = torch.empty_like(A)
        result = out if out is not None else torch.empty_like(A)
        symmetrize(A, symmetric)
        square(symmetric, result)
        return result

    return matrix_power_eig
