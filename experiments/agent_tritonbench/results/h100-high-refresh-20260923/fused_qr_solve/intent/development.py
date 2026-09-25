import torch
import intent
import intent.language as I


M = 512
N = 256
RHS = 10
PANEL = 16


@intent.kernel
def initialize(A: I.In[I.f32, (M, N)], V: I.Out[I.f32, (M, N)],
               R: I.Out[I.f32, (N, N)]):
    V[:, :] = A[:, :]
    R[:, :] = I.zeros((N, N), dtype=I.f32)


@intent.kernel
def factor_panel(V: I.InOut[I.f32, (M, N)], R: I.InOut[I.f32, (N, N)],
                 BEGIN: I.index, END: I.index):
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for column in columns[BEGIN:END]:
        values = V[rows, column]
        length = I.sqrt(I.reduce.sum(values * values, axis=0, acc_dtype=I.f32))
        normalized = values / length
        V[rows, column] = normalized
        R[column, column] = length
        if column + 1 < END:
            remaining = columns[column + 1:END]
            block = V[rows, remaining]
            projection = I.reduce.sum(
                I.reshape(normalized, (M, 1)) * block, axis=0, acc_dtype=I.f32)
            R[column, remaining] = projection
            V[rows, remaining] = block - I.outer(normalized, projection)


@intent.kernel
def project_panel(V: I.In[I.f32, (M, N)], R: I.InOut[I.f32, (N, N)],
                  BEGIN: I.index, END: I.index):
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    panel = columns[BEGIN:END]
    remaining = columns[END:N]
    R[panel, remaining] = I.matmul(V[rows, panel], V[rows, remaining],
                                   transpose_lhs=True, acc_dtype=I.f32)


@intent.kernel
def update_trailing(V: I.InOut[I.f32, (M, N)], R: I.In[I.f32, (N, N)],
                    BEGIN: I.index, END: I.index):
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    panel = columns[BEGIN:END]
    remaining = columns[END:N]
    correction = I.matmul(V[rows, panel], R[panel, remaining], acc_dtype=I.f32)
    V[rows, remaining] = V[rows, remaining] - correction


@intent.kernel
def project_rhs(Q: I.In[I.f32, (M, N)], B: I.In[I.f32, (M, RHS)],
                Y: I.Out[I.f32, (N, RHS)]):
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    rhs = I.domain(0, RHS)
    Y[columns, rhs] = I.matmul(Q[rows, columns], B[rows, rhs],
                               transpose_lhs=True, acc_dtype=I.f32)


@intent.kernel
def back_substitute(R: I.In[I.f32, (N, N)], Y: I.In[I.f32, (N, RHS)],
                    X: I.Out[I.f32, (N, RHS)]):
    rows = I.domain(0, N)
    rhs = I.domain(0, RHS)
    indices = I.indices(rows)
    residual = Y[rows, rhs]
    for ordinal in I.domain(0, N):
        row = N - 1 - ordinal
        value = residual[row, rhs] / R[row, row]
        X[row, rhs] = value
        residual = I.select(I.reshape(indices < row, (N, 1)),
                            residual - I.outer(R[rows, row], value), residual)


def build(context):
    setup = context.compile("blocked_qr_initialize", initialize)
    factor = context.compile("blocked_qr_factor", factor_panel)
    project = context.compile("blocked_qr_project", project_panel)
    update = context.compile("blocked_qr_update", update_trailing)
    project_b = context.compile("blocked_qr_project_rhs", project_rhs)
    substitute = context.compile("blocked_qr_substitute", back_substitute)

    def fused_qr_solve(A, b):
        vectors = torch.empty_like(A)
        triangular = torch.empty((N, N), dtype=A.dtype, device=A.device)
        transformed = torch.empty((N, RHS), dtype=b.dtype, device=b.device)
        output = torch.empty_like(transformed)
        setup(A, vectors, triangular)
        for begin in range(0, N, PANEL):
            end = min(begin + PANEL, N)
            factor(vectors, triangular, begin, end)
            if end < N:
                project(vectors, triangular, begin, end)
                update(vectors, triangular, begin, end)
        project_b(vectors, b, transformed)
        substitute(triangular, transformed, output)
        return output

    return fused_qr_solve
