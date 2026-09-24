import torch
import intent
import intent.language as I


N = 256
PANEL = 32


@intent.kernel
def initialize(A: I.In[I.f32, (N, N)], B: I.In[I.f32, (N, 1)],
               W: I.Out[I.f32, (N, N + 1)]):
    rows = I.domain(0, N)
    columns = I.domain(0, N)
    W[rows, columns] = A[rows, columns]
    W[rows, N] = B[rows, 0]


@intent.kernel
def factor_panel(W: I.InOut[I.f32, (N, N + 1)],
                 BEGIN: I.Constexpr[int], END: I.Constexpr[int]):
    rows = I.domain(0, N)
    columns = I.domain(0, N + 1)
    for k in rows[BEGIN:END]:
        active = rows[k:N]
        _, relative = I.arg_reduce.max(I.abs(W[active, k]), axis=0)
        pivot = k + relative
        if pivot != k:
            top = W[k, columns]
            bottom = W[pivot, columns]
            W[k, columns] = bottom
            W[pivot, columns] = top
        if k + 1 < N:
            below = rows[k + 1:N]
            multipliers = W[below, k] / W[k, k]
            if k + 1 < END:
                within_panel = columns[k + 1:END]
                pivot_values = W[k, within_panel]
                W[below, within_panel] = (
                    W[below, within_panel] - I.outer(multipliers, pivot_values)
                )
            W[below, k] = multipliers


@intent.kernel
def solve_panel_row(W: I.InOut[I.f32, (N, N + 1)],
                    BEGIN: I.Constexpr[int], END: I.Constexpr[int]):
    rows = I.domain(0, N)
    columns = I.domain(0, N + 1)
    for col in I.parallel(columns[END:N + 1]):
        for row in rows[BEGIN:END]:
            if row > BEGIN:
                previous = rows[BEGIN:row]
                products = W[row, previous] * W[previous, col]
                total = I.reduce.sum(products, axis=0, acc_dtype=I.f32)
                W[row, col] = W[row, col] - total


@intent.kernel
def update_trailing(W: I.InOut[I.f32, (N, N + 1)],
                    BEGIN: I.Constexpr[int], END: I.Constexpr[int]):
    rows = I.domain(0, N)
    columns = I.domain(0, N + 1)
    below = rows[END:N]
    right = columns[END:N + 1]
    panel = rows[BEGIN:END]
    correction = I.matmul(W[below, panel], W[panel, right], acc_dtype=I.f32)
    W[below, right] = W[below, right] - correction


@intent.kernel
def back_substitute(W: I.In[I.f32, (N, N + 1)],
                    X: I.Out[I.f32, (N, 1)]):
    rows = I.domain(0, N)
    indices = I.indices(rows)
    residual = W[rows, N]
    for ordinal in I.domain(0, N):
        row = N - 1 - ordinal
        value = residual[row] / W[row, row]
        X[row, 0] = value
        residual = I.select(indices < row,
                            residual - W[rows, row] * value, residual)


def build(context):
    initialize_artifact = context.compile("blocked_solve_initialize", initialize)
    stages = []
    for begin in range(0, N, PANEL):
        end = min(begin + PANEL, N)
        bindings = {"BEGIN": begin, "END": end}
        factor = context.compile(f"blocked_solve_factor_{begin}", factor_panel,
                                 constexprs=bindings)
        row = context.compile(f"blocked_solve_row_{begin}", solve_panel_row,
                              constexprs=bindings)
        update = None
        if end < N:
            update = context.compile(f"blocked_solve_update_{begin}", update_trailing,
                                     constexprs=bindings)
        stages.append((factor, row, update))
    finish = context.compile("blocked_solve_substitute", back_substitute)

    def solve(A, B):
        workspace = torch.empty((N, N + 1), device=A.device, dtype=A.dtype)
        output = torch.empty_like(B)
        initialize_artifact(A, B, workspace)
        for factor, row, update in stages:
            factor(workspace)
            row(workspace)
            if update is not None:
                update(workspace)
        finish(workspace, output)
        return output

    return solve
