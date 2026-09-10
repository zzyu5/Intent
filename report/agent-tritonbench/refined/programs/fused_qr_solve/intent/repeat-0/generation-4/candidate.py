import torch
import intent
import intent.language as I


@intent.kernel
def form_normal_equations(
    A: I.In[I.f32, ("M", "N")],
    b: I.In[I.f32, ("M", "K")],
    gram: I.Out[I.f32, ("N", "N")],
    rhs: I.Out[I.f32, ("N", "K")],
):
    rows = I.domain(0, A.shape[0])
    n = A.shape[1]
    k = b.shape[1]

    for i in I.parallel(I.domain(0, n)):
        for j in I.parallel(I.domain(0, n)):
            gram[i, j] = I.dot(A[rows, i], A[rows, j], acc_dtype=I.f32)
        for r in I.parallel(I.domain(0, k)):
            rhs[i, r] = I.dot(A[rows, i], b[rows, r], acc_dtype=I.f32)


@intent.kernel
def ldl_solve(
    gram: I.InOut[I.f32, ("N", "N")],
    rhs: I.InOut[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("N", "K")],
):
    n = gram.shape[0]
    nrhs = rhs.shape[1]

    for i in I.domain(0, n):
        diagonal = gram[i, i]
        for q in I.domain(0, i):
            lower = gram[i, q]
            diagonal = diagonal - lower * lower * gram[q, q]
        gram[i, i] = diagonal

        for j in I.domain(i + 1, n):
            value = gram[j, i]
            for q in I.domain(0, i):
                value = value - gram[j, q] * gram[i, q] * gram[q, q]
            gram[j, i] = I.fdiv(value, diagonal)

    for i in I.domain(0, n):
        for r in I.domain(0, nrhs):
            value = rhs[i, r]
            for q in I.domain(0, i):
                value = value - gram[i, q] * rhs[q, r]
            rhs[i, r] = value

    for i in I.domain(0, n):
        for r in I.domain(0, nrhs):
            rhs[i, r] = I.fdiv(rhs[i, r], gram[i, i])

    for reverse_i in I.domain(0, n):
        i = n - 1 - reverse_i
        for r in I.domain(0, nrhs):
            value = rhs[i, r]
            for q in I.domain(i + 1, n):
                value = value - gram[q, i] * rhs[q, r]
            rhs[i, r] = value

    for i in I.parallel(I.domain(0, output.shape[0])):
        for r in I.parallel(I.domain(0, output.shape[1])):
            output[i, r] = rhs[i, r]


def build(context):
    form = context.compile("fused_qr_solve_form_normal_equations", form_normal_equations)
    solve = context.compile("fused_qr_solve_ldl", ldl_solve)

    def wrapper(A, b):
        n = A.shape[1]
        nrhs = b.shape[1]

        gram = torch.empty((n, n), dtype=A.dtype, device=A.device)
        rhs = torch.empty((n, nrhs), dtype=b.dtype, device=b.device)
        output = torch.empty((n, nrhs), dtype=b.dtype, device=b.device)

        form(A, b, gram, rhs)
        solve(gram, rhs, output)
        return output

    return wrapper
