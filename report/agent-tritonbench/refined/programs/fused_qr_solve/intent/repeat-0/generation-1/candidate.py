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
    gram_rows = I.domain(0, gram.shape[0])
    gram_cols = I.domain(0, gram.shape[1])
    rhs_rows = I.domain(0, rhs.shape[0])
    rhs_cols = I.domain(0, rhs.shape[1])

    gram[gram_rows, gram_cols] = I.matmul(
        A,
        A,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )
    rhs[rhs_rows, rhs_cols] = I.matmul(
        A,
        b,
        transpose_lhs=True,
        acc_dtype=I.f32,
    )


@intent.kernel
def ldl_solve(
    gram: I.InOut[I.f32, ("N", "N")],
    rhs: I.InOut[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("N", "K")],
):
    n = gram.shape[0]
    nrhs = rhs.shape[1]

    # In-place LDL^T factorization. The lower triangle stores unit-lower L,
    # while the diagonal stores D.
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

    # Forward solve L*y = A^T*b.
    for i in I.domain(0, n):
        for r in I.domain(0, nrhs):
            value = rhs[i, r]
            for q in I.domain(0, i):
                value = value - gram[i, q] * rhs[q, r]
            rhs[i, r] = value

    # Diagonal solve D*z = y.
    for i in I.domain(0, n):
        for r in I.domain(0, nrhs):
            rhs[i, r] = I.fdiv(rhs[i, r], gram[i, i])

    # Backward solve L^T*x = z. A forward domain with a reversed index keeps
    # the logical loop step positive.
    for reverse_i in I.domain(0, n):
        i = n - 1 - reverse_i
        for r in I.domain(0, nrhs):
            value = rhs[i, r]
            for q in I.domain(i + 1, n):
                value = value - gram[q, i] * rhs[q, r]
            rhs[i, r] = value

    rows = I.domain(0, output.shape[0])
    cols = I.domain(0, output.shape[1])
    output[rows, cols] = rhs[rows, cols]


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
