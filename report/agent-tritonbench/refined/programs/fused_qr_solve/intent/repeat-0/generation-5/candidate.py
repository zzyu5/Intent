import torch
import intent
import intent.language as I


@intent.kernel
def fused_normal_equations_ldl(
    A: I.In[I.f32, (512, 256)],
    b: I.In[I.f32, (512, 10)],
    gram: I.Out[I.f32, (256, 256)],
    rhs: I.Out[I.f32, (256, 10)],
    output: I.Out[I.f32, (256, 10)],
    residual: I.Out[I.f32, (512, 10)],
):
    # Form the lower triangle of A^T A and A^T b.  Keeping the reduction as
    # an ordered scalar loop avoids requiring a physical contraction layout.
    for i in I.domain(0, 256):
        for j in I.domain(0, i + 1):
            total = I.cast(0, I.f32)
            for row in I.domain(0, 512):
                total = total + A[row, i] * A[row, j]
            gram[i, j] = total

        for r in I.domain(0, 10):
            total = I.cast(0, I.f32)
            for row in I.domain(0, 512):
                total = total + A[row, i] * b[row, r]
            rhs[i, r] = total

    # In-place LDL^T factorization.  The lower triangle stores unit-lower L
    # and the diagonal stores D.
    for i in I.domain(0, 256):
        diagonal = gram[i, i]
        for q in I.domain(0, i):
            lower = gram[i, q]
            diagonal = diagonal - lower * lower * gram[q, q]
        gram[i, i] = diagonal

        for j in I.domain(i + 1, 256):
            value = gram[j, i]
            for q in I.domain(0, i):
                value = value - gram[j, q] * gram[i, q] * gram[q, q]
            gram[j, i] = I.fdiv(value, diagonal)

    # First solve G x = A^T b, using rhs as the mutable solve workspace.
    for i in I.domain(0, 256):
        for r in I.domain(0, 10):
            value = rhs[i, r]
            for q in I.domain(0, i):
                value = value - gram[i, q] * rhs[q, r]
            rhs[i, r] = value

    for i in I.domain(0, 256):
        for r in I.domain(0, 10):
            rhs[i, r] = I.fdiv(rhs[i, r], gram[i, i])

    for reverse_i in I.domain(0, 256):
        i = 255 - reverse_i
        for r in I.domain(0, 10):
            value = rhs[i, r]
            for q in I.domain(i + 1, 256):
                value = value - gram[q, i] * rhs[q, r]
            rhs[i, r] = value

    for i in I.domain(0, 256):
        for r in I.domain(0, 10):
            output[i, r] = rhs[i, r]

    # Iterative refinement against the original least-squares residual makes
    # the normal-equation solve track the QR result more closely.
    for refinement in I.domain(0, 2):
        for row in I.domain(0, 512):
            for r in I.domain(0, 10):
                value = b[row, r]
                for j in I.domain(0, 256):
                    value = value - A[row, j] * output[j, r]
                residual[row, r] = value

        for i in I.domain(0, 256):
            for r in I.domain(0, 10):
                correction = I.cast(0, I.f32)
                for row in I.domain(0, 512):
                    correction = correction + A[row, i] * residual[row, r]
                rhs[i, r] = correction

        for i in I.domain(0, 256):
            for r in I.domain(0, 10):
                value = rhs[i, r]
                for q in I.domain(0, i):
                    value = value - gram[i, q] * rhs[q, r]
                rhs[i, r] = value

        for i in I.domain(0, 256):
            for r in I.domain(0, 10):
                rhs[i, r] = I.fdiv(rhs[i, r], gram[i, i])

        for reverse_i in I.domain(0, 256):
            i = 255 - reverse_i
            for r in I.domain(0, 10):
                value = rhs[i, r]
                for q in I.domain(i + 1, 256):
                    value = value - gram[q, i] * rhs[q, r]
                rhs[i, r] = value

        for i in I.domain(0, 256):
            for r in I.domain(0, 10):
                output[i, r] = output[i, r] + rhs[i, r]


def build(context):
    compiled = context.compile(
        "fused_qr_solve_normal_equations_ldl",
        fused_normal_equations_ldl,
    )

    def wrapper(A, b):
        gram = torch.empty((256, 256), dtype=A.dtype, device=A.device)
        rhs = torch.empty((256, 10), dtype=b.dtype, device=b.device)
        output = torch.empty((256, 10), dtype=b.dtype, device=b.device)
        residual = torch.empty((512, 10), dtype=b.dtype, device=b.device)
        compiled(A, b, gram, rhs, output, residual)
        return output

    return wrapper
