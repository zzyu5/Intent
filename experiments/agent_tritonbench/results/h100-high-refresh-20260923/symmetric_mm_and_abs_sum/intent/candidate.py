import torch
import intent
import intent.language as I


@intent.kernel
def _symmetric_mm(
    a: I.In[I.f32, ("N", "K")],
    c: I.In[I.f32, ("N", "N")],
    d: I.Out[I.f32, ("N", "N")],
    alpha: I.f32,
    beta: I.f32,
):
    n, k = a.shape
    rows = I.domain(0, n)
    inner = I.domain(0, k)
    columns = I.domain(0, n)

    a_value = a[rows, inner]
    c_value = c[rows, columns]
    product = I.matmul(
        a_value,
        a_value,
        transpose_rhs=True,
        acc_dtype=I.f32,
    )
    d[rows, columns] = alpha * product + beta * c_value


@intent.kernel
def _absolute_sum(
    d: I.In[I.f32, ("N", "N")],
    out: I.Out[I.f32, ()],
):
    n, _ = d.shape
    rows = I.domain(0, n)
    columns = I.domain(0, n)
    values = d[rows, columns]
    total = I.reduce.sum(I.abs(values), axis=(0, 1), acc_dtype=I.f32)
    out[()] = total


def build(context):
    symmetric_mm = context.compile("symmetric_mm_stage", _symmetric_mm, constexprs={})
    absolute_sum = context.compile("absolute_sum_stage", _absolute_sum, constexprs={})

    def symmetric_mm_and_abs_sum(
        A: torch.Tensor,
        C: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.5,
    ) -> torch.Tensor:
        intermediate = torch.empty(C.shape, device=C.device, dtype=torch.float32)
        output = torch.empty((), device=C.device, dtype=torch.float32)
        symmetric_mm(A, C, intermediate, alpha, beta)
        absolute_sum(intermediate, output)
        return output

    return symmetric_mm_and_abs_sum
