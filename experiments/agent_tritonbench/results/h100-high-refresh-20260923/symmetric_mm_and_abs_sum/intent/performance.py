import torch
import intent
import intent.language as I


@intent.kernel
def _symmetric_update(
    A: I.In[I.f32, (1024, 1024)],
    C: I.In[I.f32, (1024, 1024)],
    D: I.Out[I.f32, (1024, 1024)],
    alpha: I.f32,
    beta: I.f32,
):
    rows = I.domain(0, 1024)
    cols = I.domain(0, 1024)
    product = I.matmul(A, A, transpose_rhs=True, acc_dtype=I.f32)
    D[rows, cols] = product * alpha + C[rows, cols] * beta


@intent.kernel
def _group_abs_sums(
    D: I.In[I.f32, (1024, 1024)],
    partials: I.Out[I.f32, (16,)],
):
    groups = I.domain(0, 16)
    cols = I.domain(0, 1024)
    for group in I.parallel(groups):
        begin = group * 64
        rows = I.domain(begin, begin + 64)
        block = D[rows, cols]
        partials[group] = I.reduce.sum(
            I.abs(block), axis=(0, 1), acc_dtype=I.f32
        )


@intent.kernel
def _finish_abs_sum(
    partials: I.In[I.f32, (16,)],
    result: I.Out[I.f32, ()],
):
    groups = I.domain(0, 16)
    result[()] = I.reduce.sum(partials[groups], axis=0, acc_dtype=I.f32)


def build(context):
    update = context.compile("symmetric_update", _symmetric_update)
    group_abs = context.compile("group_abs_sums", _group_abs_sums)
    finish = context.compile("finish_abs_sum", _finish_abs_sum)

    def symmetric_mm_and_abs_sum(
        A: "torch.Tensor",
        C: "torch.Tensor",
        alpha: "float" = 1.0,
        beta: "float" = 0.5,
    ) -> "torch.Tensor":
        D = torch.empty_like(C)
        partials = torch.empty((16,), device=A.device, dtype=torch.float32)
        result = torch.empty((), device=A.device, dtype=torch.float32)

        update(A, C, D, alpha, beta)
        group_abs(D, partials)
        finish(partials, result)
        return result

    return symmetric_mm_and_abs_sum
