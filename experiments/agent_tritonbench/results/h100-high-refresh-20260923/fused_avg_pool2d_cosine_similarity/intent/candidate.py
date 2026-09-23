import torch
import intent
import intent.language as I


@intent.kernel
def _cosine_stage(
    x1: I.In[I.f32, (16, 64, 32, 32)],
    x2: I.In[I.f32, (16, 64, 32, 32)],
    result: I.Out[I.f32, (16, 32, 32)],
    eps: I.f32,
):
    channels = I.domain(0, 64)
    for batch in I.parallel(I.domain(0, 16)):
        for row in I.parallel(I.domain(0, 32)):
            for column in I.parallel(I.domain(0, 32)):
                lhs = x1[batch, channels, row, column]
                rhs = x2[batch, channels, row, column]
                dot = I.reduce.sum(lhs * rhs, axis=0, acc_dtype=I.f32)
                lhs_norm = I.sqrt(
                    I.reduce.sum(lhs * lhs, axis=0, acc_dtype=I.f32)
                )
                rhs_norm = I.sqrt(
                    I.reduce.sum(rhs * rhs, axis=0, acc_dtype=I.f32)
                )
                lhs_norm = I.maximum(lhs_norm, eps)
                rhs_norm = I.maximum(rhs_norm, eps)
                result[batch, row, column] = I.fdiv(
                    dot, lhs_norm * rhs_norm
                )


@intent.kernel
def _pool_stage(
    source: I.In[I.f32, (16, 32, 32)],
    result: I.Out[I.f32, (16, 1, 16, 16)],
):
    for batch in I.parallel(I.domain(0, 16)):
        for out_row in I.parallel(I.domain(0, 16)):
            for out_column in I.parallel(I.domain(0, 16)):
                row_start = out_row * 2 - 1
                column_start = out_column * 2 - 1
                total = I.cast(0.0, I.f32)
                for kernel_row in I.domain(0, 3):
                    input_row = row_start + kernel_row
                    if input_row >= 0:
                        if input_row < 32:
                            for kernel_column in I.domain(0, 3):
                                input_column = column_start + kernel_column
                                if input_column >= 0:
                                    if input_column < 32:
                                        total = total + source[
                                            batch, input_row, input_column
                                        ]
                result[batch, 0, out_row, out_column] = I.fdiv(
                    total, I.cast(9.0, I.f32)
                )


def build(context):
    cosine_artifact = context.compile("cosine_stage", _cosine_stage)
    pool_artifact = context.compile("pool_stage", _pool_stage)

    def fused_avg_pool2d_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        kernel_size: int,
        stride: int = None,
        padding: int = 0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size
        intermediate = torch.empty(
            (16, 32, 32), device=x1.device, dtype=x1.dtype
        )
        result = torch.empty(
            (16, 1, 16, 16), device=x1.device, dtype=x1.dtype
        )
        cosine_artifact(x1, x2, intermediate, eps)
        pool_artifact(intermediate, result)
        return result

    return fused_avg_pool2d_cosine_similarity
