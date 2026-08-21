import intent
import intent.language as I

from kernels.contraction.gemm import Activation
from kernels.convolution.direct import CONV2D_FILTER_HEIGHT
from kernels.convolution.direct import CONV2D_FILTER_WIDTH


@intent.kernel
def gemm_loop_interchange(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    c: I.Out[I.f16, ("M", "N")],
    ACTIVATION: I.Constexpr[Activation],
):
    M, K = a.shape
    _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    accumulator = I.contract(
        a[m_axis, k_axis],
        b[k_axis, n_axis],
        reduce=((1, 0),),
        acc_dtype=I.f32,
    )
    if ACTIVATION == Activation.RELU:
        accumulator = I.maximum(accumulator, 0.0)
    c[m_axis, n_axis] = I.cast(accumulator, I.f16)


@intent.kernel
def conv2d_reduce_order(
    x: I.In[I.f16, ("B", "H", "W")],
    weight: I.In[I.f16, ("R", "S")],
    output: I.Out[I.f16, ("B", "H", "W")],
):
    B, H, W = x.shape
    height = I.domain(0, H)
    width = I.domain(0, W)
    kernel_rows = I.domain(0, CONV2D_FILTER_HEIGHT)
    kernel_columns = I.domain(0, CONV2D_FILTER_WIDTH)
    for batch in I.parallel(I.domain(0, B)):
        output_row_indices = I.reshape(
            I.indices(height),
            (height, 1, 1, 1),
        )
        output_column_indices = I.reshape(
            I.indices(width),
            (1, width, 1, 1),
        )
        kernel_row_indices = I.indices(kernel_rows)[:, None]
        kernel_column_indices = I.indices(kernel_columns)
        input_rows = (
            output_row_indices
            + kernel_row_indices
            - CONV2D_FILTER_HEIGHT // 2
        )
        input_columns = (
            output_column_indices
            + kernel_column_indices
            - CONV2D_FILTER_WIDTH // 2
        )
        products = I.cast(
            x[batch, input_rows, input_columns],
            I.f32,
        ) * I.cast(weight[kernel_rows, kernel_columns], I.f32)
        reduced_rows = I.reduce.sum(
            products,
            axis=2,
            identity=0.0,
            acc_dtype=I.f32,
        )
        reduced = I.reduce.sum(
            reduced_rows,
            axis=2,
            identity=0.0,
            acc_dtype=I.f32,
        )
        output[batch, height, width] = I.cast(
            reduced,
            I.f16,
        )
