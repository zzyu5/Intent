import intent
import intent.language as I


M = 4096
K = 4096
N = 14336


class Activation(I.Enum):
    NONE = 0
    RELU = 1


@intent.kernel
def gemm(
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
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            accumulator = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            if ACTIVATION == Activation.RELU:
                accumulator = I.maximum(accumulator, 0.0)
            c[mr, nr] = I.cast(accumulator, I.f16)


@intent.kernel
def bf16_gemm(
    a: I.In[I.bf16, ("M", "K")],
    b: I.In[I.bf16, ("K", "N")],
    c: I.Out[I.bf16, ("M", "N")],
):
    M, K = a.shape
    _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            accumulator = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            c[mr, nr] = I.cast(accumulator, I.bf16)


@intent.kernel
def quantized_gemm(
    a: I.In[I.f16, ("M", "K")],
    b: I.In[I.f16, ("K", "N")],
    bias: I.In[I.f32, ("N",)],
    residual: I.In[I.f16, ("M", "N")],
    output_scale: I.In[I.f32, ("N",)],
    output: I.Out[I.i8, ("M", "N")],
):
    M, K = a.shape
    _, N = b.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            accumulator = I.contract(
                a[mr, k_axis],
                b[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            activated = I.maximum(accumulator + bias[nr], 0.0)
            fused = activated + I.cast(residual[mr, nr], I.f32)
            scaled = fused / output_scale[nr]
            saturated = I.minimum(I.maximum(scaled, -128.0), 127.0)
            output[mr, nr] = I.cast(saturated, I.i8)
