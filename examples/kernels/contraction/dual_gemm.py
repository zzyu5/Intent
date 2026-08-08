import intent
import intent.language as I


M = 2048
K = 4096
N = 4096


@intent.kernel
def gated_dual_gemm(
    x: I.In[I.f16, ("M", "K")],
    gate_weight: I.In[I.f16, ("K", "N")],
    value_weight: I.In[I.f16, ("K", "N")],
    y: I.Out[I.f16, ("M", "N")],
):
    M, K = x.shape
    _, N = gate_weight.shape
    m_axis = I.domain(0, M)
    n_axis = I.domain(0, N)
    k_axis = I.domain(0, K)
    for mr in I.parallel(I.partition(m_axis, extent=I.auto("M_TILE"))):
        for nr in I.parallel(I.partition(n_axis, extent=I.auto("N_TILE"))):
            gate = I.contract(
                x[mr, k_axis],
                gate_weight[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            value = I.contract(
                x[mr, k_axis],
                value_weight[k_axis, nr],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            y[mr, nr] = I.cast(I.maximum(gate, 0.0) * value, I.f16)
