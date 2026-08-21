import intent
import intent.language as I


TOKENS = 4096
HIDDEN = 4096
PROJECTION = 4096


@intent.kernel
def fused_qkv_projection(
    x: I.In[I.f16, ("M", "K")],
    q_weight: I.In[I.f16, ("K", "NQ")],
    k_weight: I.In[I.f16, ("K", "NK")],
    v_weight: I.In[I.f16, ("K", "NV")],
    q_output: I.Out[I.f16, ("M", "NQ")],
    k_output: I.Out[I.f16, ("M", "NK")],
    v_output: I.Out[I.f16, ("M", "NV")],
):
    M, K = x.shape
    NQ = q_weight.shape[1]
    NK = k_weight.shape[1]
    NV = v_weight.shape[1]
    rows = I.domain(0, M)
    reduction = I.domain(0, K)
    q_columns = I.domain(0, NQ)
    k_columns = I.domain(0, NK)
    v_columns = I.domain(0, NV)
    q_output[rows, q_columns] = I.cast(
        I.contract(
            x[rows, reduction],
            q_weight[reduction, q_columns],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        ),
        I.f16,
    )
    k_output[rows, k_columns] = I.cast(
        I.contract(
            x[rows, reduction],
            k_weight[reduction, k_columns],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        ),
        I.f16,
    )
    v_output[rows, v_columns] = I.cast(
        I.contract(
            x[rows, reduction],
            v_weight[reduction, v_columns],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        ),
        I.f16,
    )
