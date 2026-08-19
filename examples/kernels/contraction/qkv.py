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
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for q_region in I.parallel(
            I.partition(q_columns, extent=I.auto("NQ_TILE"))
        ):
            q_output[row_region, q_region] = I.cast(
                I.contract(
                    x[row_region, reduction],
                    q_weight[reduction, q_region],
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                ),
                I.f16,
            )
        for k_region in I.parallel(
            I.partition(k_columns, extent=I.auto("NK_TILE"))
        ):
            k_output[row_region, k_region] = I.cast(
                I.contract(
                    x[row_region, reduction],
                    k_weight[reduction, k_region],
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                ),
                I.f16,
            )
        for v_region in I.parallel(
            I.partition(v_columns, extent=I.auto("NV_TILE"))
        ):
            v_output[row_region, v_region] = I.cast(
                I.contract(
                    x[row_region, reduction],
                    v_weight[reduction, v_region],
                    reduce=((1, 0),),
                    acc_dtype=I.f32,
                ),
                I.f16,
            )
