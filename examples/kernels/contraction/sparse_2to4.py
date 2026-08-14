import intent
import intent.language as I


M = 2048
N = 4096
K = 4096


@intent.kernel
def sparse_2to4_gemm(
    compressed: I.In[I.f16, ("M", "KC")],
    metadata: I.In[I.i16, ("M", "KE")],
    rhs: I.In[I.f16, ("K", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, _ = compressed.shape
    K, N = rhs.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            output[row_region, column_region] = I.sparse_contract_2to4(
                compressed[row_region, :],
                metadata[row_region, :],
                rhs[reduction, column_region],
                acc_dtype=I.f32,
            )
