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
    output[rows, columns] = I.sparse_contract_2to4(
        compressed[rows, :],
        metadata[rows, :],
        rhs[reduction, columns],
        acc_dtype=I.f32,
    )
