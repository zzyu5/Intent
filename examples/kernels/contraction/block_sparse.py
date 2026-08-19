import intent
import intent.language as I


M = 4096
N = 4096
K = 4096
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


@intent.kernel
def block_sparse_matmul(
    lhs: I.In[I.f16, ("M", "K")],
    rhs: I.In[I.f16, ("K", "N")],
    block_mask: I.In[I.u8, ("MB", "NB", "KB")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, K = lhs.shape
    N = rhs.shape[1]
    MB, NB, KB = block_mask.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for row_region in I.parallel(I.partition(rows, extent=BLOCK_M)):
        for column_region in I.parallel(I.partition(columns, extent=BLOCK_N)):
            accumulation = I.state_stream(
                reduction,
                extent=BLOCK_K,
                init=(I.zeros((row_region, column_region), dtype=I.f32),),
            )
            with accumulation:
                for reduction_region, accumulator in accumulation:
                    row_block = I.indices(row_region)[0] // BLOCK_M
                    column_block = I.indices(column_region)[0] // BLOCK_N
                    reduction_block = I.indices(reduction_region)[0] // BLOCK_K
                    I.assume_in_bounds(row_block, block_mask, axis=0)
                    I.assume_in_bounds(column_block, block_mask, axis=1)
                    I.assume_in_bounds(reduction_block, block_mask, axis=2)
                    enabled = (
                        block_mask[row_block, column_block, reduction_block]
                        != 0
                    )
                    partial = I.contract(
                        lhs[row_region, reduction_region],
                        rhs[reduction_region, column_region],
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    accumulation.yield_(
                        accumulator
                        + I.mask(partial, valid=enabled, fill=0.0)
                    )
            output[row_region, column_region] = I.cast(
                accumulation.result, I.f16
            )
