import intent
import intent.language as I


M = 512
N = 512
K_BLOCKS = 24
BLOCK_SIZE = 32


@intent.kernel
def block_scaled_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "KB", 32)],
    lhs_scale: I.In[I.f8e8m0fnu, ("M", "KB")],
    rhs: I.In[I.f8e4m3fn, ("KB", 32, "N")],
    rhs_scale: I.In[I.f8e8m0fnu, ("KB", "N")],
    output: I.Out[I.f32, ("M", "N")],
):
    M, KB, KI = lhs.shape
    N = rhs.shape[2]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    k_blocks = I.domain(0, KB)
    k_inner = I.domain(0, KI)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            accumulation = I.state_stream(
                k_blocks,
                extent=1,
                init=(I.zeros((row_region, column_region), dtype=I.f32),),
            )
            with accumulation:
                for k_block_region, accumulator in accumulation:
                    scaled_lhs = (
                        I.reshape(
                            I.cast(
                                lhs[row_region, k_block_region, k_inner], I.f32
                            ),
                            (row_region, 32),
                        )
                        * I.reshape(
                            I.cast(
                                lhs_scale[row_region, k_block_region], I.f32
                            ),
                            (row_region, 1),
                        )
                    )
                    scaled_rhs = (
                        I.reshape(
                            I.cast(
                                rhs[k_block_region, k_inner, column_region], I.f32
                            ),
                            (32, column_region),
                        )
                        * I.reshape(
                            I.cast(
                                rhs_scale[k_block_region, column_region], I.f32
                            ),
                            (1, column_region),
                        )
                    )
                    partial = I.contract(
                        scaled_lhs,
                        scaled_rhs,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    accumulation.yield_(accumulator + partial)
            output[row_region, column_region] = accumulation.result
