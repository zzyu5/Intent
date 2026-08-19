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
                extent=I.auto("SCALE_GROUP_TILE"),
                init=(I.zeros((row_region, column_region), dtype=I.f32),),
            )
            with accumulation:
                for k_block_region, accumulator in accumulation:
                    partial = I.scaled_contract(
                        lhs[row_region, k_block_region, k_inner],
                        lhs_scale[row_region, k_block_region],
                        rhs[k_block_region, k_inner, column_region],
                        rhs_scale[k_block_region, column_region],
                        acc_dtype=I.f32,
                        lhs_group_size=32,
                        rhs_group_size=32,
                    )
                    accumulation.yield_(accumulator + partial)
            output[row_region, column_region] = accumulation.result


@intent.kernel
def scaled_fp8_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "K")],
    rhs: I.In[I.f8e4m3fn, ("K", "N")],
    output: I.Out[I.f16, ("M", "N")],
    lhs_scale: I.f32,
    rhs_scale: I.f32,
):
    M, K = lhs.shape
    N = rhs.shape[1]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            value = I.contract(
                lhs[row_region, reduction],
                rhs[reduction, column_region],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            output[row_region, column_region] = I.cast(
                value * lhs_scale * rhs_scale,
                I.f16,
            )


@intent.kernel
def scaled_fp8_splitk_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "IT", "SP", 256)],
    rhs: I.In[I.f8e4m3fn, ("IT", "SP", 256, "N")],
    output: I.InOut[I.f16, ("M", "N")],
    lhs_scale: I.f32,
    rhs_scale: I.f32,
):
    M, IT, SP, _ = lhs.shape
    N = rhs.shape[3]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    iterations = I.domain(0, IT)
    splits = I.domain(0, SP)
    reduction = I.domain(0, 256)
    for split in I.parallel(splits):
        for row_region in I.parallel(
            I.partition(rows, extent=I.auto("M_TILE"))
        ):
            for column_region in I.parallel(
                I.partition(columns, extent=I.auto("N_TILE"))
            ):
                accumulation = I.state_stream(
                    iterations,
                    extent=1,
                    init=(I.zeros((row_region, column_region), dtype=I.f32),),
                )
                with accumulation:
                    for iteration_region, accumulator in accumulation:
                        lhs_block = I.reshape(
                            lhs[
                                row_region,
                                iteration_region,
                                split,
                                reduction,
                            ],
                            (row_region, 256),
                        )
                        rhs_block = I.reshape(
                            rhs[
                                iteration_region,
                                split,
                                reduction,
                                column_region,
                            ],
                            (256, column_region),
                        )
                        partial = I.contract(
                            lhs_block,
                            rhs_block,
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        )
                        accumulation.yield_(
                            accumulator + partial
                        )
                I.atomic_add(
                    output,
                    index=(row_region, column_region),
                    value=I.cast(
                        accumulation.result * lhs_scale * rhs_scale,
                        I.f16,
                    ),
                )


@intent.kernel
def deepgemm_fp8_2xacc(
    lhs: I.In[I.f8e4m3fn, ("M", "KB", 128)],
    rhs: I.In[I.f8e4m3fn, ("N", "KB", 128)],
    lhs_scale: I.In[I.f32, ("M", "KB")],
    rhs_scale: I.In[I.f32, ("NB", "KB")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, KB, KI = lhs.shape
    N = rhs.shape[0]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, KI)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            accumulation = I.state_stream(
                I.domain(0, KB),
                extent=1,
                init=(I.zeros((row_region, column_region), dtype=I.f32),),
            )
            with accumulation:
                for block_region, accumulator in accumulation:
                    lhs_block = I.reshape(
                        lhs[row_region, block_region, reduction],
                        (row_region, KI),
                    )
                    rhs_block = I.reshape(
                        rhs[column_region, block_region, reduction],
                        (column_region, KI),
                    )
                    partial = I.contract(
                        lhs_block,
                        rhs_block,
                        reduce=((1, 1),),
                        acc_dtype=I.f32,
                    )
                    column_blocks = I.indices(column_region) // 128
                    I.assume_in_bounds(column_blocks, rhs_scale, axis=0)
                    left_scale = I.reshape(
                        lhs_scale[row_region, block_region],
                        (row_region,),
                    )
                    right_scale = I.reshape(
                        rhs_scale[column_blocks, block_region],
                        (column_region,),
                    )
                    accumulation.yield_(
                        accumulator
                        + I.reshape(partial, (row_region, column_region))
                        * left_scale[:, None]
                        * right_scale[None, :]
                    )
            output[row_region, column_region] = I.cast(
                accumulation.result,
                I.bf16,
            )
