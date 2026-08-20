import intent
import intent.language as I


M = 512
K = 2048
N = 4096
PACK_FACTOR = 8
GROUP_SIZE = 64
W4A8_M = 512
W4A8_K = 2048
W4A8_N = 4096
W4A8_PACK_FACTOR = 2


@intent.kernel
def weight_only_int4_matmul(
    activation: I.In[I.f16, ("M", "K")],
    packed_weight: I.In[I.i32, ("P", "N")],
    scales: I.In[I.f16, ("G", "N")],
    output: I.Out[I.f16, ("M", "N")],
):
    M, K = activation.shape
    N = output.shape[1]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            accumulation = I.state_stream(
                reduction,
                extent=I.auto("K_TILE"),
                init=(I.zeros((row_region, column_region), dtype=I.f32),),
            )
            with accumulation:
                for k_region, accumulator in accumulation:
                    k_indices = I.indices(k_region)
                    packed_indices = k_indices // PACK_FACTOR
                    shifts = I.cast(
                        (k_indices % PACK_FACTOR) * 4,
                        I.i32,
                    )
                    packed = packed_weight[packed_indices, column_region]
                    nibble = (packed >> shifts[:, None]) & 15
                    sign = -((nibble & 8) << 1)
                    unpacked = nibble + sign
                    group_indices = k_indices // GROUP_SIZE
                    group_scales = scales[group_indices, column_region]
                    dequantized = I.cast(unpacked, I.f16) * group_scales
                    partial = I.contract(
                        activation[row_region, k_region],
                        dequantized,
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    )
                    accumulation.yield_(accumulator + partial)
            output[row_region, column_region] = I.cast(
                accumulation.result,
                I.f16,
            )


@intent.kernel
def w4a8_packed_matmul(
    activation: I.In[I.i8, ("M", "K")],
    packed_weight: I.In[I.u8, ("N", "P")],
    output: I.Out[I.i32, ("N", "M")],
):
    M, K = activation.shape
    N = packed_weight.shape[0]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            accumulation = I.state_stream(
                reduction,
                extent=I.auto("K_TILE"),
                init=(I.zeros((column_region, row_region), dtype=I.i32),),
            )
            with accumulation:
                for k_region, accumulator in accumulation:
                    k_indices = I.indices(k_region)
                    packed_indices = k_indices // W4A8_PACK_FACTOR
                    shifts = I.cast(
                        (k_indices % W4A8_PACK_FACTOR) * 4,
                        I.u8,
                    )
                    packed = I.mask(
                        packed_weight[column_region, packed_indices],
                        valid=packed_indices[None, :]
                        < packed_weight.shape[1],
                        fill=I.cast(0, I.u8),
                    )
                    nibble = (packed >> shifts[None, :]) & 15
                    signed = I.cast(
                        I.cast(nibble, I.i32) - I.cast(
                        (nibble & 8) << 1,
                        I.i32,
                        ),
                        I.i8,
                    )
                    partial = I.contract(
                        signed,
                        activation[row_region, k_region],
                        reduce=((1, 1),),
                        acc_dtype=I.i32,
                    )
                    accumulation.yield_(accumulator + partial)
            output[column_region, row_region] = accumulation.result


@intent.kernel
def fp8_e4m3_matmul(
    lhs: I.In[I.f8e4m3fn, ("M", "K")],
    rhs_transposed: I.In[I.f8e4m3fn, ("N", "K")],
    output: I.Out[I.f8e4m3fn, ("M", "N")],
):
    M, K = lhs.shape
    N = rhs_transposed.shape[0]
    reduction = I.domain(0, K)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            result = I.contract(
                lhs[row_region, reduction],
                rhs_transposed[column_region, reduction],
                reduce=((1, 1),),
                acc_dtype=I.f32,
            )
            output[row_region, column_region] = I.cast(result, I.f8e4m3fn)


@intent.kernel
def fp8_e5m2_matmul(
    lhs: I.In[I.f8e5m2, ("M", "K")],
    rhs_transposed: I.In[I.f8e5m2, ("N", "K")],
    output: I.Out[I.f8e5m2, ("M", "N")],
):
    M, K = lhs.shape
    N = rhs_transposed.shape[0]
    reduction = I.domain(0, K)
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    for row_region in I.parallel(
        I.partition(rows, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            result = I.contract(
                lhs[row_region, reduction],
                rhs_transposed[column_region, reduction],
                reduce=((1, 1),),
                acc_dtype=I.f32,
            )
            output[row_region, column_region] = I.cast(result, I.f8e5m2)
