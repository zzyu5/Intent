import intent
import intent.language as I


M = 512
K = 2048
N = 4096
PACK_FACTOR = 8
GROUP_SIZE = 64


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
