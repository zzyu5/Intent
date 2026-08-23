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
BITNET_VALUES_PER_WORD = 16
BITNET_BYTES_PER_WORD = 4


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
    accumulation = I.state_stream(
        reduction,
        extent=I.auto("K_TILE"),
        init=(I.zeros((rows, columns), dtype=I.f32),),
    )
    with accumulation:
        for k_region, accumulator in accumulation:
            k_indices = I.indices(k_region)
            packed_indices = k_indices // PACK_FACTOR
            shifts = I.cast(
                (k_indices % PACK_FACTOR) * 4,
                I.i32,
            )
            packed = packed_weight[packed_indices, columns]
            nibble = (packed >> shifts[:, None]) & 15
            sign = -((nibble & 8) << 1)
            unpacked = nibble + sign
            group_indices = k_indices // GROUP_SIZE
            group_scales = scales[group_indices, columns]
            dequantized = I.cast(unpacked, I.f16) * group_scales
            partial = I.contract(
                activation[rows, k_region],
                dequantized,
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            accumulation.yield_(accumulator + partial)
    output[rows, columns] = I.cast(
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
    accumulation = I.state_stream(
        reduction,
        extent=I.auto("K_TILE"),
        init=(I.zeros((columns, rows), dtype=I.i32),),
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
                packed_weight[columns, packed_indices],
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
                activation[rows, k_region],
                reduce=((1, 1),),
                acc_dtype=I.i32,
            )
            accumulation.yield_(accumulator + partial)
    output[columns, rows] = accumulation.result


@intent.kernel
def bitnet_int2_matmul(
    activation: I.In[I.i8, ("M", "K")],
    packed_weight: I.In[I.u8, ("N", "P")],
    output: I.Out[I.i32, ("M", "N")],
):
    M, K = activation.shape
    N = packed_weight.shape[0]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    accumulation = I.state_stream(
        reduction,
        extent=I.auto("K_TILE"),
        init=(I.zeros((rows, columns), dtype=I.i32),),
    )
    with accumulation:
        for k_region, accumulator in accumulation:
            k_indices = I.indices(k_region)
            packed_indices = (
                (k_indices // BITNET_VALUES_PER_WORD) * BITNET_BYTES_PER_WORD
                + (k_indices % BITNET_BYTES_PER_WORD)
            )
            shifts = I.cast(
                ((k_indices // BITNET_BYTES_PER_WORD) % BITNET_BYTES_PER_WORD)
                * 2,
                I.u8,
            )
            packed = packed_weight[columns, packed_indices]
            decoded = I.cast(
                (packed >> shifts[None, :]) & I.cast(3, I.u8),
                I.i8,
            )
            partial = I.contract(
                activation[rows, k_region],
                decoded,
                reduce=((1, 1),),
                acc_dtype=I.i32,
            )
            accumulation.yield_(accumulator + partial)
    output[rows, columns] = accumulation.result


@intent.kernel
def dequant_bf16_fp4_matmul(
    activation: I.In[I.bf16, ("M", "K")],
    packed_weight: I.In[I.u8, ("N", "P")],
    output: I.Out[I.bf16, ("M", "N")],
):
    M, K = activation.shape
    N = packed_weight.shape[0]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    accumulation = I.state_stream(
        reduction,
        extent=I.auto("K_TILE"),
        init=(I.zeros((rows, columns), dtype=I.f32),),
    )
    with accumulation:
        for k_region, accumulator in accumulation:
            k_indices = I.indices(k_region)
            packed_pair = (k_indices // 4) * 2
            packed_pair_next = packed_pair + 1
            I.assume_in_bounds(packed_pair, packed_weight, axis=1)
            I.assume_in_bounds(packed_pair_next, packed_weight, axis=1)
            first = I.cast(packed_weight[columns, packed_pair], I.u16)
            second = I.cast(packed_weight[columns, packed_pair_next], I.u16)
            word = (first << I.cast(8, I.u16)) | second
            decode_mask = I.cast(0x81C0, I.u16)
            decoded_zero = word & decode_mask
            decoded_one = (word << I.cast(3, I.u16)) & decode_mask
            decoded_two = (word << I.cast(6, I.u16)) & decode_mask
            decoded_three = (
                (word << I.cast(1, I.u16)) & I.cast(0x8000, I.u16)
            ) | (
                (word >> I.cast(3, I.u16)) & I.cast(0x0180, I.u16)
            ) | (
                (word >> I.cast(7, I.u16)) & I.cast(0x0040, I.u16)
            )
            position = k_indices % 4
            zero = I.cast(0, I.u16)
            bits = (
                I.mask(decoded_zero, valid=position[None, :] == 0, fill=zero)
                | I.mask(decoded_one, valid=position[None, :] == 1, fill=zero)
                | I.mask(decoded_two, valid=position[None, :] == 2, fill=zero)
                | I.mask(decoded_three, valid=position[None, :] == 3, fill=zero)
            )
            decoded = I.cast(
                I.cast(I.bitcast(bits, I.bf16), I.f32) * (2.0**126),
                I.bf16,
            )
            partial = I.contract(
                activation[rows, k_region],
                decoded,
                reduce=((1, 1),),
                acc_dtype=I.f32,
            )
            accumulation.yield_(accumulator + partial)
    output[rows, columns] = I.cast(accumulation.result, I.bf16)


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
    result = I.contract(
        lhs[rows, reduction],
        rhs_transposed[columns, reduction],
        reduce=((1, 1),),
        acc_dtype=I.f32,
    )
    output[rows, columns] = I.cast(result, I.f8e4m3fn)


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
    result = I.contract(
        lhs[rows, reduction],
        rhs_transposed[columns, reduction],
        reduce=((1, 1),),
        acc_dtype=I.f32,
    )
    output[rows, columns] = I.cast(result, I.f8e5m2)
