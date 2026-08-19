import intent
import intent.language as I


TOKENS = 2048
HIDDEN = 4096
STREAMS = 4
MIX_COMPONENTS = STREAMS * (STREAMS + 2)


@intent.kernel
def mhc_gemm_rms_scale(
    x: I.In[I.bf16, ("T", "K")],
    weight: I.In[I.bf16, ("K", "N")],
    bias: I.In[I.bf16, ("N",)],
    mixed: I.Out[I.bf16, ("T", "N")],
    rms: I.Out[I.f32, ("T", 1)],
    STREAMS: I.Constexpr[int],
    ALPHA_PRE: I.Constexpr[float],
    ALPHA_POST: I.Constexpr[float],
    ALPHA_RESIDUAL: I.Constexpr[float],
):
    T, K = x.shape
    N = weight.shape[1]
    tokens = I.domain(0, T)
    columns = I.domain(0, N)
    reduction = I.domain(0, K)
    for token_region in I.parallel(
        I.partition(tokens, extent=I.auto("M_TILE"))
    ):
        for column_region in I.parallel(
            I.partition(columns, extent=I.auto("N_TILE"))
        ):
            accumulation = I.state_stream(
                reduction,
                extent=I.auto("K_TILE"),
                init=(
                    I.zeros((token_region,), dtype=I.f32),
                    I.zeros((token_region, column_region), dtype=I.f32),
                ),
            )
            with accumulation:
                for reduction_region, (square_sum, linear) in accumulation:
                    values = I.cast(x[token_region, reduction_region], I.f32)
                    accumulation.yield_(
                        square_sum
                        + I.reduce.sum(values * values, axis=1, identity=0.0),
                        linear
                        + I.contract(
                            x[token_region, reduction_region],
                            weight[reduction_region, column_region],
                            reduce=((1, 0),),
                            acc_dtype=I.f32,
                        ),
                    )
            square_sum, linear = accumulation.result
            root_mean_square = I.rsqrt(square_sum / I.cast(K, I.f32))
            column_index = I.indices(column_region)
            pre = column_index < STREAMS
            post = (column_index >= STREAMS) and (column_index < 2 * STREAMS)
            zero = I.cast(column_index, I.f32) * 0.0
            scale = I.mask(
                zero + ALPHA_PRE,
                valid=pre,
                fill=zero + ALPHA_RESIDUAL,
            )
            scale = I.mask(
                zero + ALPHA_POST,
                valid=post,
                fill=scale,
            )
            normalized = (
                linear * scale * root_mean_square[:, None]
                + I.cast(bias[column_region], I.f32)
            )
            sigmoid = 1.0 / (1.0 + I.exp(-normalized))
            result = I.mask(
                normalized,
                valid=(pre == False) and (post == False),
                fill=sigmoid,
            )
            result = I.mask(2.0 * sigmoid, valid=post, fill=result)
            mixed[token_region, column_region] = I.cast(result, I.bf16)
            if column_index[0] == 0:
                rms[token_region, 0] = 1.0 / root_mean_square


@intent.kernel
def mhc_apply_residual(
    residual: I.In[I.bf16, ("T", "S", "D")],
    layer_output: I.In[I.bf16, ("T", "D")],
    post_mix: I.In[I.f32, ("T", "S")],
    residual_mix: I.In[I.f32, ("T", "S", "S")],
    output: I.Out[I.bf16, ("T", "S", "D")],
):
    T, S, D = residual.shape
    dimensions = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        for output_stream in I.parallel(I.domain(0, S)):
            mixed_residual = I.zeros((dimensions,), dtype=I.f32)
            for source_stream in range(STREAMS):
                mixed_residual = mixed_residual + I.cast(
                    residual[token, source_stream, dimensions], I.f32
                ) * residual_mix[token, output_stream, source_stream]
            output[token, output_stream, dimensions] = I.cast(
                mixed_residual
                + post_mix[token, output_stream]
                * I.cast(layer_output[token, dimensions], I.f32),
                I.bf16,
            )


@intent.kernel
def mhc_sinkhorn(
    logits: I.In[I.f32, ("T", "S", "S")],
    normalized: I.Out[I.f32, ("T", "S", "S")],
):
    T, S, _ = logits.shape
    rows = I.domain(0, S)
    columns = I.domain(0, S)
    for token in I.parallel(I.domain(0, T)):
        matrix = I.exp(logits[token, rows, columns])
        for _ in range(20):
            row_sum = I.reduce.sum(matrix, axis=1, identity=0.0)
            matrix = matrix / row_sum[:, None]
            column_sum = I.reduce.sum(matrix, axis=0, identity=0.0)
            matrix = matrix / column_sum[None, :]
        normalized[token, rows, columns] = matrix


@intent.kernel
def mhc_pre_gemm_sqrsum(
    residual_flat: I.In[I.bf16, ("T", "K")],
    weight: I.In[I.f32, ("C", "K")],
    mixes: I.Out[I.f32, ("T", "C")],
    square_sum: I.Out[I.f32, ("T",)],
):
    T, K = residual_flat.shape
    C = weight.shape[0]
    tokens = I.domain(0, T)
    reduction = I.domain(0, K)
    components = I.domain(0, C)
    for token_region in I.parallel(
        I.partition(tokens, extent=I.auto("M_TILE"))
    ):
        values = I.cast(residual_flat[token_region, reduction], I.f32)
        square_sum[token_region] = I.reduce.sum(
            values * values,
            axis=1,
            identity=0.0,
        )
        mixes[token_region, components] = I.contract(
            residual_flat[token_region, reduction],
            I.cast(weight[components, reduction], I.bf16),
            reduce=((1, 1),),
            acc_dtype=I.f32,
        )


@intent.kernel
def mhc_pre_fuse(
    mixes: I.In[I.f32, ("T", "C")],
    square_sum: I.In[I.f32, ("T",)],
    scale: I.In[I.f32, (3,)],
    base: I.In[I.f32, ("C",)],
    residual: I.In[I.bf16, ("T", "S", "D")],
    post_mix: I.Out[I.f32, ("T", "S")],
    residual_mix: I.Out[I.f32, ("T", "S", "S")],
    layer_input: I.Out[I.bf16, ("T", "D")],
    RMS_EPS: I.Constexpr[float],
    PRE_EPS: I.Constexpr[float],
    SINKHORN_EPS: I.Constexpr[float],
    POST_MULTIPLIER: I.Constexpr[float],
    SINKHORN_REPEATS: I.Constexpr[int],
):
    T, S, D = residual.shape
    streams = I.domain(0, S)
    hidden = I.domain(0, D)
    for token in I.parallel(I.domain(0, T)):
        normalization = I.rsqrt(
            square_sum[token] / I.cast(S * D, I.f32) + RMS_EPS
        )
        pre_logits = (
            mixes[token, streams] * normalization * scale[0]
            + base[streams]
        )
        pre = 1.0 / (1.0 + I.exp(-pre_logits)) + PRE_EPS
        post_indices = I.indices(streams) + S
        post_logits = (
            mixes[token, post_indices] * normalization * scale[1]
            + base[post_indices]
        )
        post_mix[token, streams] = (
            1.0 / (1.0 + I.exp(-post_logits))
        ) * POST_MULTIPLIER
        matrix_indices = (
            2 * S
            + I.indices(streams)[:, None] * S
            + I.indices(streams)[None, :]
        )
        matrix = (
            mixes[token, matrix_indices] * normalization * scale[2]
            + base[matrix_indices]
        )
        row_maximum = I.reduce.max(matrix, axis=1, identity=-I.inf)
        matrix = I.exp(matrix - row_maximum[:, None])
        row_sum = I.reduce.sum(matrix, axis=1, identity=0.0)
        matrix = matrix / row_sum[:, None] + SINKHORN_EPS
        column_sum = I.reduce.sum(matrix, axis=0, identity=0.0)
        matrix = matrix / (column_sum[None, :] + SINKHORN_EPS)
        for _ in range(SINKHORN_REPEATS - 1):
            row_sum = I.reduce.sum(matrix, axis=1, identity=0.0)
            matrix = matrix / (row_sum[:, None] + SINKHORN_EPS)
            column_sum = I.reduce.sum(matrix, axis=0, identity=0.0)
            matrix = matrix / (column_sum[None, :] + SINKHORN_EPS)
        residual_mix[token, streams, streams] = matrix
        layer_input[token, hidden] = I.cast(
            I.reduce.sum(
                I.cast(residual[token, streams, hidden], I.f32)
                * pre[:, None],
                axis=0,
                identity=0.0,
            ),
            I.bf16,
        )
