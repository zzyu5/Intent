import intent
import intent.language as I


ROWS = 4096
FEATURES = 4096
PARTIAL_GROUPS = 128


@intent.kernel
def layer_norm_backward_rows(
    x: I.In[I.bf16, ("M", "N")],
    dy: I.In[I.bf16, ("M", "N")],
    weight: I.In[I.bf16, ("N",)],
    dx: I.Out[I.bf16, ("M", "N")],
    dw_partial: I.InOut[I.f32, ("G", "N")],
    db_partial: I.InOut[I.f32, ("G", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        x_values = I.cast(x[row, columns], I.f32)
        dy_values = I.cast(dy[row, columns], I.f32)
        weight_values = I.cast(weight[columns], I.f32)
        mean = I.reduce.sum(x_values, axis=0, identity=0.0) * inverse_features
        centered = x_values - mean
        second_moment = (
            I.reduce.sum(x_values * x_values, axis=0, identity=0.0)
            * inverse_features
        )
        variance = (
            second_moment - mean * mean
        )
        rstd = I.rsqrt(variance + epsilon)
        normalized = centered * rstd
        weighted_dy = weight_values * dy_values
        mean_weighted_dy = (
            I.reduce.sum(weighted_dy, axis=0, identity=0.0) * inverse_features
        )
        mean_weighted_dy_normalized = (
            (
                I.reduce.sum(weighted_dy * x_values, axis=0, identity=0.0)
                * inverse_features
                - mean_weighted_dy * mean
            )
            * rstd
        )
        dx[row, columns] = I.cast(
            (
                weighted_dy
                - mean_weighted_dy
                - normalized * mean_weighted_dy_normalized
            )
            * rstd,
            I.bf16,
        )
        group = row % PARTIAL_GROUPS
        I.scatter_reduce(
            dw_partial,
            index=(group, columns),
            value=dy_values * normalized,
            combine=I.add,
        )
        I.scatter_reduce(
            db_partial,
            index=(group, columns),
            value=dy_values,
            combine=I.add,
        )


@intent.kernel
def layer_norm_backward_reduce(
    dw_partial: I.In[I.f32, ("G", "N")],
    db_partial: I.In[I.f32, ("G", "N")],
    dw: I.Out[I.f32, ("N",)],
    db: I.Out[I.f32, ("N",)],
):
    G, N = dw_partial.shape
    rows = I.domain(0, G)
    features = I.domain(0, N)
    for feature_region in I.parallel(
        I.partition(features, extent=I.auto("FEATURE_TILE"))
    ):
        accumulation = I.state_stream(
            rows,
            extent=I.auto("ROW_TILE"),
            init=(
                I.zeros((feature_region,), dtype=I.f32),
                I.zeros((feature_region,), dtype=I.f32),
            ),
        )
        with accumulation:
            for row_region, (dw_value, db_value) in accumulation:
                accumulation.yield_(
                    dw_value
                    + I.reduce.sum(
                        dw_partial[row_region, feature_region],
                        axis=0,
                        identity=0.0,
                    ),
                    db_value
                    + I.reduce.sum(
                        db_partial[row_region, feature_region],
                        axis=0,
                        identity=0.0,
                    ),
                )
        dw_value, db_value = accumulation.result
        dw[feature_region] = dw_value
        db[feature_region] = db_value
