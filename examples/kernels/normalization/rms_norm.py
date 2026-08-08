import intent
import intent.language as I


ROWS = 8192
FEATURES = 4096


@intent.kernel
def weighted_rms_norm(
    x: I.In[I.f32, ("M", "N")],
    weight: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("M", "N")],
    inverse_features: I.f32,
    epsilon: I.f32,
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        mean_square = (
            I.reduce.sum(values * values, axis=0, identity=0.0)
            * inverse_features
        )
        y[row, columns] = values * I.rsqrt(mean_square + epsilon) * weight[columns]
