import intent
import intent.language as I


ROWS = 8192
COLUMNS = 8192
ROW_MAJOR_NOALIAS = I.constraints(
    strides=(None, 1),
    layout="row_major",
    noalias=True,
)


@intent.kernel
def stable_softmax(
    x: I.In[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
    y: I.Out[I.f32, ("M", "N"), ROW_MAJOR_NOALIAS],
):
    M, N = x.shape
    columns = I.domain(0, N)
    for row in I.parallel(I.domain(0, M)):
        values = x[row, columns]
        maximum = I.reduce.max(values, axis=0, identity=-I.inf)
        numerator = I.exp(values - maximum)
        denominator = I.reduce.sum(numerator, axis=0, identity=0.0)
        y[row, columns] = numerator / denominator
