import intent
import intent.language as I


BATCH = 8
HEADS = 32
SPLITS = 16
HEAD_DIMENSION = 128


@intent.kernel
def splitk_attention_reduce(
    partial: I.In[I.bf16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.bf16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for dimension_region in I.parallel(
                I.partition(dimensions, extent=I.auto("D_TILE"))
            ):
                lse = partial_lse[batch, head, splits]
                maximum = I.reduce.max(
                    lse,
                    axis=0,
                    identity=-I.inf,
                    acc_dtype=I.f32,
                )
                weights = I.exp2(lse - maximum)
                denominator = I.reduce.sum(
                    weights,
                    axis=0,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                numerator = I.reshape(
                    I.contract(
                        I.reshape(weights, (1, S)),
                        I.cast(
                            partial[batch, head, splits, dimension_region], I.f32
                        ),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    ),
                    (dimension_region,),
                )
                output[batch, head, dimension_region] = I.cast(
                    numerator / denominator,
                    I.bf16,
                )


@intent.kernel
def splitk_attention_reduce_f16(
    partial: I.In[I.f16, ("B", "H", "S", "D")],
    partial_lse: I.In[I.f32, ("B", "H", "S")],
    output: I.Out[I.f16, ("B", "H", "D")],
):
    B, H, S, D = partial.shape
    splits = I.domain(0, S)
    dimensions = I.domain(0, D)
    for batch in I.parallel(I.domain(0, B)):
        for head in I.parallel(I.domain(0, H)):
            for dimension_region in I.parallel(
                I.partition(dimensions, extent=I.auto("D_TILE"))
            ):
                lse = partial_lse[batch, head, splits]
                maximum = I.reduce.max(
                    lse,
                    axis=0,
                    identity=-I.inf,
                    acc_dtype=I.f32,
                )
                weights = I.exp2(lse - maximum)
                denominator = I.reduce.sum(
                    weights,
                    axis=0,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                numerator = I.reshape(
                    I.contract(
                        I.reshape(weights, (1, S)),
                        I.cast(
                            partial[batch, head, splits, dimension_region], I.f32
                        ),
                        reduce=((1, 0),),
                        acc_dtype=I.f32,
                    ),
                    (dimension_region,),
                )
                output[batch, head, dimension_region] = I.cast(
                    numerator / denominator,
                    I.f16,
                )
