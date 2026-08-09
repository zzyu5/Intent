import intent
import intent.language as I


ROWS = 8192
K = 4096
N = 4096
GROUPS = 8


@intent.kernel
def ragged_grouped_gemm(
    x: I.In[I.f16, ("R", "K")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    member_rows: I.In[I.i32, ("R",)],
    weight: I.In[I.f16, ("G", "K", "N")],
    y: I.Out[I.f16, ("R", "N")],
):
    G, _, N = weight.shape
    groups = I.ragged(
        outer=I.domain(0, G),
        offsets=group_offsets,
        indices=member_rows,
    )
    for group in I.parallel(groups.outer):
        for member_region in I.parallel(
            I.partition(groups[group], extent=I.auto("MEMBER_TILE"))
        ):
            rows = I.members(member_region)
            values = I.gather(x, index=(rows, slice(None)))
            result = I.contract(
                values,
                weight[group, :, :],
                reduce=((1, 0),),
                acc_dtype=I.f32,
            )
            I.scatter_unique(
                y,
                index=(rows, slice(None)),
                value=I.cast(result, I.f16),
            )
