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
    weight: I.In[I.f16, ("G", "K", "N")],
    y: I.Out[I.f16, ("R", "N")],
):
    R, _ = x.shape
    G, _, N = weight.shape
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
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


@intent.kernel
def ragged_grouped_gemm_bf16(
    x: I.In[I.bf16, ("R", "K")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    weight: I.In[I.bf16, ("G", "K", "N")],
    y: I.Out[I.bf16, ("R", "N")],
):
    R, _ = x.shape
    G, _, N = weight.shape
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
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
                value=I.cast(result, I.bf16),
            )


@intent.kernel
def ragged_grouped_gemm_backward_weight(
    left: I.In[I.f16, ("R", "K")],
    right: I.In[I.f16, ("R", "N")],
    group_offsets: I.In[I.i32, ("G_PLUS_1",)],
    grad_weight: I.Out[I.f16, ("G", "K", "N")],
):
    R, K = left.shape
    N = right.shape[1]
    G = grad_weight.shape[0]
    groups = I.ragged(
        outer=I.domain(0, G),
        members=I.domain(0, R),
        offsets=group_offsets,
    )
    k_axis = I.domain(0, K)
    n_axis = I.domain(0, N)
    for group in I.parallel(groups.outer):
        for k_region in I.parallel(
            I.partition(k_axis, extent=I.auto("M_TILE"))
        ):
            for n_region in I.parallel(
                I.partition(n_axis, extent=I.auto("N_TILE"))
            ):
                members = groups[group]
                result = I.contract(
                    left[members, k_region],
                    right[members, n_region],
                    reduce=((0, 0),),
                    acc_dtype=I.f32,
                )
                grad_weight[group, k_region, n_region] = I.cast(result, I.f16)


@intent.kernel
def routed_expert_projection_bf16(
    x: I.In[I.bf16, ("T", "K")],
    expert_offsets: I.In[I.i32, ("E_PLUS_1",)],
    member_routes: I.In[I.i32, ("R",)],
    weight: I.In[I.bf16, ("E", "N", "K")],
    output: I.Out[I.bf16, ("T", "TOP_K", "N")],
    TOP_K: I.Constexpr[int],
):
    E = weight.shape[0]
    R = member_routes.shape[0]
    experts = I.ragged(
        outer=I.domain(0, E),
        members=I.domain(0, R),
        offsets=expert_offsets,
        indices=member_routes,
    )
    for expert in I.parallel(experts.outer):
        for route_region in I.parallel(
            I.partition(experts[expert], extent=I.auto("MEMBER_TILE"))
        ):
            routes = I.members(route_region)
            token = routes // TOP_K
            slot = routes % TOP_K
            I.assume_in_bounds(token, x, axis=0)
            values = I.gather(x, index=(token, slice(None)))
            result = I.contract(
                values,
                weight[expert, :, :],
                reduce=((1, 1),),
                acc_dtype=I.f32,
            )
            I.scatter_unique(
                output,
                index=(token, slot, slice(None)),
                value=I.cast(result, I.bf16),
            )


@intent.kernel
def aligned_expert_projection_bf16(
    x: I.In[I.bf16, ("T", "K")],
    sorted_route_ids: I.In[I.i32, ("P",)],
    expert_blocks: I.In[I.i32, ("EB",)],
    total_padded: I.In[I.i32, (1,)],
    weight: I.In[I.bf16, ("E", "N", "K")],
    route_output: I.InOut[I.bf16, ("R", "N")],
    TOP_K: I.Constexpr[int],
    BLOCK_SIZE: I.Constexpr[int],
):
    T, _ = x.shape
    R, N = route_output.shape
    route_offsets = I.domain(0, BLOCK_SIZE)
    for block in I.parallel(I.domain(0, expert_blocks.shape[0])):
        start = block * BLOCK_SIZE
        active_block = start < total_padded[0]
        route_positions = start + I.indices(route_offsets)
        route_ids = sorted_route_ids[route_positions]
        valid_route = active_block and (route_ids >= 0) and (route_ids < R)
        safe_route = I.mask(
            route_ids,
            valid=valid_route,
            fill=I.cast(0, I.i32),
        )
        token = safe_route // TOP_K
        valid_token = valid_route and (token < T)
        values = I.gather(
            x,
            index=(I.cast(token, I.index), slice(None)),
            valid=valid_token[:, None],
            fill=I.cast(0.0, I.bf16),
        )
        expert = expert_blocks[block]
        I.assume_in_bounds(expert, weight, axis=0)
        result = I.contract(
            values,
            weight[expert, :, :],
            reduce=((1, 1),),
            acc_dtype=I.f32,
        )
        columns = I.domain(0, N)
        I.scatter_unique(
            route_output,
            index=(I.cast(safe_route, I.index), columns),
            value=I.cast(result, I.bf16),
            valid=valid_route[:, None],
        )
