from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _symmetric_product(
    A: I.In[I.f32, ("M", "K")],
    P: I.Out[I.f32, ("M", "M")],
    TILE: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
):
    M, K = A.shape
    rows = I.domain(0, M)
    cols = I.domain(0, M)
    k_axis = I.domain(0, K)
    tile_ids = I.domain(0, GROUPS)

    for tile_row in I.parallel(tile_ids):
        row_region = rows[tile_row * TILE : (tile_row + 1) * TILE]
        for tile_col in I.parallel(tile_ids):
            col_region = cols[tile_col * TILE : (tile_col + 1) * TILE]
            lhs = A[row_region, k_axis]
            rhs = A[col_region, k_axis]
            P[row_region, col_region] = I.matmul(
                lhs,
                rhs,
                acc_dtype=I.f32,
                transpose_rhs=True,
            )


@intent.kernel
def _scaled_tiles_and_partial_abs_sum(
    P: I.In[I.f32, ("M", "M")],
    C: I.In[I.f32, ("M", "M")],
    partials: I.Out[I.f32, ("G", "G")],
    alpha: I.f32,
    beta: I.f32,
    TILE: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
):
    M, N = P.shape
    rows = I.domain(0, M)
    cols = I.domain(0, N)
    tile_ids = I.domain(0, GROUPS)

    for tile_row in I.parallel(tile_ids):
        row_region = rows[tile_row * TILE : (tile_row + 1) * TILE]
        for tile_col in I.parallel(tile_ids):
            col_region = cols[tile_col * TILE : (tile_col + 1) * TILE]
            value = alpha * P[row_region, col_region] + beta * C[row_region, col_region]
            partials[tile_row, tile_col] = I.reduce.sum(
                I.abs(value),
                axis=(0, 1),
                acc_dtype=I.f32,
            )


@intent.kernel
def _reduce_partials(
    partials: I.In[I.f32, ("G", "G")],
    result: I.Out[I.f32, ()],
    GROUPS: I.Constexpr[int],
):
    tiles = I.domain(0, GROUPS)
    result[()] = I.reduce.sum(
        partials[tiles, tiles],
        axis=(0, 1),
        acc_dtype=I.f32,
    )


def build(context):
    tile = 128
    groups = 8
    product = context.compile(
        "symmetric_product_tiles",
        _symmetric_product,
        constexprs={"TILE": tile, "GROUPS": groups},
    )
    scaled = context.compile(
        "scaled_tiles_partial_abs_sum",
        _scaled_tiles_and_partial_abs_sum,
        constexprs={"TILE": tile, "GROUPS": groups},
    )
    reduction = context.compile(
        "reduce_partial_abs_sums",
        _reduce_partials,
        constexprs={"GROUPS": groups},
    )

    def symmetric_mm_and_abs_sum(A: torch.Tensor, C: torch.Tensor, alpha: float = 1.0, beta: float = 0.5) -> torch.Tensor:
        product_values = torch.empty(
            (A.shape[0], A.shape[0]),
            device=A.device,
            dtype=torch.float32,
        )
        partials = torch.empty(
            (groups, groups),
            device=A.device,
            dtype=torch.float32,
        )
        result = torch.empty((), device=A.device, dtype=torch.float32)

        product(A, product_values)
        scaled(product_values, C, partials, alpha, beta)
        reduction(partials, result)
        return result

    return symmetric_mm_and_abs_sum
