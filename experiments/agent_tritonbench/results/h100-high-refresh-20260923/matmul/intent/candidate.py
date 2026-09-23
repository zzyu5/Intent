import torch
import intent
import intent.language as I


TILE_M = 128
TILE_N = 128
GRID_M = 8
GRID_N = 8


@intent.kernel
def tiled_matmul(
    lhs: I.In[I.f16, (1024, 1024)],
    rhs: I.In[I.f16, (1024, 1024)],
    out: I.Out[I.f16, (1024, 1024)],
):
    all_rows = I.domain(0, 1024)
    all_cols = I.domain(0, 1024)
    reduction = I.domain(0, 1024)

    # Each output tile is independent; the contraction keeps fp32 accumulation.
    for tile_m in I.parallel(I.domain(0, GRID_M)):
        rows = all_rows[tile_m * TILE_M : (tile_m + 1) * TILE_M]
        for tile_n in I.parallel(I.domain(0, GRID_N)):
            cols = all_cols[tile_n * TILE_N : (tile_n + 1) * TILE_N]
            lhs_tile = lhs[rows, reduction]
            rhs_tile = rhs[reduction, cols]
            tile = I.matmul(lhs_tile, rhs_tile, acc_dtype=I.f32)
            out[rows, cols] = I.cast(tile, I.f16)


def build(context):
    matmul_artifact = context.compile("tiled_matmul_1024", tiled_matmul)

    def matmul(tensor1, tensor2):
        result = torch.empty(
            (1024, 1024), device=tensor1.device, dtype=tensor1.dtype
        )
        matmul_artifact(tensor1, tensor2, result)
        return result

    return matmul
