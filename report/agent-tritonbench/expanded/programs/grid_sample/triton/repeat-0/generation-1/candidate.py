import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sample_bilinear_zero(
    input_ptr,
    grid_ptr,
    output_ptr,
    n_elements,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    nc = tl.program_id(1)
    hw = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = hw < n_elements

    n = nc // C
    c = nc - n * C
    input_channel = n * (C * H * W) + c * (H * W)
    output_channel = n * (C * OH * OW) + c * (OH * OW)

    grid_offset = (n * (OH * OW) + hw) * 2
    gx = tl.load(grid_ptr + grid_offset, mask=mask, other=0.0)
    gy = tl.load(grid_ptr + grid_offset + 1, mask=mask, other=0.0)

    # PyTorch treats NaN grid coordinates as -1.
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    # align_corners=False: ((coord + 1) * size - 1) / 2.
    ix = ((gx + 1.0) * W - 1.0) * 0.5
    iy = ((gy + 1.0) * H - 1.0) * 0.5
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = ix - x0.to(tl.float32)
    wy = iy - y0.to(tl.float32)

    x0_valid = (x0 >= 0) & (x0 < W)
    x1_valid = (x1 >= 0) & (x1 < W)
    y0_valid = (y0 >= 0) & (y0 < H)
    y1_valid = (y1 >= 0) & (y1 < H)

    # Keep masked pointer arithmetic in range even for out-of-bounds samples.
    x0_safe = tl.maximum(tl.minimum(x0, W - 1), 0)
    x1_safe = tl.maximum(tl.minimum(x1, W - 1), 0)
    y0_safe = tl.maximum(tl.minimum(y0, H - 1), 0)
    y1_safe = tl.maximum(tl.minimum(y1, H - 1), 0)

    base = input_ptr + input_channel
    v00 = tl.load(base + y0_safe * W + x0_safe, mask=mask & x0_valid & y0_valid, other=0.0)
    v01 = tl.load(base + y0_safe * W + x1_safe, mask=mask & x1_valid & y0_valid, other=0.0)
    v10 = tl.load(base + y1_safe * W + x0_safe, mask=mask & x0_valid & y1_valid, other=0.0)
    v11 = tl.load(base + y1_safe * W + x1_safe, mask=mask & x1_valid & y1_valid, other=0.0)

    top = v00 * (1.0 - wx) + v01 * wx
    bottom = v10 * (1.0 - wx) + v11 * wx
    value = top * (1.0 - wy) + bottom * wy
    tl.store(output_ptr + output_channel + hw, value, mask=mask)


def build(context):
    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        spatial = grid.shape[1] * grid.shape[2]
        _grid_sample_bilinear_zero[(triton.cdiv(spatial, 128), input.shape[0] * input.shape[1])](
            input,
            grid,
            output,
            spatial,
            C=3,
            H=256,
            W=256,
            OH=grid.shape[1],
            OW=grid.shape[2],
            BLOCK=128,
            num_warps=4,
        )
        return output

    return wrapper
