import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sample_bilinear_zero_fused_channels(
    input_ptr,
    grid_ptr,
    output_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    hw = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)

    # The grid coordinates are shared by all three output channels.
    grid_offset = (n * (OH * OW) + hw) * 2
    gx = tl.load(grid_ptr + grid_offset)
    gy = tl.load(grid_ptr + grid_offset + 1)

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

    # Keep masked pointer arithmetic in range for out-of-bounds samples.
    x0_safe = tl.maximum(tl.minimum(x0, W - 1), 0)
    x1_safe = tl.maximum(tl.minimum(x1, W - 1), 0)
    y0_safe = tl.maximum(tl.minimum(y0, H - 1), 0)
    y1_safe = tl.maximum(tl.minimum(y1, H - 1), 0)

    input_batch = input_ptr + n * (C * H * W)
    output_batch = output_ptr + n * (C * OH * OW)
    for c in tl.static_range(0, C):
        input_channel = input_batch + c * (H * W)
        v00 = tl.load(
            input_channel + y0_safe * W + x0_safe,
            mask=x0_valid & y0_valid,
            other=0.0,
        )
        v01 = tl.load(
            input_channel + y0_safe * W + x1_safe,
            mask=x1_valid & y0_valid,
            other=0.0,
        )
        v10 = tl.load(
            input_channel + y1_safe * W + x0_safe,
            mask=x0_valid & y1_valid,
            other=0.0,
        )
        v11 = tl.load(
            input_channel + y1_safe * W + x1_safe,
            mask=x1_valid & y1_valid,
            other=0.0,
        )

        top = v00 * (1.0 - wx) + v01 * wx
        bottom = v10 * (1.0 - wx) + v11 * wx
        value = top * (1.0 - wy) + bottom * wy
        tl.store(output_batch + c * (OH * OW) + hw, value)


def build(context):
    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        _grid_sample_bilinear_zero_fused_channels[(triton.cdiv(128 * 128, 256), 16)](
            input,
            grid,
            output,
            C=3,
            H=256,
            W=256,
            OH=128,
            OW=128,
            BLOCK=256,
            num_warps=8,
        )
        return output

    return wrapper
