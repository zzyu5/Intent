import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1, "BLOCK_W": 64}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_H": 1, "BLOCK_W": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 2, "BLOCK_W": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1, "BLOCK_W": 128}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_H": 1, "BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 2, "BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 128}, num_warps=8, num_stages=1),
    ],
    key=[],
)
@triton.jit
def grid_sample_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    tiles_w = 128 // BLOCK_W
    tiles_h = 128 // BLOCK_H
    tile = pid % (tiles_h * tiles_w)
    batch = pid // (tiles_h * tiles_w)
    tile_h = tile // tiles_w
    tile_w = tile % tiles_w

    oh = (tile_h * BLOCK_H + tl.arange(0, BLOCK_H))[:, None]
    ow = (tile_w * BLOCK_W + tl.arange(0, BLOCK_W))[None, :]

    grid_offset = batch * 32768 + oh * 256 + ow * 2
    gx = tl.load(grid_ptr + grid_offset)
    gy = tl.load(grid_ptr + grid_offset + 1)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    ix = (gx + 1.0) * 128.0 - 0.5
    iy = (gy + 1.0) * 128.0 - 0.5
    valid_x = ix >= -1.0
    valid_y = iy >= -1.0

    # Clamp before conversion so arbitrary finite coordinates have defined
    # integer indices. The range checks below preserve zero padding behavior.
    ix = tl.maximum(tl.minimum(ix, 256.0), -1.0)
    iy = tl.maximum(tl.minimum(iy, 256.0), -1.0)
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = ix - x0.to(tl.float32)
    wy = iy - y0.to(tl.float32)

    valid_x0 = (x0 >= 0) & (x0 < 256)
    valid_x1 = x1 < 256
    valid_y0 = (y0 >= 0) & (y0 < 256)
    valid_y1 = y1 < 256
    mask00 = valid_x & valid_y & valid_x0 & valid_y0
    mask01 = valid_x & valid_y & valid_x1 & valid_y0
    mask10 = valid_x & valid_y & valid_x0 & valid_y1
    mask11 = valid_x & valid_y & valid_x1 & valid_y1

    input_base = input_ptr + batch * 196608
    output_base = output_ptr + batch * 49152 + oh * 128 + ow
    for channel in range(3):
        channel_input = input_base + channel * 65536
        v00 = tl.load(channel_input + y0 * 256 + x0, mask=mask00, other=0.0)
        v01 = tl.load(channel_input + y0 * 256 + x1, mask=mask01, other=0.0)
        v10 = tl.load(channel_input + y1 * 256 + x0, mask=mask10, other=0.0)
        v11 = tl.load(channel_input + y1 * 256 + x1, mask=mask11, other=0.0)

        top = v00 + wx * (v01 - v00)
        bottom = v10 + wx * (v11 - v10)
        values = top + wy * (bottom - top)
        tl.store(output_base + channel * 16384, values)


def build(context):
    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        grid_sample_kernel[
            lambda meta: (16 * (128 // meta["BLOCK_H"]) * (128 // meta["BLOCK_W"]),)
        ](input, grid, output)
        return output

    return wrapper
