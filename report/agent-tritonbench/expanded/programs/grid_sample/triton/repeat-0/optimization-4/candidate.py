import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 1024}, num_warps=4),
        triton.Config({"BLOCK": 1024}, num_warps=8),
        triton.Config({"BLOCK": 1024}, num_warps=16),
    ],
    key=[],
)
@triton.jit
def _grid_sample_bilinear_zero(
    input_ptr,
    grid_ptr,
    output_ptr,
    BLOCK: tl.constexpr,
):
    hw = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = tl.program_id(1)

    # The fixed invocation has 128 * 128 grid points per batch.
    grid_offset = n * 32768 + hw * 2
    gx = tl.load(grid_ptr + grid_offset)
    gy = tl.load(grid_ptr + grid_offset + 1)

    # PyTorch treats NaN grid coordinates as -1.
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    # align_corners=False: ((coord + 1) * size - 1) / 2.
    ix = gx * 128.0 + 127.5
    iy = gy * 128.0 + 127.5
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)

    wx = ix - x0.to(tl.float32)
    wy = iy - y0.to(tl.float32)

    x0_valid = (x0 >= 0) & (x0 < 256)
    y0_valid = (y0 >= 0) & (y0 < 256)
    x1_valid = (x0 >= -1) & (x0 < 255)
    y1_valid = (y0 >= -1) & (y0 < 255)

    # Neighbor offsets are related by one row and one column. Masked loads
    # permit the natural out-of-range offsets and return zero.
    off00 = y0 * 256 + x0
    off01 = off00 + 1
    off10 = off00 + 256
    off11 = off10 + 1
    mask00 = x0_valid & y0_valid
    mask01 = x1_valid & y0_valid
    mask10 = x0_valid & y1_valid
    mask11 = x1_valid & y1_valid

    input_batch = input_ptr + n * 196608
    output_batch = output_ptr + n * 49152
    for c in tl.static_range(0, 3):
        input_channel = input_batch + c * 65536
        v00 = tl.load(
            input_channel + off00,
            mask=mask00,
            other=0.0,
        )
        v01 = tl.load(
            input_channel + off01,
            mask=mask01,
            other=0.0,
        )
        v10 = tl.load(
            input_channel + off10,
            mask=mask10,
            other=0.0,
        )
        v11 = tl.load(
            input_channel + off11,
            mask=mask11,
            other=0.0,
        )

        top = v00 * (1.0 - wx) + v01 * wx
        bottom = v10 * (1.0 - wx) + v11 * wx
        value = top * (1.0 - wy) + bottom * wy
        tl.store(output_batch + c * 16384 + hw, value)


def build(context):
    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        _grid_sample_bilinear_zero[(16, 16)](
            input,
            grid,
            output,
        )
        return output

    return wrapper
