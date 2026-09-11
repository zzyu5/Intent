import torch
import triton
import triton.language as tl


@triton.jit
def grid_sample_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    out_h = pid % 128
    batch = pid // 128

    w = tl.arange(0, BLOCK_W)
    c = tl.arange(0, BLOCK_C)

    grid_base = grid_ptr + batch * 32768 + out_h * 256 + w * 2
    gx = tl.load(grid_base)
    gy = tl.load(grid_base + 1)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    ix = (gx + 1.0) * 128.0 - 0.5
    iy = (gy + 1.0) * 128.0 - 0.5
    x_in = ix >= -1.0
    y_in = iy >= -1.0

    # Keep the integer conversion defined for arbitrary finite coordinates.
    ix = tl.maximum(tl.minimum(ix, 256.0), -1.0)
    iy = tl.maximum(tl.minimum(iy, 256.0), -1.0)
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = ix - x0.to(tl.float32)
    wy = iy - y0.to(tl.float32)

    c = c[:, None]
    x0 = x0[None, :]
    x1 = x1[None, :]
    y0 = y0[None, :]
    y1 = y1[None, :]
    valid_c = c < 3
    valid_x = x_in[None, :]
    valid_y = y_in[None, :]

    input_base = input_ptr + batch * 196608 + c * 65536
    v00 = tl.load(
        input_base + y0 * 256 + x0,
        mask=valid_c & valid_x & valid_y & (x0 >= 0) & (x0 < 256) & (y0 >= 0) & (y0 < 256),
        other=0.0,
    )
    v01 = tl.load(
        input_base + y0 * 256 + x1,
        mask=valid_c & valid_x & valid_y & (x1 >= 0) & (x1 < 256) & (y0 >= 0) & (y0 < 256),
        other=0.0,
    )
    v10 = tl.load(
        input_base + y1 * 256 + x0,
        mask=valid_c & valid_x & valid_y & (x0 >= 0) & (x0 < 256) & (y1 >= 0) & (y1 < 256),
        other=0.0,
    )
    v11 = tl.load(
        input_base + y1 * 256 + x1,
        mask=valid_c & valid_x & valid_y & (x1 >= 0) & (x1 < 256) & (y1 >= 0) & (y1 < 256),
        other=0.0,
    )

    wx = wx[None, :]
    wy = wy[None, :]
    top = v00 + wx * (v01 - v00)
    bottom = v10 + wx * (v11 - v10)
    values = top + wy * (bottom - top)

    output_ptr = output_ptr + batch * 49152 + c * 16384 + out_h * 128 + w[None, :]
    tl.store(output_ptr, values, mask=valid_c)


def build(context):
    def wrapper(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        grid_sample_kernel[(2048,)](
            input,
            grid,
            output,
            BLOCK_W=128,
            BLOCK_C=4,
            num_warps=4,
        )
        return output

    return wrapper
