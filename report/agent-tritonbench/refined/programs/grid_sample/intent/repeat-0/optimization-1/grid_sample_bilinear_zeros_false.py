import torch
import triton
import triton.language as tl


@triton.jit
def _grid_sample_kernel(
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
    grid_base = batch * 32768 + out_h * 256 + w * 2

    gx = tl.load(grid_ptr + grid_base)
    gy = tl.load(grid_ptr + grid_base + 1)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    ix = (gx + 1.0) * 128.0 - 0.5
    iy = (gy + 1.0) * 128.0 - 0.5
    x_in = ix >= -1.0
    y_in = iy >= -1.0

    # Clamp before conversion so very large finite coordinates cannot produce
    # undefined integer casts.  The original range checks remain in the masks.
    ix = tl.maximum(tl.minimum(ix, 256.0), -1.0)
    iy = tl.maximum(tl.minimum(iy, 256.0), -1.0)
    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = ix - x0.to(tl.float32)
    wy = iy - y0.to(tl.float32)

    c_off = c[:, None]
    x0_off = x0[None, :]
    x1_off = x1[None, :]
    y0_off = y0[None, :]
    y1_off = y1[None, :]
    valid_c = c_off < 3
    valid_x0 = (x0_off >= 0) & (x0_off < 256)
    valid_x1 = (x1_off >= 0) & (x1_off < 256)
    valid_y0 = (y0_off >= 0) & (y0_off < 256)
    valid_y1 = (y1_off >= 0) & (y1_off < 256)
    valid_x = x_in[None, :]
    valid_y = y_in[None, :]

    input_base = input_ptr + batch * 196608 + c_off * 65536
    v00 = tl.load(
        input_base + y0_off * 256 + x0_off,
        mask=valid_c & valid_x & valid_y & valid_x0 & valid_y0,
        other=0.0,
    )
    v01 = tl.load(
        input_base + y0_off * 256 + x1_off,
        mask=valid_c & valid_x & valid_y & valid_x1 & valid_y0,
        other=0.0,
    )
    v10 = tl.load(
        input_base + y1_off * 256 + x0_off,
        mask=valid_c & valid_x & valid_y & valid_x0 & valid_y1,
        other=0.0,
    )
    v11 = tl.load(
        input_base + y1_off * 256 + x1_off,
        mask=valid_c & valid_x & valid_y & valid_x1 & valid_y1,
        other=0.0,
    )

    wx = wx[None, :]
    wy = wy[None, :]
    one_minus_x = 1.0 - wx
    one_minus_y = 1.0 - wy
    output = (
        v00 * one_minus_x * one_minus_y
        + v01 * wx * one_minus_y
        + v10 * one_minus_x * wy
        + v11 * wx * wy
    )

    output_ptr = output_ptr + batch * 49152 + c_off * 16384 + out_h * 128 + w[None, :]
    tl.store(output_ptr, output, mask=valid_c)


def launch(input, sampling_grid, output):
    _grid_sample_kernel[(2048,)](
        input,
        sampling_grid,
        output,
        BLOCK_W=128,
        BLOCK_C=4,
        num_warps=4,
    )


def run(input, sampling_grid):
    output = torch.empty(
        (input.shape[0], input.shape[1], sampling_grid.shape[1], sampling_grid.shape[2]),
        device=input.device,
        dtype=input.dtype,
    )
    launch(input, sampling_grid, output)
    return output
