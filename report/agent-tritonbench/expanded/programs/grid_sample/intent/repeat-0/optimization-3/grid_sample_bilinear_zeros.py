import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 1, "BLOCK_W": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 2, "BLOCK_W": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 2, "BLOCK_W": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 4, "BLOCK_W": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 8, "BLOCK_W": 32}, num_warps=8, num_stages=2),
    ],
    key=["H", "W", "C", "OH", "OW"],
)
@triton.jit
def _grid_sample_kernel(
    input_ptr,
    grid_ptr,
    output_ptr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_y = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)[:, None]
    pid_x = tl.program_id(2)
    offs_w = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)[None, :]

    grid_offset = (pid_n * OH + pid_y) * OW * 2 + offs_w * 2
    # Every configured tile divides the fixed 128x128 output exactly.
    gx = tl.load(grid_ptr + grid_offset)
    gy = tl.load(grid_ptr + grid_offset + 1)
    gx = tl.where(gx != gx, -1.0, gx)
    gy = tl.where(gy != gy, -1.0, gy)

    ix = (gx + 1.0) * (W * 0.5) - 0.5
    iy = (gy + 1.0) * (H * 0.5) - 0.5
    ix = tl.maximum(tl.minimum(ix, W), -1.0)
    iy = tl.maximum(tl.minimum(iy, H), -1.0)

    x0 = tl.floor(ix).to(tl.int32)
    y0 = tl.floor(iy).to(tl.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    dx = ix - x0.to(tl.float32)
    dy = iy - y0.to(tl.float32)

    x0_valid = (x0 >= 0) & (x0 < W)
    x1_valid = x1 < W
    y0_valid = (y0 >= 0) & (y0 < H)
    y1_valid = y1 < H
    m00 = x0_valid & y0_valid
    m01 = x1_valid & y0_valid
    m10 = x0_valid & y1_valid
    m11 = x1_valid & y1_valid

    row00 = y0 * W + x0
    row01 = y0 * W + x1
    row10 = y1 * W + x0
    row11 = y1 * W + x1
    out_offset = (pid_n * C) * OH * OW + pid_y * OW + offs_w
    input_plane = H * W
    input_batch = pid_n * C * input_plane

    for c in range(C):
        input_offset = c * input_plane
        v00 = tl.load(input_ptr + input_batch + input_offset + row00, mask=m00, other=0.0)
        v01 = tl.load(input_ptr + input_batch + input_offset + row01, mask=m01, other=0.0)
        v10 = tl.load(input_ptr + input_batch + input_offset + row10, mask=m10, other=0.0)
        v11 = tl.load(input_ptr + input_batch + input_offset + row11, mask=m11, other=0.0)
        top = tl.fma(v01 - v00, dx, v00)
        bottom = tl.fma(v11 - v10, dx, v10)
        value = tl.fma(bottom - top, dy, top)
        tl.store(output_ptr + out_offset + c * OH * OW, value)


def launch(input, sample_grid, output):
    H = input.shape[2]
    W = input.shape[3]
    C = input.shape[1]
    OH = sample_grid.shape[1]
    OW = sample_grid.shape[2]
    grid = lambda META: (
        input.shape[0],
        triton.cdiv(OH, META["BLOCK_H"]),
        triton.cdiv(OW, META["BLOCK_W"]),
    )
    _grid_sample_kernel[grid](
        input,
        sample_grid,
        output,
        H=H,
        W=W,
        C=C,
        OH=OH,
        OW=OW,
    )


def run(input, sample_grid):
    output = torch.empty(
        (input.shape[0], input.shape[1], sample_grid.shape[1], sample_grid.shape[2]),
        device=input.device,
        dtype=torch.float32,
    )
    launch(input, sample_grid, output)
    return output
