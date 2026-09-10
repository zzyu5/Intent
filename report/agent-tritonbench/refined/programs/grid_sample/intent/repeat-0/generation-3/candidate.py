import torch
import intent
import intent.language as I


@intent.kernel
def grid_sample_bilinear_zeros_false(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    sampling_grid: I.In[I.f32, ("N", "OH", "OW", 2)],
    output: I.Out[I.f32, ("N", "C", "OH", "OW")],
):
    width = I.cast(x.shape[3], I.i64)
    height = I.cast(x.shape[2], I.i64)
    width_f = I.cast(width, I.f32)
    height_f = I.cast(height, I.f32)

    # Scalar logical indices avoid introducing an implicit channel broadcast.
    for batch in I.parallel(I.domain(0, x.shape[0])):
        for channel in I.parallel(I.domain(0, x.shape[1])):
            for out_h in I.parallel(I.domain(0, sampling_grid.shape[1])):
                for out_w in I.parallel(I.domain(0, sampling_grid.shape[2])):
                    gx = sampling_grid[batch, out_h, out_w, 0]
                    gy = sampling_grid[batch, out_h, out_w, 1]

                    # PyTorch treats NaN grid coordinates as -1 before applying
                    # the coordinate transform.
                    gx = I.select(gx != gx, -1.0, gx)
                    gy = I.select(gy != gy, -1.0, gy)

                    # align_corners=False: ((coord + 1) * size - 1) / 2.
                    ix = ((gx + 1.0) * width_f - 1.0) * 0.5
                    iy = ((gy + 1.0) * height_f - 1.0) * 0.5

                    # Keep float-to-index conversion defined for coordinates at
                    # infinity. The clamped halo still produces zero padding.
                    ix = I.minimum(I.maximum(ix, -1.0), width_f)
                    iy = I.minimum(I.maximum(iy, -1.0), height_f)

                    ix0 = I.cast(ix, I.i64)
                    iy0 = I.cast(iy, I.i64)
                    ix0 = I.select(ix < I.cast(ix0, I.f32), ix0 - 1, ix0)
                    iy0 = I.select(iy < I.cast(iy0, I.f32), iy0 - 1, iy0)
                    ix1 = ix0 + 1
                    iy1 = iy0 + 1

                    valid_x0 = (ix0 >= 0) & (ix0 < width)
                    valid_x1 = (ix1 >= 0) & (ix1 < width)
                    valid_y0 = (iy0 >= 0) & (iy0 < height)
                    valid_y1 = (iy1 >= 0) & (iy1 < height)

                    ix0_load = I.minimum(I.maximum(ix0, 0), width - 1)
                    ix1_load = I.minimum(I.maximum(ix1, 0), width - 1)
                    iy0_load = I.minimum(I.maximum(iy0, 0), height - 1)
                    iy1_load = I.minimum(I.maximum(iy1, 0), height - 1)

                    v00 = x[batch, channel, iy0_load, ix0_load]
                    v01 = x[batch, channel, iy0_load, ix1_load]
                    v10 = x[batch, channel, iy1_load, ix0_load]
                    v11 = x[batch, channel, iy1_load, ix1_load]

                    wx = ix - I.cast(ix0, I.f32)
                    wy = iy - I.cast(iy0, I.f32)
                    one_minus_x = 1.0 - wx
                    one_minus_y = 1.0 - wy

                    mx0 = I.cast(valid_x0, I.f32)
                    mx1 = I.cast(valid_x1, I.f32)
                    my0 = I.cast(valid_y0, I.f32)
                    my1 = I.cast(valid_y1, I.f32)

                    row0 = v00 * mx0 * my0 * one_minus_x + v01 * mx1 * my0 * wx
                    row1 = v10 * mx0 * my1 * one_minus_x + v11 * mx1 * my1 * wx
                    output[batch, channel, out_h, out_w] = row0 * one_minus_y + row1 * wy


def build(context):
    compiled = context.compile(
        "grid_sample_bilinear_zeros_false",
        grid_sample_bilinear_zeros_false,
    )

    def wrapper(
        input,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ):
        output = torch.empty(
            (input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]),
            dtype=input.dtype,
            device=input.device,
        )
        compiled(input, grid, output)
        return output

    return wrapper
