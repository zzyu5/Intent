import torch
import intent
import intent.language as I


@intent.kernel
def _grid_sample_bilinear_zero(
    input: I.In[I.f32, (16, 3, 256, 256)],
    grid: I.In[I.f32, (16, 128, 128, 2)],
    output: I.Out[I.f32, (16, 3, 128, 128)],
):
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    neg_one = I.cast(-1.0, I.f32)
    width = I.cast(128.0, I.f32)
    height = I.cast(128.0, I.f32)

    for n in I.parallel(I.domain(0, 16)):
        for oh in I.parallel(I.domain(0, 128)):
            for ow in I.parallel(I.domain(0, 128)):
                gx0 = grid[n, oh, ow, 0]
                gy0 = grid[n, oh, ow, 1]

                # PyTorch grid_sample maps a NaN grid coordinate to -1.
                gx = I.select(gx0 != gx0, neg_one, gx0)
                gy = I.select(gy0 != gy0, neg_one, gy0)

                ix = (gx + one) * width - half
                iy = (gy + one) * height - half
                x0f = I.floor(ix)
                y0f = I.floor(iy)
                x0 = I.cast(x0f, I.index)
                y0 = I.cast(y0f, I.index)
                x1 = x0 + 1
                y1 = y0 + 1
                wx = ix - I.cast(x0, I.f32)
                wy = iy - I.cast(y0, I.f32)

                w00 = (one - wx) * (one - wy)
                w01 = wx * (one - wy)
                w10 = (one - wx) * wy
                w11 = wx * wy

                for c in I.parallel(I.domain(0, 3)):
                    v00 = zero
                    v01 = zero
                    v10 = zero
                    v11 = zero

                    if y0 >= 0:
                        if y0 < 256:
                            if x0 >= 0:
                                if x0 < 256:
                                    v00 = input[n, c, y0, x0]
                            if x1 >= 0:
                                if x1 < 256:
                                    v01 = input[n, c, y0, x1]
                    if y1 >= 0:
                        if y1 < 256:
                            if x0 >= 0:
                                if x0 < 256:
                                    v10 = input[n, c, y1, x0]
                            if x1 >= 0:
                                if x1 < 256:
                                    v11 = input[n, c, y1, x1]

                    value = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
                    output[n, c, oh, ow] = value


def build(context):
    compiled = context.compile("grid_sample_bilinear_zero", _grid_sample_bilinear_zero)

    def grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False):
        output = torch.empty((16, 3, 128, 128), device=input.device, dtype=input.dtype)
        compiled(input, grid, output)
        return output

    return grid_sample
