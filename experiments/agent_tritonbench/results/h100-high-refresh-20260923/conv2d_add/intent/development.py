import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_add(
    input: I.In[I.f32, ("B", "IC", "H", "W")],
    weight: I.In[I.f32, ("OC", "IC", "KH", "KW")],
    other: I.In[I.f32, ("B", "OC", "OH", "OW")],
    output: I.Out[I.f32, ("B", "OC", "OH", "OW")],
    stride_h: I.index, stride_w: I.index,
    pad_h: I.index, pad_w: I.index,
    dilation_h: I.index, dilation_w: I.index,
    alpha: I.f32,
):
    B, IC, H, W = input.shape
    OC, _, KH, KW = weight.shape
    _, _, OH, OW = output.shape
    batch = I.domain(0, B)
    channels = I.domain(0, IC)
    filters = I.domain(0, OC)
    rows = I.domain(0, OH)
    columns = I.domain(0, OW)
    kernel_rows = I.domain(0, KH)
    kernel_columns = I.domain(0, KW)
    r = I.reshape(I.indices(rows), (OH, 1, 1, 1))
    c = I.reshape(I.indices(columns), (1, OW, 1, 1))
    kr = I.reshape(I.indices(kernel_rows), (1, 1, KH, 1))
    kc = I.reshape(I.indices(kernel_columns), (1, 1, 1, KW))
    ir = r * stride_h - pad_h + kr * dilation_h
    ic = c * stride_w - pad_w + kc * dilation_w
    row_valid = (ir >= 0) & (ir < H)
    column_valid = (ic >= 0) & (ic < W)
    safe_r = I.select(row_valid, ir, I.cast(0, I.index))
    safe_c = I.select(column_valid, ic, I.cast(0, I.index))
    samples = input[batch, channels, safe_r, safe_c]
    samples = I.select(row_valid & column_valid, samples, I.cast(0.0, I.f32))
    samples = I.reshape(samples, (B, 1, IC, OH, OW, KH, KW))
    weights = weight[filters, channels, kernel_rows, kernel_columns]
    weights = I.reshape(weights, (1, OC, IC, 1, 1, KH, KW))
    convolution = I.reduce.sum(samples * weights, axis=(2, 5, 6), acc_dtype=I.f32)
    output[batch, filters, rows, columns] = (
        convolution + alpha * other[batch, filters, rows, columns]
    )


def _pair(value):
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build(context):
    kernel = context.compile("conv2d_add", _conv2d_add)

    def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0,
                   dilation=1, groups=1, alpha=1, out=None):
        if bias is not None or groups != 1:
            raise NotImplementedError("the fixed profile has no bias and one group")
        if not isinstance(other, torch.Tensor):
            raise NotImplementedError("the fixed profile provides a tensor addend")
        sh, sw = _pair(stride)
        dh, dw = _pair(dilation)
        kh, kw = weight.shape[2:]
        if isinstance(padding, str):
            if padding == "same":
                ph, pw = (kh - 1) // 2, (kw - 1) // 2
            elif padding == "valid":
                ph, pw = 0, 0
            else:
                raise ValueError("unsupported padding mode")
        else:
            ph, pw = _pair(padding)
        oh = (input.shape[2] + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (input.shape[3] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        if out is None:
            out = torch.empty((input.shape[0], weight.shape[0], oh, ow),
                              dtype=input.dtype, device=input.device)
        kernel(input, weight, other, out, sh, sw, ph, pw, dh, dw, float(alpha))
        return out

    return conv2d_add
