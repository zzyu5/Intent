import intent
import intent.language as I
import torch


@intent.kernel
def conv2d_add_kernel(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    weight: I.In[I.f32, ("K", "C", "R", "S")],
    other: I.In[I.f32, ("N", "K", "OH", "OW")],
    output: I.Out[I.f32, ("N", "K", "OH", "OW")],
):
    batches = I.domain(0, input.shape[0])
    output_channels = I.domain(0, weight.shape[0])
    output_rows = I.domain(0, other.shape[2])
    output_columns = I.domain(0, other.shape[3])
    input_channels = I.domain(0, input.shape[1])
    kernel_rows = I.domain(0, weight.shape[2])
    kernel_columns = I.domain(0, weight.shape[3])

    for n in I.parallel(batches):
        for k in I.parallel(output_channels):
            for oh in I.parallel(output_rows):
                for ow in I.parallel(output_columns):
                    acc = I.cast(0, I.f32)
                    for c in input_channels:
                        for kh in kernel_rows:
                            for kw in kernel_columns:
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if ih >= 0:
                                    if ih < input.shape[2]:
                                        if iw >= 0:
                                            if iw < input.shape[3]:
                                                acc = acc + input[n, c, ih, iw] * weight[k, c, kh, kw]
                    output[n, k, oh, ow] = acc + other[n, k, oh, ow]


def build(context):
    compiled = context.compile("conv2d_add", conv2d_add_kernel)

    def wrapper(input, weight, bias=None, other=None, stride=1, padding=0,
                dilation=1, groups=1, alpha=1, out=None):
        if bias is not None:
            raise NotImplementedError("fixed profile has no bias")
        if other is None:
            raise NotImplementedError("fixed profile supplies other")
        if stride != 1 or padding != 1 or dilation != 1 or groups != 1 or alpha != 1:
            raise NotImplementedError("fixed profile uses unit stride, dilation, and alpha")

        result = out if out is not None else torch.empty_like(other)
        compiled(input, weight, other, result)
        return result

    return wrapper
