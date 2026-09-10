import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_add_kernel(
    input: I.In[I.f32, ("N", "IC", "IH", "IW")],
    weight: I.In[I.f32, ("OC", "IC", "KH", "KW")],
    other: I.In[I.f32, ("N", "OC", "OH", "OW")],
    alpha: I.f32,
    output: I.Out[I.f32, ("N", "OC", "OH", "OW")],
):
    for n in I.parallel(I.domain(0, other.shape[0])):
        for oc in I.parallel(I.domain(0, weight.shape[0])):
            for oh in I.parallel(I.domain(0, other.shape[2])):
                for ow in I.parallel(I.domain(0, other.shape[3])):
                    acc = I.cast(0.0, I.f32)
                    for ic in I.domain(0, weight.shape[1]):
                        for kh in I.domain(0, weight.shape[2]):
                            ih = oh + kh - 1
                            if ih >= 0:
                                if ih < input.shape[2]:
                                    for kw in I.domain(0, weight.shape[3]):
                                        iw = ow + kw - 1
                                        if iw >= 0:
                                            if iw < input.shape[3]:
                                                acc = acc + input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    output[n, oc, oh, ow] = acc + alpha * other[n, oc, oh, ow]


def build(context):
    compiled = context.compile("conv2d_add_fixed", conv2d_add_kernel)

    def wrapper(input, weight, bias=None, other=None, stride=1, padding=0,
                dilation=1, groups=1, alpha=1, out=None):
        result = out if out is not None else torch.empty_like(other)
        compiled(input, weight, other, alpha, result)
        return result

    return wrapper
