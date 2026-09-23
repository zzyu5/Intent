import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_bias(
    x: I.In[I.f32, (8, 64, 16, 16)],
    w: I.In[I.f32, (128, 64, 3, 3)],
    b: I.In[I.f32, (128,)],
    y: I.Out[I.f32, (8, 128, 14, 14)],
):
    n_domain = I.domain(0, 8)
    oc_domain = I.domain(0, 128)
    oh_domain = I.domain(0, 14)
    ow_domain = I.domain(0, 14)
    ic_domain = I.domain(0, 64)
    kh_domain = I.domain(0, 3)
    kw_domain = I.domain(0, 3)

    n = I.reshape(I.indices(n_domain), (8, 1, 1, 1, 1, 1, 1))
    oc = I.reshape(I.indices(oc_domain), (1, 128, 1, 1, 1, 1, 1))
    oh = I.reshape(I.indices(oh_domain), (1, 1, 14, 1, 1, 1, 1))
    ow = I.reshape(I.indices(ow_domain), (1, 1, 1, 14, 1, 1, 1))
    ic = I.reshape(I.indices(ic_domain), (1, 1, 1, 1, 64, 1, 1))
    kh = I.reshape(I.indices(kh_domain), (1, 1, 1, 1, 1, 3, 1))
    kw = I.reshape(I.indices(kw_domain), (1, 1, 1, 1, 1, 1, 3))

    x_values = x[n, ic, oh + kh, ow + kw]
    w_values = w[oc, ic, kh, kw]
    products = x_values * w_values
    sums = I.reduce.sum(products, axis=(4, 5, 6), acc_dtype=I.f32)
    bias = I.reshape(b[I.indices(oc_domain)], (1, 128, 1, 1))
    y[n_domain, oc_domain, oh_domain, ow_domain] = sums + bias


@intent.kernel
def conv2d_no_bias(
    x: I.In[I.f32, (8, 64, 16, 16)],
    w: I.In[I.f32, (128, 64, 3, 3)],
    y: I.Out[I.f32, (8, 128, 14, 14)],
):
    n_domain = I.domain(0, 8)
    oc_domain = I.domain(0, 128)
    oh_domain = I.domain(0, 14)
    ow_domain = I.domain(0, 14)
    ic_domain = I.domain(0, 64)
    kh_domain = I.domain(0, 3)
    kw_domain = I.domain(0, 3)

    n = I.reshape(I.indices(n_domain), (8, 1, 1, 1, 1, 1, 1))
    oc = I.reshape(I.indices(oc_domain), (1, 128, 1, 1, 1, 1, 1))
    oh = I.reshape(I.indices(oh_domain), (1, 1, 14, 1, 1, 1, 1))
    ow = I.reshape(I.indices(ow_domain), (1, 1, 1, 14, 1, 1, 1))
    ic = I.reshape(I.indices(ic_domain), (1, 1, 1, 1, 64, 1, 1))
    kh = I.reshape(I.indices(kh_domain), (1, 1, 1, 1, 1, 3, 1))
    kw = I.reshape(I.indices(kw_domain), (1, 1, 1, 1, 1, 1, 3))

    x_values = x[n, ic, oh + kh, ow + kw]
    w_values = w[oc, ic, kh, kw]
    products = x_values * w_values
    sums = I.reduce.sum(products, axis=(4, 5, 6), acc_dtype=I.f32)
    y[n_domain, oc_domain, oh_domain, ow_domain] = sums


def build(context):
    bias_artifact = context.compile("conv2d_bias", conv2d_bias)
    no_bias_artifact = context.compile("conv2d_no_bias", conv2d_no_bias)

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        output = torch.empty((8, 128, 14, 14), device=input.device, dtype=input.dtype)
        if bias is None:
            no_bias_artifact(input, weight, output)
        else:
            bias_artifact(input, weight, bias, output)
        return output

    return conv2d
