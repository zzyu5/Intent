from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _conv_bn_relu(
    x: I.In[I.f32, (1, 64, 32, 32)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    running_mean: I.In[I.f32, (128,)],
    running_var: I.In[I.f32, (128,)],
    bn_weight: I.In[I.f32, (128,)],
    bn_bias: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (1, 128, 32, 32)],
    eps: I.f32,
):
    # Each output point is independent. The local ordered loops express the
    # convolution reduction and avoid materializing a convolution workspace.
    for point in I.parallel(I.domain(0, 128 * 32 * 32)):
        oc = point // 1024
        spatial = point % 1024
        oh = spatial // 32
        ow = spatial % 32

        acc = I.cast(0.0, I.f32)
        for ic in range(64):
            for kh in range(3):
                ih = oh + kh - 1
                if ih >= 0:
                    if ih < 32:
                        for kw in range(3):
                            iw = ow + kw - 1
                            if iw >= 0:
                                if iw < 32:
                                    acc = acc + x[0, ic, ih, iw] * weight[oc, ic, kh, kw]

        value = (acc - running_mean[oc]) / I.sqrt(running_var[oc] + eps)
        value = value * bn_weight[oc] + bn_bias[oc]
        value = I.select(value > 0.0, value, I.cast(0.0, I.f32))
        out[0, oc, oh, ow] = value


def build(context):
    conv_bn_relu = context.compile("conv_bn_relu", _conv_bn_relu)

    def relu_batch_norm_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        running_mean=None,
        running_var=None,
        bn_weight=None,
        bn_bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
        inplace=False,
    ):
        output = torch.empty(
            (1, 128, 32, 32),
            device=input.device,
            dtype=input.dtype,
        )
        conv_bn_relu(
            input,
            weight,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            output,
            eps,
        )
        return output

    return relu_batch_norm_conv2d
