import torch
import intent
import intent.language as I


@intent.kernel
def _batch_norm_inference(
    input: I.In[I.f16, (32, 32, 640, 128)],
    running_mean: I.In[I.f32, (32,)],
    running_var: I.In[I.f32, (32,)],
    weight: I.In[I.f32, (32,)],
    bias: I.In[I.f32, (32,)],
    output: I.Out[I.f16, (32, 32, 640, 128)],
    eps: I.f32,
):
    batch = I.domain(0, 32)
    channel = I.domain(0, 32)
    height = I.domain(0, 640)
    width = I.domain(0, 128)

    # Expand channel parameters over the three non-channel axes.  The input
    # is promoted to f32 so the affine arithmetic is not performed in f16.
    mean = I.reshape(running_mean[channel], (1, 32, 1, 1))
    variance = I.reshape(running_var[channel], (1, 32, 1, 1))
    scale = I.reshape(weight[channel], (1, 32, 1, 1))
    shift = I.reshape(bias[channel], (1, 32, 1, 1))
    value = I.cast(input[batch, channel, height, width], I.f32)
    normalized = (value - mean) / I.sqrt(variance + eps)
    output[batch, channel, height, width] = I.cast(normalized * scale + shift, I.f16)


def build(context):
    kernel = context.compile("batch_norm_inference", _batch_norm_inference)

    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        output = torch.empty_like(input)
        kernel(input, running_mean, running_var, weight, bias, output, eps)
        return output

    return batch_norm
