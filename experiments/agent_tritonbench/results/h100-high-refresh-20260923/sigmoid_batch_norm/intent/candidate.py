import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_batch_norm(
    input: I.In[I.f32, (32, 32768)],
    running_mean: I.In[I.f32, (32768,)],
    running_var: I.In[I.f32, (32768,)],
    weight: I.In[I.f32, (32768,)],
    bias: I.In[I.f32, (32768,)],
    output: I.Out[I.f32, (32, 32768)],
    eps: I.f32,
    USE_WEIGHT: I.Constexpr[bool],
    USE_BIAS: I.Constexpr[bool],
):
    channels = I.domain(0, 32768)
    for row in I.parallel(I.domain(0, 32)):
        x = input[row, channels]
        mean = running_mean[channels]
        variance = running_var[channels]
        normalized = (x - mean) * I.rsqrt(variance + eps)
        if USE_WEIGHT:
            normalized = normalized * weight[channels]
        if USE_BIAS:
            normalized = normalized + bias[channels]
        output[row, channels] = I.sigmoid(normalized)


def build(context):
    artifacts = {}
    for use_weight in (False, True):
        for use_bias in (False, True):
            name = f"sigmoid_batch_norm_{int(use_weight)}_{int(use_bias)}"
            artifacts[(use_weight, use_bias)] = context.compile(
                name,
                _sigmoid_batch_norm,
                constexprs={"USE_WEIGHT": use_weight, "USE_BIAS": use_bias},
            )

    def sigmoid_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        del training, momentum
        use_weight = weight is not None
        use_bias = bias is not None
        output = torch.empty_like(input)
        weight_arg = weight if use_weight else running_mean
        bias_arg = bias if use_bias else running_mean
        artifacts[(use_weight, use_bias)](
            input,
            running_mean,
            running_var,
            weight_arg,
            bias_arg,
            output,
            eps,
        )
        return output

    return sigmoid_batch_norm
