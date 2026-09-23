import torch
import intent
import intent.language as I


@intent.kernel
def _fused_eval(
    x: I.In[I.f32, (1, 64, 128, 128)],
    out: I.Out[I.f32, (1, 64, 128, 128)],
    running_mean: I.In[I.f32, (64,)],
    running_var: I.In[I.f32, (64,)],
    weight: I.In[I.f32, (64,)],
    bias: I.In[I.f32, (64,)],
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
):
    rows = I.domain(0, 1)
    channels = I.domain(0, 64)
    height = I.domain(0, 128)
    width = I.domain(0, 128)

    for n in I.parallel(rows):
        for c in I.parallel(channels):
            mean = running_mean[c]
            inv_std = I.fdiv(I.cast(1.0, I.f32), I.sqrt(running_var[c] + eps))
            for h in I.parallel(height):
                for w in I.parallel(width):
                    normalized = (x[n, c, h, w] - mean) * inv_std
                    affine = normalized
                    if HAS_WEIGHT:
                        affine = affine * weight[c]
                    if HAS_BIAS:
                        affine = affine + bias[c]
                    clipped = I.minimum(I.maximum(affine + I.cast(3.0, I.f32), I.cast(0.0, I.f32)), I.cast(6.0, I.f32))
                    out[n, c, h, w] = I.fdiv(clipped, I.cast(6.0, I.f32))


def build(context):
    kernels = {}
    for has_weight in (False, True):
        for has_bias in (False, True):
            key = (has_weight, has_bias)
            kernels[key] = context.compile(
                "fused_hardsigmoid_batch_norm_%d_%d" % (has_weight, has_bias),
                _fused_eval,
                constexprs={
                    "HAS_WEIGHT": has_weight,
                    "HAS_BIAS": has_bias,
                },
            )

    def fused_hardsigmoid_batch_norm(
        x,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
        inplace=False,
    ):
        del training, momentum
        has_weight = weight is not None
        has_bias = bias is not None
        weight_arg = weight if has_weight else running_mean
        bias_arg = bias if has_bias else running_mean
        out = x if inplace else torch.empty_like(x)
        kernels[(has_weight, has_bias)](
            x,
            out,
            running_mean,
            running_var,
            weight_arg,
            bias_arg,
            eps,
        )
        return out

    return fused_hardsigmoid_batch_norm
