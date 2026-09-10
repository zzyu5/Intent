import torch
import intent
import intent.language as I


@intent.fn
def _gelu_exact(value):
    return 0.5 * value * (1.0 + I.erf(value * 0.7071067811865475))


@intent.fn
def _gelu_tanh(value):
    cubic = value * value * value
    return 0.5 * value * (
        1.0 + I.tanh(0.7978845608028654 * (value + 0.044715 * cubic))
    )


@intent.kernel
def _sub_gelu_exact_tensor(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    value = input[elements] - alpha * other[elements]
    output[elements] = _gelu_exact(value)


@intent.kernel
def _sub_gelu_tanh_tensor(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    value = input[elements] - alpha * other[elements]
    output[elements] = _gelu_tanh(value)


@intent.kernel
def _sub_gelu_exact_scalar(
    input: I.In[I.f32, ("N",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    value = input[elements] - alpha * other
    output[elements] = _gelu_exact(value)


@intent.kernel
def _sub_gelu_tanh_scalar(
    input: I.In[I.f32, ("N",)],
    other: I.f32,
    alpha: I.f32,
    output: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input.shape[0])
    value = input[elements] - alpha * other
    output[elements] = _gelu_tanh(value)


def build(context):
    exact_tensor = context.compile("sub_gelu_exact_tensor", _sub_gelu_exact_tensor)
    tanh_tensor = context.compile("sub_gelu_tanh_tensor", _sub_gelu_tanh_tensor)
    exact_scalar = context.compile("sub_gelu_exact_scalar", _sub_gelu_exact_scalar)
    tanh_scalar = context.compile("sub_gelu_tanh_scalar", _sub_gelu_tanh_scalar)

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if approximate == "none":
            tensor_artifact = exact_tensor
            scalar_artifact = exact_scalar
        elif approximate == "tanh":
            tensor_artifact = tanh_tensor
            scalar_artifact = tanh_scalar
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")

        alpha = float(alpha)
        if torch.is_tensor(other):
            artifact = tensor_artifact
            inputs = (input, other, alpha)
        else:
            artifact = scalar_artifact
            inputs = (input, float(other), alpha)

        if out is None:
            return artifact.run(*inputs)
        artifact(*inputs, out)
        return out

    return wrapper
