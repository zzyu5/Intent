import torch
import intent
import intent.language as I

def build(context):
    exact_tensor = context.load_source('sub_gelu_exact_tensor.py')
    tanh_tensor = context.load_source('sub_gelu_tanh_tensor.py')
    exact_scalar = context.load_source('sub_gelu_exact_scalar.py')
    tanh_scalar = context.load_source('sub_gelu_tanh_scalar.py')

    def wrapper(input, other, alpha=1, approximate='none', out=None):
        if approximate == 'none':
            tensor_artifact = exact_tensor
            scalar_artifact = exact_scalar
        elif approximate == 'tanh':
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
