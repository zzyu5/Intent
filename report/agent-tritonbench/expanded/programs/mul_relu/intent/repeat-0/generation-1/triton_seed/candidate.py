import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mul_relu.py')

    def wrapper(input, other, inplace=False, out=None):
        if out is not None:
            output = out
        elif inplace:
            output = input
        else:
            output = torch.empty_like(input)
        compiled(input, other, output)
        return output
    return wrapper
