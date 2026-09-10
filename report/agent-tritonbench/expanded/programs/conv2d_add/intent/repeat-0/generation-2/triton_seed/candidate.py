import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('conv2d_add_fixed.py')

    def wrapper(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
        if out is None:
            return compiled.run(input, weight, other, alpha)
        compiled(input, weight, other, out, alpha)
        return out
    return wrapper
