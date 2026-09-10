import intent
import intent.language as I
import torch

def build(context):
    compiled = context.load_source('conv2d_add.py')

    def wrapper(input, weight, bias=None, other=None, stride=1, padding=0, dilation=1, groups=1, alpha=1, out=None):
        if bias is not None:
            raise NotImplementedError('fixed profile has no bias')
        if other is None:
            raise NotImplementedError('fixed profile supplies other')
        if stride != 1 or padding != 1 or dilation != 1 or (groups != 1) or (alpha != 1):
            raise NotImplementedError('fixed profile uses unit stride, dilation, and alpha')
        result = out if out is not None else torch.empty_like(other)
        compiled(input, weight, other, result)
        return result
    return wrapper
