import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('rsqrt_generation_3.py')

    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty(input.shape, device=input.device, dtype=input.dtype)
        compiled(input, out)
        return out
    return wrapper
