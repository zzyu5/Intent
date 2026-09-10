import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('fused_index_select_eq.py')

    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            return compiled.run(input, index, other).view(torch.bool)
        compiled(input, index, other, out.view(torch.int8))
        return out
    return wrapper
