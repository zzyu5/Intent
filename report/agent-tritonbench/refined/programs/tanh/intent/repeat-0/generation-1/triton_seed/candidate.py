import intent
import intent.language as I
import torch

def build(context):
    compiled = context.load_source('tanh_kernel.py')

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out
    return wrapper
