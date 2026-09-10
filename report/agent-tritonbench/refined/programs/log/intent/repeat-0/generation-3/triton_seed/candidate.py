import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('log_candidate_3.py')

    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)
        compiled.run(input, out)
        return out
    return wrapper
