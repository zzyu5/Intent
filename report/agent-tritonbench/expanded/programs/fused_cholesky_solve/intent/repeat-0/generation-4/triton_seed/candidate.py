import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('fused_cholesky_solve.py')

    def wrapper(A, b):
        L = torch.empty_like(A)
        y = torch.empty_like(b)
        x = torch.empty_like(b)
        compiled(A, b, L, y, x)
        return x
    return wrapper
