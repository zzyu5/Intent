import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('solve_system_lu.py')

    def wrapper(A, B, *, left=True, out=None):
        if left:
            result = out if out is not None else torch.empty_like(B)
            work = torch.empty((A.shape[-1] * (A.shape[-1] + 1),), device=A.device, dtype=A.dtype)
            compiled(A, B, work, result)
            return result
        a_view = A.transpose(-2, -1)
        b_view = B.transpose(-2, -1)
        result = out if out is not None else torch.empty_like(B)
        result_view = result.transpose(-2, -1)
        work = torch.empty((a_view.shape[-1] * (a_view.shape[-1] + 1),), device=A.device, dtype=A.dtype)
        compiled(a_view, b_view, work, result_view)
        return result
    return wrapper
