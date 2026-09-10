import torch
import intent
import intent.language as I

def build(context):
    factor_kernel = context.load_source('ldl_factor.py')
    forward_kernel = context.load_source('forward_solve.py')
    backward_kernel = context.load_source('backward_solve.py')

    def wrapper(A, b):
        factor = torch.empty_like(A)
        factor_kernel.run(A, factor)
        intermediate = torch.empty_like(b)
        forward_kernel.run(factor, b, intermediate)
        output = torch.empty_like(b)
        backward_kernel.run(factor, intermediate, output)
        return output
    return wrapper
