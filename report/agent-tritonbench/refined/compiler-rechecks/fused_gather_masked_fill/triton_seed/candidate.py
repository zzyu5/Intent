import intent
import intent.language as I

def build(context):
    compiled = context.load_source('fused_gather_masked_fill_kernel.py')

    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        if out is None:
            return compiled.run(input, index, mask, value)
        compiled(input, index, mask, value, out)
        return out
    return wrapper
