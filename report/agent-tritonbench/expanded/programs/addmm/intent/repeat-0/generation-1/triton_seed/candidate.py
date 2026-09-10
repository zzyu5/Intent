import intent
import intent.language as I

def build(context):
    compiled = context.load_source('addmm_kernel.py')

    def wrapper(input, mat1, mat2, *, beta=1, alpha=1, out=None):
        if out is None:
            return compiled.run(input, mat1, mat2)
        compiled(input, mat1, mat2, out)
        return out
    return wrapper
