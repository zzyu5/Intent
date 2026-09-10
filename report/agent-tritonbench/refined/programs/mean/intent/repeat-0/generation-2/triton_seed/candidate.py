import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mean.py')

    def wrapper(input_tensor, dim, keepdim=False, dtype=None, out=None):
        if out is None:
            return compiled.run(input_tensor)
        compiled(input_tensor, out)
        return out
    return wrapper
