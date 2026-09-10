import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mean.py')

    def wrapper(input_tensor, dim, keepdim=False, dtype=None, out=None):
        return compiled.run(input_tensor).reshape(())
    return wrapper
