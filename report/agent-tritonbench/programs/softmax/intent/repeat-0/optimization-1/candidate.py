import intent
import intent.language as I

def build(context):
    compiled = context.load_source('softmax_kernel.py')

    def wrapper(input, dim, dtype=None):
        return compiled.run(input)
    return wrapper
