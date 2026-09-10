import intent
import intent.language as I

def build(context):
    compiled = context.load_source('sum.py')

    def wrapper(input, dim, keepdim=False, *, dtype=None):
        return compiled.run(input)
    return wrapper
