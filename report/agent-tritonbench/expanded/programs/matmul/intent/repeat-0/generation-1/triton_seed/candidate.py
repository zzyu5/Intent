import intent
import intent.language as I

def build(context):
    compiled = context.load_source('matmul.py')

    def wrapper(input, other, *, out=None):
        if out is None:
            return compiled.run(input, other)
        compiled(input, other, out)
        return out
    return wrapper
