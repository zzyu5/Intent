import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mul_relu.py')

    def wrapper(input, other, inplace=False, out=None):
        if out is not None:
            compiled(input, other, out)
            return out
        if inplace:
            compiled(input, other, input)
            return input
        return compiled.run(input, other)
    return wrapper
