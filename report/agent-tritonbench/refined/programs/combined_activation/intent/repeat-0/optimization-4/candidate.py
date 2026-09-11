import intent
import intent.language as I

def build(context):
    compiled = context.load_source('combined_activation.py')

    def wrapper(input, weight1, weight2, bias, *, out=None):
        if out is None:
            return compiled.run(input, weight1, weight2, bias)
        compiled(input, weight1, weight2, bias, out)
        return out
    return wrapper
