import intent
import intent.language as I

def build(context):
    compiled = context.load_source('add_gelu.py')

    def wrapper(input, other, alpha=1, approximate='none', out=None):
        if out is None:
            return compiled.run(input, other, alpha)
        compiled(input, other, alpha, out)
        return out
    return wrapper
