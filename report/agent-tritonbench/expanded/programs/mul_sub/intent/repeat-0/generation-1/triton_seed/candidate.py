import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mul_sub.py')

    def wrapper(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            return compiled.run(input, other_mul, other_sub, alpha)
        compiled(input, other_mul, other_sub, alpha, out)
        return out
    return wrapper
