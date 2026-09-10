import intent
import intent.language as I

def build(context):
    compiled = context.load_source('fused_mv_sigmoid_sub.py')

    def wrapper(input, vec, other, alpha=1, *, out=None):
        if out is None:
            return compiled.run(input, vec, other, alpha)
        compiled(input, vec, other, alpha, out)
        return out
    return wrapper
