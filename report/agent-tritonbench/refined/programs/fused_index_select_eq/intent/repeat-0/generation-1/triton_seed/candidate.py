import intent
import intent.language as I

def build(context):
    compiled = context.load_source('fused_index_select_eq.py')

    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            return compiled.run(input, index, other)
        compiled(input, index, other, out)
        return out
    return wrapper
