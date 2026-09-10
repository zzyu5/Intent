import intent
import intent.language as I

def build(context):
    compiled = context.load_source('add_mean.py')

    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        return compiled.run(input, other)
    return wrapper
