import intent
import intent.language as I

def build(context):
    compiled = context.load_source('exp_mean.py')

    def wrapper(input, dim=None, keepdim=False, dtype=None, out=None):
        return compiled.run(input)
    return wrapper
