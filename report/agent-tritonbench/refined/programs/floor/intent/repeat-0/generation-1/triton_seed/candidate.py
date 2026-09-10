import intent
import intent.language as I

def build(context):
    compiled = context.load_source('floor.py')

    def wrapper(input, *, out=None):
        if out is None:
            return compiled.run(input)
        compiled(input, out)
        return out
    return wrapper
