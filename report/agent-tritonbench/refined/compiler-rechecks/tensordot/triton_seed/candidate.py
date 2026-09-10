import intent
import intent.language as I

def build(context):
    compiled = context.load_source('tensordot.py')

    def wrapper(a, b, dims):
        return compiled.run(a, b)
    return wrapper
