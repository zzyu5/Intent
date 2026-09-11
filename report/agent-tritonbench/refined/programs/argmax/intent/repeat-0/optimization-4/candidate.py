import intent
import intent.language as I

def build(context):
    compiled = context.load_source('argmax.py')

    def wrapper(input, dim, keepdim=False):
        return compiled.run(input)
    return wrapper
