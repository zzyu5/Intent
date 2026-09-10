import intent
import intent.language as I

def build(context):
    compiled = context.load_source('permute_copy.py')

    def wrapper(input, dims):
        return compiled.run(input)
    return wrapper
