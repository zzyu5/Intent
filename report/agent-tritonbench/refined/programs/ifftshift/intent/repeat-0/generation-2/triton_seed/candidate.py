import intent
import intent.language as I

def build(context):
    compiled = context.load_source('ifftshift_1d.py')

    def wrapper(input, dim=None):
        return compiled.run(input)
    return wrapper
