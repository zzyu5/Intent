import intent
import intent.language as I

def build(context):
    compiled_out = context.load_source('relu_out.py')
    compiled_inplace = context.load_source('relu_inplace.py')

    def wrapper(input, inplace=False):
        if inplace:
            compiled_inplace.run(input)
            return input
        return compiled_out.run(input)
    return wrapper
