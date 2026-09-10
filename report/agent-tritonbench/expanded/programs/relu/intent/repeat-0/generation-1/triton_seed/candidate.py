import intent
import intent.language as I

def build(context):
    compiled = context.load_source('relu.py')

    def wrapper(input, inplace=False):
        if inplace:
            compiled(input, input)
            return input
        return compiled.run(input)
    return wrapper
