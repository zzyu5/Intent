import intent
import intent.language as I

def build(context):
    compiled = context.load_source('leaky_relu.py')

    def wrapper(input, negative_slope=0.01, inplace=False):
        if inplace:
            compiled(input, negative_slope, input)
            return input
        return compiled.run(input, negative_slope)
    return wrapper
