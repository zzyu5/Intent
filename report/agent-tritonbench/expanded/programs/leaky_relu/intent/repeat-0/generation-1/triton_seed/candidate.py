import intent
import intent.language as I

def build(context):
    compiled_out = context.load_source('leaky_relu_out.py')
    compiled_inplace = context.load_source('leaky_relu_inplace.py')

    def wrapper(input, negative_slope=0.01, inplace=False):
        if inplace:
            compiled_inplace.run(input, negative_slope)
            return input
        return compiled_out.run(input, negative_slope)
    return wrapper
