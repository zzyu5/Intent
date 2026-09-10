import intent
import intent.language as I

def build(context):
    compiled = context.load_source('relu_conv2d.py')

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
        return compiled.run(input, weight)
    return wrapper
