import intent
import intent.language as I

def build(context):
    compiled = context.load_source('conv2d_fixed_nchw.py')

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        return compiled.run(input, weight, bias)
    return wrapper
