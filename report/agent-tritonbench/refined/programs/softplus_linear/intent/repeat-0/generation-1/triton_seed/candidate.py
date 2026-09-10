import intent
import intent.language as I

def build(context):
    with_bias = context.load_source('softplus_linear_with_bias.py')
    without_bias = context.load_source('softplus_linear_without_bias.py')

    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        if bias is None:
            return without_bias.run(input, weight)
        return with_bias.run(input, weight, bias)
    return wrapper
