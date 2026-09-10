import intent
import intent.language as I

def build(context):
    with_bias = context.load_source('softplus_linear_bias.py')
    without_bias = context.load_source('softplus_linear_no_bias.py')

    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        beta = float(beta)
        threshold = float(threshold)
        if bias is None:
            return without_bias.run(input, weight, beta, threshold)
        return with_bias.run(input, weight, bias, beta, threshold)
    return wrapper
