def build(context):
    fused = context.load_source('softplus_linear_fused.py')

    def wrapper(input, weight, bias=None, beta=1, threshold=20):
        return fused.run(input, weight, bias, beta, threshold)

    return wrapper
