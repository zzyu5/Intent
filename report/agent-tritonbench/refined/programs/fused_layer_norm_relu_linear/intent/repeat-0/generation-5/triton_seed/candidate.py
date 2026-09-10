import intent
import intent.language as I

def build(context):
    linear_bias_compiled = context.load_source('fused_linear_relu_bias.py')
    linear_no_bias_compiled = context.load_source('fused_linear_relu_no_bias.py')
    norm_compiled = context.load_source('fused_layer_norm_rows.py')

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-05, elementwise_affine=True):
        if bias is None:
            activated = linear_no_bias_compiled.run(input, weight)
        else:
            activated = linear_bias_compiled.run(input, weight, bias)
        return norm_compiled.run(activated, eps)
    return wrapper
