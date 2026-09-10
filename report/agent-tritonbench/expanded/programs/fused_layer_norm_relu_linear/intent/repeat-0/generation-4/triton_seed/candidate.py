import intent
import intent.language as I

def build(context):
    linear = context.load_source('fused_linear_generation.py')
    relu_bias = context.load_source('fused_relu_bias_generation.py')
    layer_norm = context.load_source('fused_layer_norm_generation.py')

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-05, elementwise_affine=True):
        linear_output = linear.run(input, weight)
        activated = relu_bias.run(linear_output, bias)
        return layer_norm.run(activated, eps)
    return wrapper
