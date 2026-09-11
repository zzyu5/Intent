import torch

def build(context):
    fused = context.load_source("fused_optimized.py")

    def wrapper(input, weight, bias=None, normalized_shape=None, eps=1e-05, elementwise_affine=True):
        return fused.run(input, weight, bias, eps)

    return wrapper
