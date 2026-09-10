import intent
import intent.language as I

def build(context):
    compiled = context.load_source('grid_sample_bilinear_zeros.py')

    def wrapper(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
        return compiled.run(input, grid)
    return wrapper
