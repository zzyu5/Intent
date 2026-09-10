import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('grid_sample_bilinear_zeros_false.py')

    def wrapper(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False):
        output = torch.empty((input.shape[0], input.shape[1], grid.shape[1], grid.shape[2]), dtype=input.dtype, device=input.device)
        compiled(input, grid, output)
        return output
    return wrapper
