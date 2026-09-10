import torch
import triton
import triton.language as tl


@triton.jit
def _permute_copy_storage(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, dims):
        output_shape = tuple(input.shape[dim] for dim in dims)
        output_strides = tuple(input.stride()[dim] for dim in dims)
        output = torch.empty_strided(
            output_shape,
            output_strides,
            dtype=input.dtype,
            device=input.device,
        )
        n_elements = input.numel()
        _permute_copy_storage[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
