import torch
import intent
import intent.language as I


@intent.kernel
def _asin_kernel(
    input_tensor: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    n = input_tensor.shape[0]
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    half_pi = I.cast(1.5707288, I.f32)
    c0 = I.cast(1.5707288, I.f32)
    c1 = I.cast(-0.2121144, I.f32)
    c2 = I.cast(0.0742610, I.f32)
    c3 = I.cast(-0.0187293, I.f32)

    for i in I.parallel(I.domain(0, n)):
        x = I.cast(input_tensor[i], I.f32)
        ax = I.abs(x)

        # This minimax-style acos approximation remains accurate near both
        # zero and the endpoints; sqrt also gives NaN outside [-1, 1].
        p = c3 * ax + c2
        p = p * ax + c1
        p = p * ax + c0
        magnitude = half_pi - I.sqrt(one - ax) * p
        value = I.select(x < zero, -magnitude, magnitude)
        output[i] = I.cast(value, I.f16)


def build(context):
    kernel = context.compile("asin_pointwise", _asin_kernel)

    def asin(input_tensor):
        output = torch.empty_like(input_tensor)
        kernel(input_tensor, output)
        return output

    return asin
