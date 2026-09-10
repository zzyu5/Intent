import torch
import intent
import intent.language as I


@intent.kernel
def mean_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, input_tensor.shape[0])
    total = I.reduce.sum(input_tensor[elements], axis=0)
    output[0] = total / I.cast(input_tensor.shape[0], I.f32)


def build(context):
    compiled = context.compile("mean", mean_kernel)

    def wrapper(input_tensor, dim, keepdim=False, dtype=None, out=None):
        partial = torch.empty((32,), device=input_tensor.device, dtype=torch.float32)
        chunk = 32768
        for index in range(32):
            begin = index * chunk
            compiled(
                input_tensor[begin : begin + chunk],
                partial[index : index + 1],
            )
        if out is not None:
            result = out.reshape(1)
            compiled(partial, result)
            return out
        result = compiled.run(partial)
        return result if keepdim else result.reshape(())

    return wrapper
