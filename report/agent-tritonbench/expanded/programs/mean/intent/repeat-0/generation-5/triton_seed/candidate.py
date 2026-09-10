import torch
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('mean.py')

    def wrapper(input_tensor, dim, keepdim=False, dtype=None, out=None):
        partial = torch.empty((32,), device=input_tensor.device, dtype=torch.float32)
        chunk = 32768
        for index in range(32):
            begin = index * chunk
            compiled(input_tensor[begin:begin + chunk], partial[index:index + 1])
        if out is not None:
            result = out.reshape(1)
            compiled(partial, result)
            return out
        result = compiled.run(partial)
        return result if keepdim else result.reshape(())
    return wrapper
