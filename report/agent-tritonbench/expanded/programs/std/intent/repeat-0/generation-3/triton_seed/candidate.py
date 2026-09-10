import torch
import intent
import intent.language as I

def build(context):
    chunk_sum = context.load_source('std_chunk_sum.py')
    mean_kernel = context.load_source('std_mean.py')
    chunk_variance = context.load_source('std_chunk_variance.py')
    finalize = context.load_source('std_finalize.py')

    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        chunk_size = 65536
        count = input.shape[0]
        chunks = (count + chunk_size - 1) // chunk_size
        partials = torch.empty((chunks,), device=input.device, dtype=torch.float32)
        for chunk in range(chunks):
            start = chunk * chunk_size
            length = min(chunk_size, count - start)
            chunk_sum(input.narrow(0, start, length), partials.narrow(0, chunk, 1))
        mean = torch.empty((1,), device=input.device, dtype=torch.float32)
        mean_kernel(partials, input, mean)
        variance_partials = torch.empty((chunks,), device=input.device, dtype=torch.float32)
        for chunk in range(chunks):
            start = chunk * chunk_size
            length = min(chunk_size, count - start)
            chunk_variance(input.narrow(0, start, length), mean, variance_partials.narrow(0, chunk, 1))
        if out is None:
            result = torch.empty((1,), device=input.device, dtype=torch.float32)
            finalize(variance_partials, input, result)
            return result.reshape(())
        finalize(variance_partials, input, out.reshape(1))
        return out
    return wrapper
