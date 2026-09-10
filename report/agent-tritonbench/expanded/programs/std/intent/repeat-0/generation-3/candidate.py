import torch
import intent
import intent.language as I


@intent.kernel
def std_chunk_sum(
    x: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, x.shape[0])
    total = I.reduce.sum(x[elements], axis=0)
    output[0] = total


@intent.kernel
def std_mean(
    partials: I.In[I.f32, ("P",)],
    count_source: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, partials.shape[0])
    total = I.reduce.sum(partials[elements], axis=0)
    count = I.cast(count_source.shape[0], I.f32)
    output[0] = I.fdiv(total, count)


@intent.kernel
def std_chunk_variance(
    x: I.In[I.f32, ("N",)],
    mean: I.In[I.f32, (1,)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, x.shape[0])
    centered = x[elements] - mean[0]
    squared = centered * centered
    output[0] = I.reduce.sum(squared, axis=0)


@intent.kernel
def std_finalize(
    partials: I.In[I.f32, ("P",)],
    count_source: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, partials.shape[0])
    sum_squared = I.reduce.sum(partials[elements], axis=0)
    count = I.cast(count_source.shape[0], I.f32)
    variance = I.fdiv(sum_squared, count - 1.0)
    output[0] = I.sqrt(variance)


def build(context):
    chunk_sum = context.compile("std_chunk_sum", std_chunk_sum)
    mean_kernel = context.compile("std_mean", std_mean)
    chunk_variance = context.compile("std_chunk_variance", std_chunk_variance)
    finalize = context.compile("std_finalize", std_finalize)

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
            chunk_variance(
                input.narrow(0, start, length),
                mean,
                variance_partials.narrow(0, chunk, 1),
            )

        if out is None:
            result = torch.empty((1,), device=input.device, dtype=torch.float32)
            finalize(variance_partials, input, result)
            return result.reshape(())

        finalize(variance_partials, input, out.reshape(1))
        return out

    return wrapper
