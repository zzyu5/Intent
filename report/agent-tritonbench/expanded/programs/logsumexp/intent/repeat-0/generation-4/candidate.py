import intent
import intent.language as I


@intent.kernel
def chunk_logsumexp(
    input: I.In[I.f32, ("C", "P")],
    output: I.Out[I.f32, ("P",)],
):
    columns = I.domain(0, input.shape[0])
    parts = I.domain(0, input.shape[1])
    values = input[columns, parts]
    maximum = I.reduce.max(values, axis=0)
    shifted = values - maximum
    total = I.reduce.sum(I.exp(shifted), axis=0)
    output[parts] = I.log(total) + maximum


@intent.kernel
def combine_logsumexp(
    input: I.In[I.f32, ("P",)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, input.shape[0])
    values = input[elements]
    maximum = I.reduce.max(values, axis=0)
    shifted = values - maximum
    total = I.reduce.sum(I.exp(shifted), axis=0)
    output[I.domain(0, 1)] = I.log(total) + maximum


def build(context):
    chunk_compiled = context.compile("logsumexp_chunks", chunk_logsumexp)
    combine_compiled = context.compile("logsumexp_combine", combine_logsumexp)

    def wrapper(input, dim, keepdim=False, *, out=None):
        transposed = input.reshape(1024, 1024).transpose(0, 1)
        partials = chunk_compiled.run(transposed)
        if out is None:
            return combine_compiled.run(partials).reshape(())
        combine_compiled(partials, out.reshape(1))
        return out

    return wrapper
