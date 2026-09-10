import intent
import intent.language as I

def build(context):
    chunk_compiled = context.load_source('logsumexp_chunks.py')
    combine_compiled = context.load_source('logsumexp_combine.py')

    def wrapper(input, dim, keepdim=False, *, out=None):
        transposed = input.reshape(1024, 1024).transpose(0, 1)
        partials = chunk_compiled.run(transposed)
        if out is None:
            return combine_compiled.run(partials).reshape(())
        combine_compiled(partials, out.reshape(1))
        return out
    return wrapper
