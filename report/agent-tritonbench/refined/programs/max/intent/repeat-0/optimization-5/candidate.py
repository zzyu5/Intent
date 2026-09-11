from collections import namedtuple
import intent
import intent.language as I
MaxResult = namedtuple('max', ('values', 'indices'))

def build(context):
    compiled = context.load_source('max_reduce.py')

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            (values, indices) = compiled.run(input)
        else:
            (values, indices) = out
            compiled(input, values.reshape(1), indices.reshape(1))
        if not keepdim:
            values = values.reshape(())
            indices = indices.reshape(())
        return MaxResult(values, indices)
    return wrapper
