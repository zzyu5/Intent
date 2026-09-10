from collections import namedtuple
import intent
import intent.language as I
_MaxResult = namedtuple('max', ('values', 'indices'))

def build(context):
    compiled = context.load_source('max_1d.py')

    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim not in (0, -1):
            raise ValueError('dim must be 0 for a one-dimensional input')
        if out is None:
            (values, indices) = compiled.run(input)
            if keepdim:
                values = values.reshape((1,))
                indices = indices.reshape((1,))
            return _MaxResult(values, indices)
        (values_out, indices_out) = out
        values_arg = values_out.reshape(()) if keepdim else values_out
        indices_arg = indices_out.reshape(()) if keepdim else indices_out
        compiled(input, values_arg, indices_arg)
        return _MaxResult(values_out, indices_out)
    return wrapper
