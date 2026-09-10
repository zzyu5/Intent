import math
import intent
import intent.language as I

def build(context):
    compiled = context.load_source('min_rows.py')

    def wrapper(input, dim, keepdim=False, *, out=None):
        if dim != 0:
            raise ValueError('this specialization reduces dimension 0')
        if out is None:
            (values, indices) = compiled.run(input)
        else:
            (values, indices) = out
            if keepdim:
                compiled(input, values.squeeze(0), indices.squeeze(0))
            else:
                compiled(input, values, indices)
        if keepdim:
            if out is None:
                values = values.unsqueeze(0)
                indices = indices.unsqueeze(0)
        return (values, indices)
    return wrapper
