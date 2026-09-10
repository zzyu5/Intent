import intent
import intent.language as I

def build(context):
    compiled = context.load_source('min_dim0.py')

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            (values, indices) = compiled.run(input)
            if keepdim:
                values = values.reshape(1, values.shape[0])
                indices = indices.reshape(1, indices.shape[0])
            return (values, indices)
        if keepdim:
            values_target = out[0].select(0, 0)
            indices_target = out[1].select(0, 0)
        else:
            values_target = out[0]
            indices_target = out[1]
        compiled(input, values_target, indices_target)
        return out
    return wrapper
