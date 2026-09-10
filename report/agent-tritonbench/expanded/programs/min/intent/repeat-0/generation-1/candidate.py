import intent
import intent.language as I


@intent.kernel
def min_dim0(
    input: I.In[I.f32, ("M", "N")],
    output_values: I.Out[I.f32, ("N",)],
    output_indices: I.Out[I.i64, ("N",)],
):
    columns = I.domain(0, input.shape[1])
    reduced_values, reduced_indices = I.arg_reduce.max(-input, axis=0)
    output_values[columns] = -reduced_values
    output_indices[columns] = reduced_indices


def build(context):
    compiled = context.compile("min_dim0", min_dim0)

    def wrapper(input, dim, keepdim=False, *, out=None):
        if out is None:
            values, indices = compiled.run(input)
            if keepdim:
                values = values.reshape(1, values.shape[0])
                indices = indices.reshape(1, indices.shape[0])
            return values, indices

        if keepdim:
            values_target = out[0].select(0, 0)
            indices_target = out[1].select(0, 0)
        else:
            values_target = out[0]
            indices_target = out[1]
        compiled(input, values_target, indices_target)
        return out

    return wrapper
