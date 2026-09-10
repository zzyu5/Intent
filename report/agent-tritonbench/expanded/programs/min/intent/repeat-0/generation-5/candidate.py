import intent
import intent.language as I


@intent.kernel
def min_dim0(
    input: I.In[I.f32, ("M", "N")],
    output_values: I.Out[I.f32, ("N",)],
    output_indices: I.Out[I.i64, ("N",)],
):
    columns = I.domain(0, input.shape[1])

    for column in I.parallel(columns):
        best = input[0, column]
        best_index = I.cast(0, I.i64)

        for row in I.domain(1, input.shape[0]):
            current = input[row, column]

            if current != current:
                if best == best:
                    best = current
                    best_index = I.cast(row, I.i64)
            elif best == best:
                if current < best:
                    best = current
                    best_index = I.cast(row, I.i64)

        output_values[column] = best
        output_indices[column] = best_index


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
