import intent
import intent.language as I


@intent.kernel
def exp_mean_partials(
    input: I.In[I.f32, (8192, 128)],
    partials: I.Out[I.f32, (8192,)],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    exponentials = I.exp2(input[rows, columns] * 1.4426950408889634)
    partials[rows] = I.reduce.sum(exponentials, axis=1)


@intent.kernel
def exp_mean_finish(
    partials: I.In[I.f32, (128, 64)],
    output: I.Out[I.f32, (1,)],
):
    rows = I.domain(0, partials.shape[0])
    columns = I.domain(0, partials.shape[1])
    total = I.reduce.sum(partials[rows, columns], axis=(0, 1))
    output[0] = I.fdiv(total, 1048576.0)


def build(context):
    partials_kernel = context.compile("exp_mean_partials", exp_mean_partials)
    finish_kernel = context.compile("exp_mean_finish", exp_mean_finish)

    def wrapper(input, dim=None, keepdim=False, dtype=None, out=None):
        reshaped = input.reshape((8192, 128))
        partials = partials_kernel.run(reshaped)
        if out is None:
            reduced = finish_kernel.run(partials.reshape((128, 64)))
            return reduced.reshape(())
        finish_kernel(partials.reshape((128, 64)), out.reshape((1,)))
        return out

    return wrapper
