import intent
import intent.language as I


@intent.kernel
def exp_mean_partials(
    input: I.In[I.f32, (128, 8192)],
    partials: I.Out[I.f32, (128,)],
):
    rows = I.domain(0, input.shape[0])
    columns = I.domain(0, input.shape[1])
    exponentials = I.exp2(input[rows, columns] * 1.4426950408889634)
    partials[rows] = I.reduce.sum(exponentials, axis=1)


@intent.kernel
def exp_mean_finish(
    partials: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, partials.shape[0])
    total = I.reduce.sum(partials[elements], axis=0)
    output[0] = I.fdiv(total, 1048576.0)


def build(context):
    partials_kernel = context.compile("exp_mean_partials", exp_mean_partials)
    finish_kernel = context.compile("exp_mean_finish", exp_mean_finish)

    def wrapper(input, dim=None, keepdim=False, dtype=None, out=None):
        reshaped = input.reshape((128, 8192))
        partials = partials_kernel.run(reshaped)
        if out is None:
            return finish_kernel.run(partials).reshape(())
        finish_kernel(partials, out.reshape((1,)))
        return out

    return wrapper
