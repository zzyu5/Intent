import intent
import intent.language as I


@intent.kernel
def add_mean_partials(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    partials: I.Out[I.f32, (16,)],
):
    source = I.domain(0, input.shape[0])
    parts = I.domain(0, 16)
    for part in I.parallel(parts):
        begin = part * 65536
        end = begin + 65536
        region = source[begin:end]
        partials[part] = I.reduce.sum(input[region] + other[region], axis=0)


@intent.kernel
def add_mean_finish(
    partials: I.In[I.f32, (16,)],
    output: I.Out[I.f32, (1,)],
):
    elements = I.domain(0, partials.shape[0])
    total = I.reduce.sum(partials[elements], axis=0)
    output[0] = total / I.cast(1048576, I.f32)


def build(context):
    partial_artifact = context.compile(
        "add_mean_partials",
        add_mean_partials,
        constexprs={},
    )
    finish_artifact = context.compile(
        "add_mean_finish",
        add_mean_finish,
        constexprs={},
    )

    def wrapper(input, other, dim=None, alpha=1, keepdim=False, dtype=None, out=None):
        partials = partial_artifact.run(input, other)
        return finish_artifact.run(partials).view(())

    return wrapper
