import intent
import intent.language as I


@intent.fn
def _sigmoid(value):
    return I.fdiv(1.0, 1.0 + I.exp2(-value * 1.4426950408889634))


@intent.kernel
def combined_activation(
    input: I.In[I.f32, ("M", "K")],
    weight1: I.In[I.f32, ("K", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    activation = _sigmoid(I.matmul(input, weight1, acc_dtype=I.f32))
    activation = I.tanh(activation)

    rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])
    output[rows, columns] = (
        activation[rows, columns] * weight2[columns] + bias[columns]
    )


def build(context):
    compiled = context.compile("combined_activation", combined_activation)

    def wrapper(input, weight1, weight2, bias, *, out=None):
        if out is None:
            return compiled.run(input, weight1, weight2, bias)
        compiled(input, weight1, weight2, bias, out)
        return out

    return wrapper
