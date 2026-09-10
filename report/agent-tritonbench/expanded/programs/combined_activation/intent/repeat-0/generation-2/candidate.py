import intent
import intent.language as I


@intent.kernel
def combined_activation(
    input: I.In[I.f32, ("M", "K")],
    weight1: I.In[I.f32, ("K", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, input.shape[0])
    reduction = I.domain(0, input.shape[1])
    columns = I.domain(0, weight1.shape[1])

    input_value = input[rows, reduction]
    weight1_value = weight1[reduction, columns]
    weight2_value = weight2[columns]
    bias_value = bias[columns]

    activation = I.matmul(input_value, weight1_value, acc_dtype=I.f32)
    activation_shape = activation.shape
    exp_scale = I.full(activation_shape, fill=1.4426950408889634, dtype=I.f32)
    one = I.full(activation_shape, fill=1.0, dtype=I.f32)
    exp_term = I.exp2(-activation * exp_scale)
    sigmoid = I.fdiv(one, one + exp_term)

    row_ones = I.full((input.shape[0],), fill=1.0, dtype=I.f32)
    weight2_matrix = I.outer(row_ones, weight2_value)
    bias_matrix = I.outer(row_ones, bias_value)
    result = I.tanh(sigmoid) * weight2_matrix + bias_matrix

    output[rows, columns] = result[rows, columns]


def build(context):
    compiled = context.compile("combined_activation", combined_activation)

    def wrapper(input, weight1, weight2, bias, *, out=None):
        if out is None:
            return compiled.run(input, weight1, weight2, bias)
        compiled(input, weight1, weight2, bias, out)
        return out

    return wrapper
