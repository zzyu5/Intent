import intent
import intent.language as I


PARAMETERS = 8 * 1024 * 1024


@intent.kernel
def adamw_update(
    gradient: I.In[I.f32, ("N",)],
    parameter: I.InOut[I.f32, ("N",)],
    first_moment: I.InOut[I.f32, ("N",)],
    second_moment: I.InOut[I.f32, ("N",)],
    learning_rate: I.f32,
    beta1: I.f32,
    beta2: I.f32,
    inverse_bias1: I.f32,
    inverse_bias2: I.f32,
    epsilon: I.f32,
    weight_decay: I.f32,
):
    N = parameter.shape[0]
    for index in I.parallel(I.domain(0, N)):
        gradient_value = gradient[index]
        first = beta1 * first_moment[index] + (1.0 - beta1) * gradient_value
        second = (
            beta2 * second_moment[index]
            + (1.0 - beta2) * gradient_value * gradient_value
        )
        corrected_first = first * inverse_bias1
        corrected_second = second * inverse_bias2
        denominator = 1.0 / I.rsqrt(corrected_second) + epsilon
        updated = (
            parameter[index] * (1.0 - learning_rate * weight_decay)
            - learning_rate
            * corrected_first
            / denominator
        )
        first_moment[index] = first
        second_moment[index] = second
        parameter[index] = updated
