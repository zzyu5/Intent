import intent
import intent.language as I


TOKENS = 4096
VOCABULARY = 16384
IGNORE_INDEX = -100


@intent.kernel
def cross_entropy_forward(
    logits: I.In[I.f32, ("M", "V")],
    labels: I.In[I.i32, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    prediction: I.Out[I.i32, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
    VOCABULARY: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        values = logits[row, classes]
        maximum, predicted = I.arg_reduce.max(
            values,
            axis=0,
            identity=-I.inf,
        )
        denominator = I.reduce.sum(
            I.exp(values - maximum),
            axis=0,
            identity=0.0,
        )
        label = labels[row]
        valid = label != IGNORE_INDEX
        safe_label = label % VOCABULARY
        I.assume_in_bounds(safe_label, logits, axis=1)
        target = logits[row, safe_label]
        loss[row] = (maximum + I.log(denominator) - target) * I.cast(
            valid, I.f32
        )
        valid_index = I.cast(valid, I.i32)
        prediction[row] = predicted * valid_index + valid_index - 1


@intent.kernel
def cross_entropy_backward(
    logits: I.In[I.f32, ("M", "V")],
    labels: I.In[I.i32, ("M",)],
    dloss: I.In[I.f32, ("M",)],
    dlogits: I.Out[I.f32, ("M", "V")],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        values = logits[row, classes]
        maximum = I.reduce.max(values, axis=0, identity=-I.inf)
        exponentials = I.exp(values - maximum)
        denominator = I.reduce.sum(exponentials, axis=0, identity=0.0)
        label = labels[row]
        valid = label != IGNORE_INDEX
        is_target = I.cast(I.indices(classes), I.i32) == label
        probability = exponentials / denominator
        gradient = (
            probability - I.cast(is_target, I.f32)
        ) * dloss[row] * I.cast(valid, I.f32)
        dlogits[row, classes] = gradient
