import intent
import intent.language as I


TOKENS = 4096
VOCABULARY = 16384
IGNORE_INDEX = -100


@intent.kernel
def fused_cross_entropy(
    logits: I.InOut[I.f32, ("M", "V")],
    labels: I.In[I.i32, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    prediction: I.Out[I.i32, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        if label == IGNORE_INDEX:
            loss[row] = 0.0
            prediction[row] = -1
            clear_stream = I.state_stream(
                classes,
                extent=I.auto("CLASS_TILE"),
                init=(I.cast(0.0, I.f32),),
            )
            with clear_stream:
                for class_region, zero in clear_stream:
                    logits[row, class_region] = zero
                    clear_stream.yield_(zero)
        else:
            I.assume_in_bounds(label, logits, axis=1)
            target = logits[row, label]
            statistics = I.state_stream(
                classes,
                extent=I.auto("CLASS_TILE"),
                init=(
                    I.cast(-I.inf, I.f32),
                    I.cast(0.0, I.f32),
                    I.cast(V, I.i32),
                ),
            )
            with statistics:
                for class_region, (maximum, denominator, predicted) in statistics:
                    values = logits[row, class_region]
                    local_maximum, local_predicted = I.arg_reduce.max(
                        values,
                        axis=0,
                        identity=-I.inf,
                    )
                    local_wins = (local_maximum > maximum) or (
                        (local_maximum == maximum) and (local_predicted < predicted)
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    local_sum = I.reduce.sum(
                        I.exp(values - next_maximum),
                        axis=0,
                        identity=0.0,
                    )
                    statistics.yield_(
                        next_maximum,
                        denominator * I.exp(maximum - next_maximum) + local_sum,
                        local_predicted if local_wins else predicted,
                    )
            maximum, denominator, predicted = statistics.result
            loss[row] = maximum + I.log(denominator) - target
            prediction[row] = predicted

            gradient_stream = I.state_stream(
                classes,
                extent=I.auto("CLASS_TILE"),
                init=(maximum, denominator),
            )
            with gradient_stream:
                for class_region, (final_maximum, final_denominator) in gradient_stream:
                    values = logits[row, class_region]
                    indices = I.cast(I.indices(class_region), I.i32)
                    probability = I.exp(values - final_maximum) / final_denominator
                    gradient = probability - I.cast(indices == label, I.f32)
                    logits[row, class_region] = gradient
                    gradient_stream.yield_(final_maximum, final_denominator)


@intent.kernel
def fused_cross_entropy_bf16(
    logits: I.InOut[I.bf16, ("M", "V")],
    labels: I.In[I.i64, ("M",)],
    loss: I.Out[I.f32, ("M",)],
    prediction: I.Out[I.i64, ("M",)],
    IGNORE_INDEX: I.Constexpr[int],
):
    M, V = logits.shape
    classes = I.domain(0, V)
    for row in I.parallel(I.domain(0, M)):
        label = labels[row]
        if label == IGNORE_INDEX:
            loss[row] = 0.0
            prediction[row] = -1
            logits[row, classes] = I.cast(0.0, I.bf16)
        else:
            I.assume_in_bounds(label, logits, axis=1)
            target = I.cast(I.gather(logits, index=(row, label)), I.f32)
            statistics = I.state_stream(
                classes,
                extent=I.auto("CLASS_TILE"),
                init=(
                    I.cast(-I.inf, I.f32),
                    I.cast(0.0, I.f32),
                    I.cast(V, I.i64),
                ),
            )
            with statistics:
                for class_region, (maximum, denominator, predicted) in statistics:
                    values = I.cast(logits[row, class_region], I.f32)
                    local_maximum, local_predicted = I.arg_reduce.max(
                        values,
                        axis=0,
                        identity=-I.inf,
                    )
                    local_predicted = I.cast(local_predicted, I.i64)
                    local_wins = (local_maximum > maximum) or (
                        (local_maximum == maximum) and (local_predicted < predicted)
                    )
                    next_maximum = I.maximum(maximum, local_maximum)
                    statistics.yield_(
                        next_maximum,
                        denominator * I.exp(maximum - next_maximum)
                        + I.reduce.sum(
                            I.exp(values - next_maximum),
                            axis=0,
                            identity=0.0,
                        ),
                        local_predicted if local_wins else predicted,
                    )
            maximum, denominator, predicted = statistics.result
            loss[row] = maximum + I.log(denominator) - target
            prediction[row] = predicted
            writer = I.state_stream(
                classes,
                extent=I.auto("CLASS_TILE"),
                init=(maximum, denominator),
            )
            with writer:
                for class_region, (final_maximum, final_denominator) in writer:
                    values = I.cast(logits[row, class_region], I.f32)
                    class_index = I.cast(I.indices(class_region), I.i64)
                    probability = I.exp(values - final_maximum) / final_denominator
                    logits[row, class_region] = I.cast(
                        probability - I.cast(class_index == label, I.f32),
                        I.bf16,
                    )
                    writer.yield_(final_maximum, final_denominator)
