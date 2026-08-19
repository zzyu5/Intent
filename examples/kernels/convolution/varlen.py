import intent
import intent.language as I


PACKED_TOKENS = 5120
HIDDEN = 4096
WIDTH = 4
SEQUENCES = 4


@intent.kernel
def varlen_causal_depthwise_conv1d(
    x: I.In[I.bf16, ("U", "D")],
    sequence_offsets: I.In[I.i64, ("B_PLUS_1",)],
    weight: I.In[I.f32, ("D", WIDTH)],
    bias: I.In[I.f32, ("D",)],
    output: I.Out[I.bf16, ("U", "D")],
    final_state: I.Out[I.bf16, ("B", WIDTH - 1, "D")],
):
    U, D = x.shape
    B = final_state.shape[0]
    sequences = I.ragged(
        outer=I.domain(0, B),
        members=I.domain(0, U),
        offsets=sequence_offsets,
    )
    channels = I.domain(0, D)
    for sequence in I.parallel(sequences.outer):
        members = sequences[sequence]
        sequence_start = I.indices(members)[0]
        sequence_end = I.end(members)
        for member in I.parallel(members):
            accumulation = I.cast(bias[channels], I.f32)
            for tap in range(WIDTH):
                source_index = I.cast(member, I.i64) - tap
                valid = source_index >= sequence_start
                safe_index = I.maximum(source_index, sequence_start)
                value = I.gather(x, index=(safe_index, channels))
                accumulation = accumulation + I.mask(
                    I.cast(value, I.f32) * weight[channels, tap],
                    valid=valid,
                    fill=0.0,
                )
            sigmoid = 1.0 / (1.0 + I.exp(-accumulation))
            output[member, channels] = I.cast(accumulation * sigmoid, I.bf16)
        for history in I.parallel(I.domain(0, WIDTH - 1)):
            source_index = sequence_end - (WIDTH - 1) + history
            safe_index = I.maximum(source_index, sequence_start)
            final_state[sequence, history, channels] = I.gather(
                x,
                index=(safe_index, channels),
            )
