import intent
import intent.language as I


BATCH = 128
LENGTH = 4096


@intent.kernel
def selective_state_scan(
    x: I.In[I.f32, ("B", "L")],
    decay: I.In[I.f32, ("B", "L")],
    drive: I.In[I.f32, ("B", "L")],
    output: I.Out[I.f32, ("B", "L")],
):
    B, L = x.shape
    positions = I.domain(0, L)
    for batch in I.parallel(I.domain(0, B)):
        state = I.cast(0.0, I.f32)
        for position in I.ordered(positions):
            state = (
                decay[batch, position] * state
                + drive[batch, position] * x[batch, position]
            )
            output[batch, position] = state
