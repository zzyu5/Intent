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
        elements = I.state_stream(
            positions,
            extent=1,
            init=(I.cast(0.0, I.f32),),
        )
        with elements:
            for position, state in elements:
                next_value = (
                    decay[batch, position] * state
                    + drive[batch, position] * x[batch, position]
                )
                output[batch, position] = next_value
                next_state = I.reduce.sum(
                    next_value,
                    axis=0,
                    identity=0.0,
                    acc_dtype=I.f32,
                )
                elements.yield_(next_state)
