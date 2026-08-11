import intent
import intent.language as I


SLOTS = 65536


@intent.kernel
def claim_zero_slots(
    state: I.InOut[I.i32, ("N",)],
    previous: I.Out[I.i32, ("N",)],
):
    N = state.shape[0]
    for index in I.parallel(I.domain(0, N)):
        old = I.atomic_cas(
            state,
            index,
            compare=0,
            value=1,
            ordering="relaxed",
            scope="device",
        )
        previous[index] = old
