from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def causal_linear_attention(
    Q: wl.View[wl.f32, (B, S, D)],
    K: wl.View[wl.f32, (B, S, D)],
    V: wl.View[wl.f32, (B, S, DV)],
    O: wl.View[wl.f32, (B, S, DV)],
    Final: wl.View[wl.f32, (B, D, DV)],
):
    for batch in range(B):
        state = wl.state(wl.f32, [D, DV], init=0.0)
        for token in range(S):
            for feature in range(D):
                key = wl.load(K[batch, token, feature])
                for channel in range(DV):
                    state[feature, channel] = state[feature, channel] + key * wl.load(V[batch, token, channel])
            for channel in range(DV):
                result = wl.f32(0.0)
                for feature in range(D):
                    result += wl.load(Q[batch, token, feature]) * state[feature, channel]
                wl.store(O[batch, token, channel], result)
        for feature in range(D):
            for channel in range(DV):
                wl.store(Final[batch, feature, channel], state[feature, channel])
