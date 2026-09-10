from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def causal_attention(
    Q: wl.View[wl.f32, (B, TQ, D)],
    K: wl.View[wl.f32, (B, TK, D)],
    V: wl.View[wl.f32, (B, TK, DV)],
    O: wl.View[wl.f32, (B, TQ, DV)],
    Scale: wl.View[wl.f32, (1,)],
):
    scale = wl.load(Scale[0])
    for batch in range(B):
        for query in range(TQ):
            maximum = wl.f32(0.0)
            denominator = wl.f32(0.0)
            accumulator = wl.state(wl.f32, [DV], init=0.0)
            for key in range(TK):
                if key <= query:
                    score = wl.f32(0.0)
                    for feature in range(D):
                        score += wl.load(Q[batch, query, feature]) * wl.load(K[batch, key, feature])
                    score *= scale
                    next_maximum = wl.select(
                        denominator == wl.f32(0.0), score,
                        wl.maximum_propagating(maximum, score),
                    )
                    previous_scale = wl.select(
                        denominator == wl.f32(0.0), wl.f32(0.0),
                        wl.exp(maximum - next_maximum),
                    )
                    probability = wl.exp(score - next_maximum)
                    for feature in range(DV):
                        accumulator[feature] = (
                            accumulator[feature] * previous_scale
                            + probability * wl.load(V[batch, key, feature])
                        )
                    denominator = denominator * previous_scale + probability
                    maximum = next_maximum
            for feature in range(DV):
                divisor = wl.select(denominator == wl.f32(0.0), wl.f32(1.0), denominator)
                wl.store(O[batch, query, feature], accumulator[feature] / divisor)
