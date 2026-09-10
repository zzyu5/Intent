from __future__ import annotations

import weft
import weft.language as wl


@weft.encoding
class Q4_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 2
    elements = 256
    d: wl.f16
    dmin: wl.f16
    sc: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    m: wl.u6[8] @ wl.pack_fields(group=4, fields=2, low_bits=4, order=wl.lo_first)
    q: wl.u4[256] @ wl.grouped(elements=64) @ wl.bit_layers(elements=32, order=wl.lo_first)


@weft.encoding
class Q8_K:
    layout = (wl.bitorder.lsb_first, wl.byteorder.little)
    alignment = 4
    elements = 256
    ds: wl.f32
    q: wl.i8[256]
    bsum: wl.i16[16]


def quantize_row(X: wl.View[wl.f32, (K,)], Y: wl.View[Q8_K, (K,)]):
    with wl.level.blocks(K, extent=256) as kb:
        block = wl.load(X[kb])
        maximum = wl.reduce(block, op="max")
        minimum = wl.reduce(block, op="min")
        extreme = minimum
        if wl.abs(maximum) > wl.abs(minimum):
            extreme = maximum
        inverse = wl.f32(0.0)
        d = wl.f32(0.0)
        if extreme != wl.f32(0.0):
            inverse = wl.f32(-127.0) / extreme
            d = wl.f32(1.0) / inverse
        with wl.level.subtiles(kb, extent=64) as chunk:
            group_values = wl.load(X[chunk])
            q = wl.narrow(group_values * inverse, wl.i8, rounding="rne", saturation=True)
            wl.store(Y[kb].q[chunk], q)
            with wl.level.subtiles(chunk, extent=16) as group:
                bsum = wl.reduce(wl.widen(q[group], wl.i16), op="add")
                wl.store(Y[kb].bsum[group], wl.i16(bsum))
        wl.store(Y[kb].ds, d)


def quantize_matrix(X: wl.View[wl.f32, (M, K)], Y: wl.View[Q8_K, (M, K)]):
    for row in range(M):
        quantize_row(X[row], Y[row])


def compute(W: wl.View[Q4_K, (K,)], X: wl.View[Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.level.blocks(K, extent=256) as kb:
        w = wl.load(W[kb])
        x = wl.load(X[kb])
        i32_acc = wl.i32(0)
        with wl.level.subtiles(extent=32) as sub:
            partial = wl.reduce_dot(w.q[sub], x.q[sub], over="k", acc_dtype=wl.i32)
            i32_acc += partial * wl.i32(w.sc[sub])
        min_group = wl.arange(0, 8, dtype=wl.u32, axis="min_group")
        minimum = wl.reduce(
            wl.widen(w.m[min_group], wl.i32)
            * (wl.widen(x.bsum[min_group * wl.u32(2)], wl.i32)
               + wl.widen(x.bsum[min_group * wl.u32(2) + wl.u32(1)], wl.i32)),
            axis="min_group",
        )
        result += wl.f32(x.ds) * (
            wl.f32(w.d) * wl.f32(i32_acc) - wl.f32(w.dmin) * wl.f32(minimum)
        )
    return result


@weft.kernel
def production_mul_mat_q4_k(
    W: wl.View[Q4_K, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.store(Y[row, column], compute(W[column], Xq[row]))
