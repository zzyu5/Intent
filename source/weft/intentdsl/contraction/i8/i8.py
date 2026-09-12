from __future__ import annotations

import weft
import weft.language as wl


def matmul_bias(A, B, Bias, C):
    with wl.level.rows(M, group=wl.auto("MR")) as mb:
        with wl.level.cols(N, group=4) as nb:
            accumulator = wl.state(wl.i32, [MR, 4], init=0)
            with wl.level.blocks(K, extent=128) as kb:
                partial = wl.state(wl.i32, [MR, 4], init=0)
                with wl.level.subtiles(extent=8) as k:
                    a = wl.load(A[mb, k])
                    b = wl.load(B[k, nb])
                    partial += wl.dot(a, b, over="k", acc_dtype=wl.i32)
                accumulator += partial
            wl.store(C[mb, nb], accumulator + wl.load(Bias[nb]))


@weft.kernel
def gemm_bias(A: wl.View[wl.i8, (M, K)], B: wl.View[wl.i8, (K, N)],
              Bias: wl.View[wl.i32, (N,)], C: wl.View[wl.i32, (M, N)]):
    matmul_bias(A, B, Bias, C)
