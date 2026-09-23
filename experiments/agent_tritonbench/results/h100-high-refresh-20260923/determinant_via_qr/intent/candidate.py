from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.fn
def _product(lhs: I.f32, rhs: I.f32):
    return lhs * rhs


@intent.kernel
def _qr_stage(
    A: I.In[I.f32, (32, 32)],
    R: I.Out[I.f32, (32, 32)],
    qdet: I.Out[I.f32, ()],
):
    rows = I.domain(0, 32)
    state = I.buffer((32, 32), I.f32, init=A)
    q_sign = I.cast(1.0, I.f32)

    # Householder reflectors are ordered; the trailing matrix update is tensorized.
    for k in rows:
        tail = rows[k:32]
        column = state[tail, k]
        norm_sq = I.reduce.sum(column * column, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm_sq)
        x0 = state[k, k]
        sign = I.cast(I.select(x0 >= 0.0, 1.0, -1.0), I.f32)
        alpha = -sign * norm

        if norm > 0.0:
            state[k, k] = x0 - alpha
            v = state[tail, k]
            v2 = I.reshape(v, (32 - k, 1))
            vnorm_sq = I.reduce.sum(v * v, axis=0, acc_dtype=I.f32)
            beta = I.cast(2.0, I.f32) / vnorm_sq

            if k < 31:
                cols = rows[k + 1:32]
                trailing = state[tail, cols]
                projection = I.reduce.sum(v2 * trailing, axis=0, acc_dtype=I.f32)
                updated = trailing - v2 * (beta * projection)
                state[tail, cols] = updated

            state[k, k] = alpha
            q_sign = q_sign * I.cast(-1.0, I.f32)
        else:
            state[k, k] = I.cast(0.0, I.f32)

    R[rows, rows] = state[rows, rows]
    qdet[()] = q_sign


@intent.kernel
def _det_stage(
    R: I.In[I.f32, (32, 32)],
    qdet: I.In[I.f32, ()],
    out: I.Out[I.f32, ()],
):
    rows = I.domain(0, 32)
    diagonal_indices = I.indices(rows)
    diagonal = R[diagonal_indices, diagonal_indices]
    diagonal_product = I.reduce(
        diagonal,
        axis=0,
        identity=I.cast(1.0, I.f32),
        combine=_product,
    )
    out[()] = qdet[()] * diagonal_product


def build(context):
    qr_artifact = context.compile("determinant_qr_stage", _qr_stage)
    det_artifact = context.compile("determinant_product_stage", _det_stage)

    def determinant_via_qr(A, *, mode="reduced", out=None):
        R = torch.empty_like(A)
        qdet = torch.empty((), device=A.device, dtype=A.dtype)
        result = out
        if result is None:
            result = torch.empty((), device=A.device, dtype=A.dtype)

        qr_artifact(A, R, qdet)
        det_artifact(R, qdet, result)
        return result

    return determinant_via_qr
