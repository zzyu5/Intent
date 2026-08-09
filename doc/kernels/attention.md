# FlashAttention Forward DSL 模板

## Canonical kernel

```python
@intent.kernel
def flash_attention_fwd(
    q: I.In[I.f16, ("B", "H", "Q", "D")],
    k: I.In[I.f16, ("B", "H", "K", "D")],
    v: I.In[I.f16, ("B", "H", "K", "DV")],
    out: I.Out[I.f16, ("B", "H", "Q", "DV")],
    scale: I.f32,
    CAUSAL: I.Constexpr[bool],
):
    B, H, Q, D = q.shape
    _, _, K, _ = k.shape
    DV = v.shape[-1]

    q_axis = I.domain(0, Q)
    k_axis = I.domain(0, K)

    for b in I.parallel(I.domain(0, B)):
        for h in I.parallel(I.domain(0, H)):
            for qr in I.parallel(
                I.partition(q_axis, extent=I.auto("Q_TILE"))
            ):
                q_block = q[b, h, qr, :]

                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.full((qr,), -I.inf, dtype=I.f32),
                        I.zeros((qr,), dtype=I.f32),
                        I.zeros((qr, DV), dtype=I.f32),
                    ),
                    stop=I.end(qr) if CAUSAL else I.end(k_axis),
                )

                with stream:
                    for kr, (m, l, acc) in stream:
                        k_block = k[b, h, kr, :]
                        v_block = v[b, h, kr, :]

                        scores = I.contract(
                            q_block,
                            k_block,
                            reduce=((1, 1),),
                            acc_dtype=I.f32,
                        )

                        scores = scores * (scale * I.LOG2E)

                        if CAUSAL:
                            q_idx = I.indices(qr)
                            k_idx = I.indices(kr)
                            valid = q_idx[:, None] >= k_idx[None, :]
                            scores = I.mask(
                                scores,
                                valid=valid,
                                fill=-I.inf,
                            )

                        local_m = I.reduce.max(
                            scores,
                            axis=1,
                            identity=-I.inf,
                        )
                        next_m = I.maximum(m, local_m)

                        alpha = I.exp2(m - next_m)
                        p = I.exp2(scores - next_m[:, None])

                        next_l = (
                            alpha * l
                            + I.reduce.sum(p, axis=1, identity=0.0)
                        )

                        p_low = I.cast(p, I.f16)
                        next_acc = (
                            alpha[:, None] * acc
                            + I.contract(
                                p_low,
                                v_block,
                                reduce=((1, 0),),
                                acc_dtype=I.f32,
                            )
                        )

                        stream.yield_(next_m, next_l, next_acc)

                _, l, acc = stream.result
                out[b, h, qr, :] = I.cast(
                    acc / l[:, None],
                    I.f16,
                )
```

## Source 固定

- Q region algorithm；
- 沿 K/V 的 ordered stream；
- `(m, l, acc)` carry state；
- QK 与 PV contractions；
- causal logical predicate；
- base-2 `exp2` 路径；
- PV operand 的显式 f16 cast；
- online update 与 final normalization。

## Physical Plan 决定

- Q/K physical tile；
- worker ownership、program mapping 与 persistent strategy；
- Q/K/V/acc placement；
- MMA 与 fragment layout；
- pipeline、prefetch、warp specialization 与 tail；
- generated launch configuration。

Packed varlen 形式不引入另一类 attention schedule。Source 用一个 ragged relation
把 `sequence -> packed token range` 写进 Kernel IR；同一 relation 的 outer domain
由 program ownership 拥有，query member domain 被分块，另一个 member domain 由
`state_stream` 顺序遍历。Plan 因而组合 `ragged ownership + ordered_stream`，三个
GPU surface 只把同一组合投影成各自的 grid、offset load 和 streamed loop。

## 边界

`state_stream` 固定 K/V segment order 与 carry update，不允许替换为 parallel partial-state merge。
`I.end(...)` 是作者声明的 exclusive logical read bound；Plan 保存该 logical node，
物理层再把 owner end 与 stream tile 组合成循环上界。它不从 causal mask 反推读取范围。

`I.mask` 表达 logical validity。Realizer 可以消除能够整体证明无效的 physical region；不能凭空发明 block-sparse skipping。Sparse descriptor 必须由 wrapper/source 提供。

Logical region 的尾块可以短于 physical tile。Transfer 的边界处理必须保证地址安全；
该有效性若穿过 contraction 进入 reduction，还必须继续作为 value validity 传播，不能
把 load 的 zero padding 当成 softmax score 的逻辑零值。
