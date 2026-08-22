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
            for q_index in I.parallel(q_axis):
                q_row = q[b, h, q_index, :]

                stream = I.state_stream(
                    k_axis,
                    extent=I.auto("K_TILE"),
                    init=(
                        I.cast(-I.inf, I.f32),
                        I.cast(0.0, I.f32),
                        I.zeros((DV,), dtype=I.f32),
                    ),
                    stop=(q_index + 1) if CAUSAL else I.end(k_axis),
                )

                with stream:
                    for kr, (m, l, acc) in stream:
                        k_block = k[b, h, kr, :]
                        v_block = v[b, h, kr, :]

                        scores = I.contract(
                            q_row,
                            k_block,
                            reduce=((0, 1),),
                            acc_dtype=I.f32,
                        )

                        scores = scores * (scale * I.LOG2E)

                        if CAUSAL:
                            k_idx = I.indices(kr)
                            valid = q_index >= k_idx
                            scores = I.mask(
                                scores,
                                valid=valid,
                                fill=-I.inf,
                            )

                        local_m = I.reduce.max(
                            scores,
                            axis=0,
                            identity=-I.inf,
                        )
                        next_m = I.maximum(m, local_m)

                        alpha = I.exp2(m - next_m)
                        p = I.exp2(scores - next_m)

                        next_l = (
                            alpha * l
                            + I.reduce.sum(p, axis=0, identity=0.0)
                        )

                        p_low = I.cast(p, I.f16)
                        next_acc = (
                            alpha * acc
                            + I.contract(
                                p_low,
                                v_block,
                                reduce=((0, 0),),
                                acc_dtype=I.f32,
                            )
                        )

                        stream.yield_(next_m, next_l, next_acc)

                _, l, acc = stream.result
                out[b, h, q_index, :] = I.cast(
                    acc / l,
                    I.f16,
                )
```

## Source 固定

- 彼此独立的 scalar Q instances；
- 沿 K/V 的 ordered stream；
- `(m, l, acc)` carry state；
- QK 与 PV contractions；
- causal logical predicate；
- base-2 `exp2` 路径；
- PV operand 的显式 f16 cast；
- online update 与 final normalization。

## Physical Plan 决定

- 从独立 Q instances 与 K stream 引入的 Q/K physical regions 与 tile；
- worker ownership、program mapping 与 persistent strategy；
- 算法结构要求的 Q/K/V/acc storage/reuse boundary；
- contraction primitive 数值角色、logical validity 与 tail；
- generated launch ownership 与合法 tuner 参数轴。

Fragment layout、寄存器分配、指令选择以及给定候选后的低层 pipeline、prefetch 与 warp specialization 交给目标 compiler。

## 作者显式的 split-K decode

上游算法若以多个 launches 组成 split-K decode，Intent source 也写成多个 `@intent.kernel`。第一段用 `partition(keys, count=P)` 的 source part identity 分别写出 partial LSE/normalizer 与 partial output；第二段读取全部 `P` 个 slots，按同一 online-normalization 代数合并。Wrapper 负责初始化 partial buffer、依次 launch 两个 kernels 并传递中间张量。

这里的 `P`、part 边界和 partial-buffer ABI 是作者算法可观察内容；每个 part 内进一步采用多大的 physical K tile、worker/grid、storage 与 pipeline 仍由 realizer/provider 决定。Compiler 不把一个单-kernel attention 自动拆成 split-K，也不把作者的两个 kernels 融成一个 launch。

Packed varlen 形式不引入另一类 attention schedule。Source 用一个 ragged relation
把 `sequence -> packed token range` 写进 Kernel IR；同一 relation 的 outer domain
由 parallel ownership range 拥有，query member instances 彼此独立，另一个 member domain 由
`state_stream` 顺序遍历。Plan 可以物理批处理 query instances，并在相应逻辑轴上组合不规则 membership、parallel ownership range 与 ordered traversal range，三个
GPU surface 只把同一组合投影成各自的 grid、offset load 和 streamed loop。

## 边界

`state_stream` 固定 K/V segment order 与 carry update，不允许替换为 parallel partial-state merge。
`state_stream.stop` 是作者声明的 exclusive logical read bound；它可以是与 streamed axis 同坐标系的 index expression，`I.end(...)` 是从 domain/region 取得该表达式的 convenience。Plan 保存该 logical expression，
物理层再把 owner end 与 stream tile 组合成循环上界。它不从 causal mask 反推读取范围。

`I.mask` 表达 logical validity。Realizer 可以消除能够整体证明无效的 physical region；不能凭空发明 block-sparse skipping。Sparse descriptor 必须由 wrapper/source 提供。

Logical region 的尾块可以短于 physical tile。Transfer 的边界处理必须保证地址安全；
该有效性若穿过 contraction 进入 reduction，还必须继续作为 value validity 传播，不能
把 load 的 zero padding 当成 softmax score 的逻辑零值。
