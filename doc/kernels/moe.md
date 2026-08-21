# Ragged MoE Expert Kernel DSL 模板

这个 kernel 消费调用方已经准备好的 expert membership，不在 kernel 内隐式选择 routing/grouping 算法。

## Canonical kernel

```python
@intent.kernel
def moe_expert_ffn(
    x: I.In[I.f16, ("T", "D")],
    route_offsets: I.In[I.i32, ("E_PLUS_1",)],
    member_routes: I.In[I.i32, ("R",)],
    route_token: I.In[I.i32, ("NR",)],
    route_weights: I.In[I.f32, ("NR",)],
    w1: I.In[I.f16, ("E", "D", "F")],
    w2: I.In[I.f16, ("E", "F", "D")],
    y: I.InOut[I.f32, ("T", "D")],
):
    T, D = x.shape
    E, _, F = w1.shape
    R = member_routes.shape[0]

    groups = I.ragged(
        outer=I.domain(0, E),
        members=I.domain(0, R),
        offsets=route_offsets,
        indices=member_routes,
    )

    for expert in I.parallel(groups.outer):
        routes = I.members(groups[expert])
        token = I.gather(route_token, index=routes)
        rw = I.gather(route_weights, index=routes)
        xv = I.gather(x, index=(token, slice(None)))

        h = I.contract(
            xv,
            w1[expert, :, :],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        )
        h = I.maximum(h, 0.0)

        route_out = I.contract(
            h,
            w2[expert, :, :],
            reduce=((1, 0),),
            acc_dtype=I.f32,
        )

        I.scatter_reduce(
            y,
            index=(token, slice(None)),
            value=rw[:, None] * route_out,
            combine=I.add,
        )
```

Wrapper 按该算法的调用约定初始化 `y`，例如在 invocation 前清零。

## Source 固定

- 调用方提供的 expert membership；
- route 到 token 的 relation；
- expert-selected W1/W2 contractions；
- activation；
- router weight；
- duplicate route 对同一 token 的 `add` combine；
- `y` 的 InOut ABI。

## Physical Plan 决定

- 从完整 ragged member domain 引入的 physical route region 与 tile；
- static 或 persistent ownership；
- expert scheduling；
- contraction primitive 数值角色与算法结构要求的 storage/reuse boundary；
- gather/scatter physical mechanism。

Target leaf 只把这些决定拼成各自的 matrix/gather/scatter primitive；layout、寄存器分配、指令选择与低层 pipeline 继续交给下层。

## 边界

`I.ragged` 只解释 `route_offsets` 与 `member_routes`，不执行 grouping，不生成 offsets，也不在 histogram、sort、atomic bucket 或 radix grouping 中做选择。

Routing/grouping 是另一个算法阶段时，由 wrapper 调用明确的 Intent kernel、target-specific kernel 或普通 host-side algorithm-library implementation；realizer 不在本 kernel 内隐式选择该算法。
