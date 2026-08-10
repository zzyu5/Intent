# 编译器形状义务：索引宽度、收缩尾块与退化输入

## 范围与结论

这一轮没有增加新算子，只处理作者无法替编译器回答的三类问题：地址表达式用多宽、物理分块制造出的收缩尾块如何补值、退化输入是正确运行还是明确拒绝。同时检查了注意力中真实可达的全掩码行。

最终结论如下。

| 项目 | 当前合同 | 实际证据 |
|---|---|---|
| 地址宽度 | 能安全表达宽地址的目标使用 64 位地址算术；不能表达的目标必须在窄化前拒绝 | Triton 对元素偏移 `2^31` 做了真实数值运行；cuTile、TileLang 对同一张量明确报不支持 |
| 非二次幂收缩维 | Physical Plan 记录逻辑 extent、向二次幂扩展和零填充；load/compute/store 分别消费同一决定 | 三个目标的 attention `D=64/80/96/256` 全部通过，`D=80` 确实触发物理扩展 |
| 退化形状 | 能自然表达的形状正确运行；零 extent 外部 view 当前明确不支持，不允许进入 target kernel | `1×1×1` GEMM、`Q=K=1` attention、空 ragged group 通过；`M/N/K=0` 和全空 ragged launch 明确拒绝 |
| 全掩码行 | 这是 DSL 算法语义，不由 realizer 猜；可达该状态的算法定义输出为零 | vector-bias attention 与 paged attention 在三个目标上均得到 finite 的全零输出 |

## 一、索引宽度

### 查到的问题

外部 view 的地址由逻辑索引和 runtime stride 相乘后相加。此前三个表面没有统一合同：cuTile 的机械投影固定窄化到 `int32`，TileLang 依赖其索引/切片表达式的隐式类型，Triton 也没有保证每个索引与 stride 在乘法前都已经宽化。因此普通形状可以运行，但一旦最大线性元素偏移超过有符号 32 位范围，存在静默回绕的可能。

需要判断的是元素偏移而不是单独的 tensor 元素数。当前 wrapper 对一个外部 view 使用下面的保守上界：

```text
maximum_linear_span = Σ ((extent_i - 1) * abs(stride_i))
```

这个上界也覆盖负 stride 的跨度；tensor 的 `data_ptr` 已经指向逻辑首元素，因此 storage offset 不需要再重复加入地址表达式。

### 采用的规则

没有把 `index_bits` 放进 Physical Plan。索引不回绕是编译器不变量，不是多个合法物理方案中的选择，也不应交给 tuner。

当前规则是“宽地址或明确拒绝”：

- Triton 将程序索引、动态索引和 stride 在乘加前统一转换为 `tl.int64`；
- cuTile 当前 tensor descriptor/索引张量路径仍要求 32 位，因此 wrapper 在任何 `ct.int32` 窄化前检查最大线性跨度，超过 `2147483647` 就抛出 `NotImplementedError`；
- TileLang 当前 bulk-copy 路径没有可核验的完整 64 位索引合同，因此采用同样的 launch 前能力检查，不把隐式类型当成保证。

这里选择无条件宽化 Triton，而没有做“按静态形状自动选宽度”，原因是 stride 是 runtime ABI 的一部分；只看符号 shape 得出的上界并不完整。对不能兑现宽地址的 target，显式能力边界比静默窄化更重要。

### 超 32 位实际验证

repro 使用一个真实 CUDA storage，构造：

```text
query.shape  = (2, 1, 1, 64)
query.stride = (2^31, 64, 64, 1)
```

storage 含 `2^31 + 64` 个 fp16 元素，约占 4 GiB；第二个 batch 的首元素偏移正好是 `2147483648`。这不是只查看生成源码，而是实际访问远端位置并与 PyTorch attention 对数值。

| provider | 结果 |
|---|---|
| Triton | 数值 PASS，最大绝对误差 `0.0` |
| cuTile | 能力检查 PASS：launch 前明确拒绝 64 位外部地址 |
| TileLang | 能力检查 PASS：launch 前明确拒绝 64 位 bulk-copy 地址 |

所以这一轮消除了静默错误，但没有声称 cuTile 与 TileLang 已经具备大于 32 位的实际地址能力。

## 二、非典型收缩维

### 查到的问题

attention 的头维 `D`/`DV` 是 contraction 中没有独立逻辑 domain 的隐式轴。原先物理 tile shape 直接沿用逻辑 extent，而且 autotune specialization key 主要包含 query/key 轴；不同头维可能复用不该复用的选择。普通 GEMM 的显式 reduction domain 尾块已经走过，但这不能自动证明隐式 contraction extent 也闭合。

### 物理决定

GPU realizer 现在从 canonical contraction facts 收集“无独立 domain 且 extent 不为 1”的轴，生成：

```mlir
intent_plan.block_extent
  logical_extent = "D"
  rounding = "power_of_two"
  fill = "zero"
```

它不看 kernel 名字。`D`/`DV` 同时进入 autotune specialization key，避免跨头维复用选择。

三个 target 共享同一条物理语义：

- on-chip tensor 和矩阵原语使用向上取整后的 physical extent；
- 外部 load/store 始终按 logical extent 做边界限制；
- contraction 尾部 load 使用乘法-加法的恒等元 `0`；
- target emitter 通过共享的 block-extent 投影读取 fill。即使某个 future transfer 没有另行携带 boundary fill，也不会因为 physical shape 扩大而无掩码越界读取。

Triton 将 physical extent 写成 `next_power_of_2(D)`，cuTile 与 TileLang 在 wrapper/builder 中物化 `PHYSICAL_D`；layout、寄存器和指令选择仍交给下层。

### 数值结果

同一份 dense-attention DSL 在三个 provider 上都运行了以下形状：

| Q | K | D | Triton | cuTile | TileLang |
|---:|---:|---:|---:|---:|---:|
| 127 | 131 | 64 | `6.10e-5` | `6.10e-5` | `6.10e-5` |
| 127 | 131 | 80 | `1.22e-4` | `1.22e-4` | `1.22e-4` |
| 127 | 131 | 96 | `1.22e-4` | `1.22e-4` | `1.22e-4` |
| 127 | 131 | 256 | `1.22e-4` | `1.22e-4` | `1.22e-4` |
| 1 | 1 | 80 | `0.0` | `0.0` | `0.0` |

表中是相对 PyTorch reference 的最大绝对误差。`D=80/96` 不是二次幂，`Q=127/K=131` 也同时压到了普通轴尾块。

另外还运行了三个组合路径：

- causal varlen attention：`D=80`，序列长度 `[0, 1, 79, 131]`，三个目标均 PASS；
- vector-bias attention：`D=80` 且整行 bias 为 `-inf`，三个目标均 PASS；
- paged attention：`D=80`、长度为零但 page metadata 存在，三个目标均 PASS。

普通 GEMM 的 `M=4093, K=4080, N=14320` 尾块，以及 grouped GEMM 的 `K=80, N=96` 也在三个目标上通过。

### TileLang 的取舍

TileLang 的 staged/deferred contraction 使用其现成 `T.copy` 与 `T.gemm`。曾尝试在 target emitter 中把所有尾块展开为显式逐元素清零和条件读取；它没有改变算法，却让正常 grouped GEMM 从约 `1.28 ms` 退到约 `1.86 ms`。当前实现保留 target 原语的边界处理，并用真实 K/N 尾块验证其数值结果。

这里仍有一项证据边界：我们已观察到当前目标原语在这些实际尾块上给出正确结果，但没有把 TileLang 下层对所有切片形态的内部证明复制进本编译器。复制那套逻辑既会使 emitter 变厚，也已经表现出明显性能代价。

## 三、退化形状

退化形状分为“仍有合法工作”和“外部 view 本身为空”两类，没有用一个含糊的 fallback 混在一起。

| 场景 | 当前行为 | 三目标结果 |
|---|---|---|
| GEMM `1×1×1` | 正常生成并运行 | 数值 PASS |
| attention `B=1,Q=1,K=1,D=80` | 正常生成并运行 | 数值 PASS，误差 `0.0` |
| grouped GEMM 中首组/末组为空 | 重复 offset 表示空 group，其余 group 正常运行 | 数值 PASS |
| varlen attention 中一条序列为空 | 重复 offset；空序列没有输出 row，其余序列正常运行 | 数值 PASS |
| GEMM `M=0`、`N=0` 或 `K=0` | wrapper 明确报当前不支持 zero-extent external view | 拒绝合同 PASS |
| grouped GEMM 所有 group 均为空 | 输入 row extent 为零，明确拒绝 | 拒绝合同 PASS |

没有自动把 `K=0` GEMM 改写为“填零输出”，也没有让单行 contraction 暗中走一个慢很多的串行 fallback。当前三个目标能表达的 `1×1×1` 形状直接运行；不能表达的零 extent 在 launch 前得到可读错误。

## 四、注意力全掩码行

检查结果不是“所有 attention 都统一插入保护”，而是按算法可达性处理。

vector-bias attention 与 paged attention 可以真实地产生没有任何有效 key 的输出 row。两份 DSL 源码现在明确写出：

1. 若 online maximum 仍为 `-inf`，归一化用的 maximum 取 `0`，避免 `-inf - -inf`；
2. 对这类 row，历史缩放因子和当前 probability 都成为 `0`；
3. 若 denominator 不大于 `0`，除法使用安全 denominator `1`；
4. 因而算法定义该 row 的 output 为全零。

三个 target 对两种 kernel 都得到 finite 的全零结果，保护来自同一份 DSL 算法，不是 realizer 或 emitter 特判。

普通 dense attention、causal varlen attention 和 causal GQA 在当前接口合同下没有可达的全掩码输出 row：非空 causal 序列的每个 query 至少能看到自身位置；空 varlen 序列没有 query row；外部 K extent 为零会被明确拒绝。因此没有为了一个不可达状态给这些源码增加额外操作。若以后加入允许任意 mask 的接口，安全归一化必须由那份 DSL 算法明确陈述。

## 五、性能哨兵

下表来自本轮最终实现的现有 repro。时间均为 CUDA Graph 下的 kernel-only p50；没有 baseline 的格子不伪造比较。

| kernel | provider | generated | upstream | generated/upstream |
|---|---|---:|---:|---:|
| dense attention | Triton | `5.0524 ms` | `4.9219 ms` | `1.0265×` |
| dense attention | cuTile | `5.0043 ms` | `4.8076 ms` | `1.0409×` |
| dense attention | TileLang | `4.8540 ms` | `6.6918 ms` | `0.7254×` |
| paged attention | Triton | `0.2677 ms` | `0.2935 ms` | `0.9122×` |
| paged attention | cuTile | `0.3617 ms` | — | — |
| paged attention | TileLang | `1.3563 ms` | — | — |
| stable softmax | Triton | `0.3666 ms` | `0.3661 ms` | `1.0013×` |
| stable softmax | cuTile | `0.3686 ms` | `0.3682 ms` | `1.0012×` |
| stable softmax | TileLang | `0.3537 ms` | `0.3707 ms` | `0.9543×` |

与本轮最后一次共享 fill 投影调整前相比，dense attention、paged attention 和 softmax 的生成结构没有多出 target-side 决策，p50 也没有可观察的退化。短 kernel 的几千分之一毫秒波动不作为结构性回退结论。

## 六、实际验证范围

本轮只使用项目既有的手工 repro 入口，没有新增 test 目录、pytest、fixture 或矩阵脚手架。实际运行包括：

```bash
./examples/run/repro.sh triton attention
./examples/run/repro.sh cutile attention
./examples/run/repro.sh tilelang attention

./examples/run/repro.sh triton paged_attention
./examples/run/repro.sh cutile paged_attention
./examples/run/repro.sh tilelang paged_attention

./examples/run/repro.sh triton softmax
./examples/run/repro.sh cutile softmax
./examples/run/repro.sh tilelang softmax
```

同一轮还分别在三个 provider 上运行了 `gemm`、`grouped_gemm`、`varlen_attention` 和 `attention_bias`，用于覆盖零 extent、空 group、空序列与全掩码行。

这份报告不把较早的全 kernel 矩阵数字混进当前实现：本轮没有重新执行所有无关入口，因此不能把历史的全量 PASS 计数当成本轮证明。当前证据覆盖了所有被修改的语义路径，并用 softmax 作为不相关路径的性能哨兵。

## 七、判断、未验证项与边界

- **已确认：**Triton 的实际地址乘加是 64 位，并真实访问了元素偏移 `2^31` 的位置。
- **已确认：**cuTile 与 TileLang 不再静默接受同一地址；当前是明确的 target capability 限制。
- **已确认：**`D=80/96` 的 implicit contraction 尾块在三个 target 上都数值正确，fill 来自共享 Physical Plan 投影。
- **已确认：**个别 ragged group 为空与整次 launch 的外部 extent 为零是两个不同合同；前者支持，后者明确拒绝。
- **判断选择：**地址规则采用“宽化或拒绝”而不是只靠 shape 推导位宽，因为 runtime stride 参与最终上界。
- **判断选择：**没有在 TileLang emitter 复制下层 `T.copy` 的逐元素边界实现；实际尾块正确且复制实现造成约 45% 的正常路径退化。
- **尚未证明：**cuTile 和 TileLang 的大于 32 位地址执行能力。本轮只证明它们会在错误发生前拒绝。
- **尚未重跑：**与这些机制无关的完整全 kernel × 三 provider 矩阵；报告没有用旧日志替代这一事实。
