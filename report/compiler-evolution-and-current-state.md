# 从边界复审到性能强化：统一推进报告

## 1. 报告范围

本文统一梳理从用户提出“第一轮：收尾遗留 + 重新自查路径”开始，到最近一次“对有真实 baseline 的算子持续追平上游”的完整推进过程。它替代这一阶段按轮次产生的零散汇报；固定性能数字仍只保存在：

- `report/baseline/kernel-performance.csv`：RTX 5090；
- `report/baseline/kernel-performance-h100.csv`：H100；
- `report/baseline/compiler-closure.md`：更早的 42-repro 历史快照，不再滚动更新。

编程模型、Kernel IR、Physical Plan 与 target-family 边界的长期判断单独保存在 `report/programming-model-and-compiler-architecture.md`。本文只回答三个问题：每轮要求解决什么、实际怎样修改编译器、当前整体到底处于什么状态。

这一阶段对应的提交顺序为：

| 阶段 | 提交 | 作用 |
|---|---|---|
| 前置双机核验 | `9c9a390` | 验证 Plan 收缩后的 118 条记录，形成第一轮的输入问题 |
| 第一轮代码 | `f64124b` | 闭合 E8M0 cast 与 staged-axis binding |
| 第一轮结论 | `04af9e8` | 关闭边界复审中的遗留分类 |
| 第二轮 | `60e5e59` | 改善 cuTile 投影并拆开 Plan 消费与 target spelling |
| 第三轮 | `0487248` | 增加真实 vendor 算法并修 pointwise ownership |
| 第四轮 | `224e510` | 双机全量、修轴角色时序与 cuTile gather 候选 |
| 第四轮补丁 | `c10b268` | 修正 cuTile emitter 中的局部名称遮蔽，恢复 C++ 构建 |
| 性能强化一 | `87288d4` | 建立 Triton row-vector launch 调整路径 |
| 性能强化二 | `132a397` | 建立 cuTile row occupancy 目标内调优 |
| 性能强化三 | `6d66d8a` | 用 cuTile `permute` 投影转置 contraction operand |
| 性能强化收敛 | `88a3742` | 收紧 Triton row launch 的适用结构并稳定选择 |
| 数字固定 | `29f1e4e` | 只刷新两张 baseline 的受影响 generated 数字 |

## 2. 进入第一轮时的真实起点

第一轮不是从空白开始。此前已经把 Python frontend、canonical Kernel MLIR、共享 GPU Physical Plan 和三个 surface emitter 接成统一链路，并完成一次 Plan schema 收缩：可由 Kernel IR 或其他已选决定唯一重算的 stage、range、scan 与 pointwise 字段被删除，leaf 不应再把这些派生事实当作第二份真理。

`9c9a390` 随后在 RTX 5090 与 H100 上并行执行 109 个 runner × 3 个 provider，每张表展开为 118 条 kernel/case。这个核验确认 Plan 收缩本身没有造成已有 pass 的数值退化，同时留下了第一轮必须重新判断的事项：

1. E8M0 在 TileLang 的 sm120 与 sm90a 路径表现不一致，原先只覆盖了一个 f32 样本；
2. staged contraction 的 feature axis 仍可能靠结果轴顺序猜测；
3. compile timeout、compile failure 与 capability unsupported 的边界需要逐格复核；
4. `private_workspace` 两个预算系数和 `partition(count=P)` 长期挂账，但尚无证据说明应该修改。

这个起点很重要：后续没有把“挂了很久”当作实现理由，而是继续要求真实 kernel 或跨设备反例先证明问题存在。

## 3. 第一轮：收尾遗留并重新审查编译器路径

### 3.1 本轮要求

本轮不是重复扫一遍代码，而是用固定四问重新检查 generic combine、stage execution、E8M0 和 GQA validity：

1. 作者是否已经把语义写进 Kernel IR；
2. 下层 target 是否已经原生承担；
3. 当前事实究竟属于算法语义、可重算分析、已选物理决定还是 target spelling；
4. 换成另一种 GPU surface，或者未来换成 RVV/CPU 后，这个归属是否仍成立。

### 3.2 修复一：E8M0 形成唯一 dtype 合同

原先的 E8M0 路径只保证当前 block-scaled f32 样本能跑：Triton 会把 `0xff` 解成无穷而不是 NaN，Triton 与 TileLang 的 E8M0→f16 也没有完整语义。修复后：

- E8M0 的 byte encoding 是 Kernel IR dtype 语义；
- Triton 与 TileLang leaf 都按相同位编码机械解码到 canonical f32，再按目标 dtype cast；
- byte 0、1..254 与 255 的语义分别明确处理；
- cuTile 已有原生 dtype 支持，继续委托，不复制一份 decode；
- 没有 GPU 型号、kernel 名称或 block-scaled shape 分支。

实际修改集中在 Triton/TileLang cast handler；Physical Plan 没有增加 dtype 语义字段。

一次性 raw-byte probe 使用 `[0, 1, 127, 254, 255]`，5090 三个 provider 与 H100 Triton/TileLang 的 f32/f16 结果一致并正确保留 NaN；H100 cuTile 由下层明确拒绝 sm90 上的 E8M0。真实 `block_scaled_matmul` 在两台设备的 Triton/TileLang 上最大误差均为 `7.629e-06`。

### 3.3 修复二：staged feature axis 不再按顺序猜

原 realizer 从 staged contraction 结果轴中倒序选择最后一个非 member、非 unit axis。现有 MoE/grouped GEMM 恰好只有一个候选，因此错误假设一直藏住。

修改后的共享判据是：

- 没有候选：诊断缺少 staged feature axis；
- 唯一候选：建立 typed StageAxis binding；
- 多个候选：在 Plan 构造时诊断歧义；
- 三个 leaf 没有机会按轴名或结果顺序补猜。

5090 上 MoE 与 grouped GEMM × 三个 provider 共 6 条直接消费者均通过数值对照。

### 3.4 遗留状态如何被关闭

- `private_workspace` 的预算系数被定性为可回退的 GPU placement policy。选错会表现为资源或性能问题，不是算法 correctness；现有真实边界没有证明系数错误，因此不改。
- `partition(count=P)` 被定性为“语义已定义、当前 realizer 未实现的 Core 子集”。全部语料仍没有真实算法必须观察固定 part count；frontend 在产生 IR 前明确拒绝，不静默改成 extent。
- `token_sparse_mla_prefill` 的长编译保留为下层 compile cost，不伪装成 unsupported。
- 5090 TileLang `absorbed_mla_prefill` 能生成 target source，但候选分别撞到 layout conflict 或 shared-memory 资源上限，因此从宽泛的 unsupported 改为 `compile_failed`。

### 3.5 本轮得到的架构结论

Generic combine 路径保持干净：typed closure 的 schema、identity、capture 和 pure body 都在 Kernel IR；Plan 只选 realization；Triton/cuTile 机械委托原生 reduce/scan，TileLang 0.1.13 在 emission 前明确拒绝 generic closure。Stage execution 的公共合同也站得住，但当前 GPU producer 的真实能力子集是 ragged staged contraction，不能把它夸大成任意程序自动分 stage。

## 4. 第二轮：提高目标源码质量并解耦物化层

### 4.1 本轮要求

这轮不新增语言能力、不改变算法和 Plan，只处理“同一个既定物理决定怎样用目标语言更直接地表达”，并验证 emitter 内部是否真的需要把 Plan 消费与 API 拼写分开。

### 4.2 cuTile 间接读取改用原生 masked gather

并排阅读生成源码后发现，四种 cuTile 间接读取都会先生成：

```python
value = ct.gather(..., check_bounds=True, padding_value=fill)
value = ct.where(logical_mask, value, fill)
```

而 cuTile 的 `ct.gather` 已经接受 `mask`、`check_bounds` 与 `padding_value`。第二条 `where` 只是重复兑现同一份 validity，不是算法语义或新的物理决定。本轮把 staged indirect gather、ordered indexed member、staged contraction indirect lhs 与 route materialization 统一投影成一次 masked gather。

5090 定向结果：

| case | 修改前 | 修改后 | 数值 |
|---|---:|---:|---|
| cuTile paged attention | 0.3436 ms | 0.3270 ms | pass |
| cuTile MoE | 10.2724 ms | 10.0743 ms | pass |

这项改动后来在第四轮被 H100 证明不能固定为唯一拼写；第四轮没有撤销 native masked gather，而是把两种等价拼写收敛成 cuTile target-local tuner 的离散候选。这个演变是本阶段最清楚的“先有单机证据，再由跨设备反例决定是否搜索”。

### 4.3 emitter 内部模块化

三个 emitter 新增各自的 `Emission/Syntax/Spelling` 模块：

- `Source`/`Handlers` 读取 Kernel IR 与 Plan，完成 capability、角色、类型、空间和 validity 的消费；
- `Syntax/Spelling` 只把已经确定的概念打印成当前 target API 名、参数名和表达式。

它不是新的 IR 或表示层：没有 schema、验证器、状态或独立事实。真实维护成本来自两处既有问题——tile 参数名和 tuner 参数名曾重复维护，E8M0 跨架构修复也散落在逐 op handler 的字符串拼接里。拆分后，目标库 API 改名或调用形式变化只需要修改对应 target 的 spelling，不再穿过 Plan 消费逻辑。

两项没有真实消费者支撑的“顺手优化”被撤回：TileLang padding 临时拷贝消除、Triton `mask=True` 省略。它们没有因为看起来整洁就留在代码里。

## 5. 第三轮：从 vendor source 增加陌生算法并定向追性能

### 5.1 本轮要求

这轮只从 `source/` 已经引入、但尚未成为 DSL baseline 的真实上游实现中选算法；先确定算法再看编译器是否能处理，不为了迁就现有能力改写。同时只测新增项与真实受影响项，不做全量。

### 5.2 新增的真实算法与 baseline entry

新增两个算法族、四条记录：

1. `causal_depthwise_conv1d_update`：one-token state-cache 更新。一次调用同时原地左移/写入 state，并完成 f32 累加和 SiLU；这不是已有整段 causal convolution 的改名。
2. Q/K in-place RoPE：`rotary_qk_inplace` 与 partial 版本形成 full、partial、inverse 三个 entry。固定形状为 `B=4, S=2048, QH=32, KH=8, D=128`，Q/K 头数不同，partial 只旋转前 64 维。

已有 vendor code 直接作为结构和可比 runtime 来源，没有再复制 source。causal update 的 vendor 目录无法拆出同 ABI、同范围的独立高性能 kernel，因此 source 数字保持空白；RoPE 的 cuTile baseline 能做到同算法、同调用数与同计时范围，因此接入真实对照。

### 5.3 新算法暴露的 shared compiler 问题

#### Ordinary store 丢失 tensor-indexing

RoPE 的第二半写回地址是 `phase + half_dimension`。表达式已进入 Kernel IR，但 ordinary `view_store` 原来只分析 boundary，没有记录 tensor-indexing；cuTile leaf 因此会把动态张量下标误当普通 block store。

修复是：

- `KernelFacts` 的 ordinary store handler 同样执行 `classifyTensorIndices`；
- cuTile store handler 只读取 Plan 的 `tensor_indexing`，非 `none` 时机械投影为 `ct.scatter`。

这不是 RoPE 特判。旧的 `shifted_row_copy` 与 `reshape_and_cache` 在两台机器上都做了定向数值回归。

#### Pointwise lane 一直被钉成整行

Embedding 的 DSL 已经显式 partition token，feature domain 只有 pointwise load/store。旧 Plan 却让一个 program 处理完整 1021-wide feature row，无法形成上游常见的二维 `(token block, feature block)` grid。

共享 realizer 现在只在以下条件同时满足时为 lane 增加独立 program ownership：作者显式写了 partition；用途只有 load/store/gather；所有用途属于同一 parallel owner；没有另一个 vector domain；该轴不承担 reduction、scan、contraction 或 region-level mask。

第一版漏掉“owner 必须来自作者显式 partition”，使 `shifted_row_copy` 的标量 row 也被错误拆成二维 grid，H100 cuTile 从 0.0163 ms 退到 0.0245 ms。补齐判据后恢复到 0.0164 ms，embedding 的收益保留。这证明最终判断基于作者 body 看见 region 还是 scalar，而不是 kernel 名或 shape 阈值。

#### `I.indices` 的可投影边界

顶层 execution region 外构造 `I.indices(domain)` 没有当前 selected physical range。过去 frontend 仍生成 SSA，直到 leaf 才报 target value 缺失。Kernel IR verifier 现在在源码位置明确拒绝该写法，不再声称一条名义上存在但无法 lowering 的能力。

### 5.4 结果

5090 上，causal update 的 Triton/cuTile/TileLang p50 分别为 0.0771/0.0277/0.0153 ms；H100 为 0.0413/0.0259/0.0138 ms。State 对照误差为 0，输出误差不超过 `1.22e-4`。

H100 的 12 个 RoPE generated 格全部通过。5090 Triton/cuTile 通过；TileLang 已生成完整 CUDA，但 CUDA 12.8 的 `nvcc -arch=sm_120a` 崩溃，所以记录为 failed 而不是 unsupported。

Embedding 的共享 ownership 修复带来：

| 设备 | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| RTX 5090 | 0.1543→0.0442 ms | 0.1085→0.0670 ms | 0.0668→0.0446 ms |
| H100 | 0.1029→0.0328 ms | 0.1437→0.0530 ms | 0.1472→0.0287 ms |

三个 target、两台设备都受益，说明改的是共享物理分配能力，而不是某个 provider 的局部技巧。

## 6. 第四轮：双设备全量回归与成熟状态收尾

### 6.1 全量范围

两台机器并行执行同一代码基线：每台 113 个 runner × 3 个 provider，共 339 次完整的 DSL→Kernel MLIR→Physical Plan→target source→下层编译→GPU 数值与计时；多 case 展开后为 122 条记录。

累计执行时间：RTX 5090 约 42 分 49 秒，H100 约 43 分 46 秒；两机并行，主等待时间不是二者相加。

### 6.2 回归一：pointwise ownership 的决定时机过早

`fp8_mqa_logits` 数值正确，但 Triton 在两台机器分别退化约 44% 和 20%。错误 Plan 中 head axis 同时得到：

```text
["lane", "parallel", "contraction_m"]
```

根因是 pointwise promotion 在 contraction M/N 角色尚未分配时就读取了不完整 facts。修复后，共享 GPU realizer 先完成 contraction/reduction/scan 等结构角色，再只提升仍为纯 lane 的 pointwise domain。RTX 5090 从约 0.0530 恢复到 0.0368 ms，H100 从约 0.063 恢复到约 0.049–0.052 ms；真正需要二维 ownership 的 embedding 不退化。

### 6.3 回归二：cuTile gather 拼写的赢家随设备变化

第二轮的 masked gather 在 5090 更快，但 H100 paged attention 从约 0.63 退到约 0.81 ms；旧的 gather+where 在 H100 更快。两种写法的 indices、validity、fill、ownership 与 tile 完全相同，差异只属于 cuTile 目标 API 的等价拼写。

最终实现将它变成 cuTile leaf 内的 `GATHER_SPELLING` 编译期候选：

- `1`：masked gather；
- `0`：gather + where；
- 由 cuTile 自己的 `exhaustive_search` 与 tile/occupancy 一起实测；
- 不进入共享 Plan，不按 sm90/sm120 分支，Triton/TileLang 看不到该概念。

结果：5090 paged attention 选择 masked gather，约 0.326 ms；H100 选择 gather+where，约 0.632 ms。Grouped GEMM 也同步改善：5090 base/tail 收到约 1.340/1.348 ms，H100 收到约 1.192/1.352 ms。

这是真正证明“同一个静态规则救不了两台机器”的实例，因此它进入 target-local search，而不是让 shared realizer 学 GPU 型号。

### 6.4 全量结果与明确边界

修复后的全量状态为：

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| RTX 5090 | 120 pass / 1 unsupported / 1 compile-failed | 120 pass / 2 unsupported | 107 pass / 11 unsupported / 1 compile-failed / 3 failed |
| H100 | 120 pass / 1 unsupported / 1 compile-failed | 118 pass / 3 unsupported / 1 compile-timeout | 111 pass / 11 unsupported |

没有旧 pass 变成错误数值或能力拒绝。`token_sparse_mla_prefill` 的 Triton 状态被进一步定位为确定的单 tensor 编译表示上限，不再用 timeout 模糊；H100 cuTile 仍是候选编译成本。

同一阶段还用 `c10b268` 修掉 cuTile emitter 的局部变量名遮蔽，恢复 C++ 构建；它没有改变 IR、Plan 或生成语义。

## 7. 第五轮：持续对照 baseline 强化通用性能能力

### 7.1 目标与纪律

这一轮不新增 kernel、不修改 DSL 算法，专门并排阅读 generated 与 upstream source，优先处理同算法、同 scope 下的结构性差距。任何保留的改动必须由可复用的物理结构或 target 原生投影解释，不能按 kernel 名称固定参数。

### 7.2 Triton row-vector launch

SwiGLU、softmax、value-select 等 row/pointwise kernel 的生成结构与上游接近，但原 wrapper 对不同 row width 使用近似固定的 warp 配置。第一版把 row launch 放入 runtime tuner；真实运行又暴露出短 kernel 的编译缓存与候选选择会让 `reshape_and_cache` 从约 0.0207 退到约 0.030 ms。

最终规则仍是 target-local 的物理 launch 选择，但适用范围被严格写清：

- 无 compiler-private stage、无 reused program axes；
- 无 atomic/scatter-reduce 等非幂等 effect；
- 排除 scan、reduction、contraction、state-stream 等主体真正依赖 region 结构的情况；
- 仅普通 row-vector，或纯 pointwise、无聚合的 no-read body；
- warp 数只由已选 physical row extent 的 next-power-of-two 得到：小于 2048 用 4，2048 起 8，8192 起 16，32768 起 32。

这不是把 scalar body升级成 region 算法；作者 body 不变，改变的是一个 Triton program 的 worker 配置。两台机器各 23 个直接消费者均完成数值对照。关键结果：

- 5090 softmax：0.3523 ms，对应 source 0.3661 ms；
- H100 softmax：0.1820 ms，对应 source 0.1886 ms；
- 5090 SwiGLU forward：0.2329→0.2265 ms，对应 source 0.2273 ms；
- H100 SwiGLU forward：0.1455→0.1241 ms，对应 source 0.1210 ms；
- value-select：5090 0.0887→0.0800 ms，H100 0.0726→0.0607 ms。

`sorted_nucleus_cutoff` 与 boolean reduction 在第一版误入 row rule 后退化，最终通过结构判据排除并恢复到 5090 0.0200/0.0056 ms、H100 0.0176/0.0096 ms。这个过程说明 row width 不是全局门槛，必须先尊重 op 的结构角色。

### 7.3 cuTile row occupancy

cuTile 对同一 row program 可以由下层选择不同 occupancy；该值不改变算法、ownership 或 tile，只是 cuTile launch/compiler 参数。因此 emitter 只在结构上适用的 row path 暴露 occupancy 候选 `(default, 1, 2, 4)`，交给 cuTile tuner 实测，cache key 包含物理维度、输入 dtype 与 device。

两台机器同一批 23 个 row consumers 都通过数值对照。Softmax 的结果为：

- 5090：0.3543 ms，对应 cuTile source 0.3690 ms；
- H100：0.3287→0.1925 ms，对应 cuTile source 0.1926 ms。

这项变化没有进入共享 Plan，因为 occupancy 是下层 cuTile 的实现参数，不是 Intent 独有算法结构才能决定的事实。

### 7.4 cuTile batched contraction 的转置投影

四种 batched GEMM layout 的 Plan 已经完整说明哪个 operand transposed；旧 leaf 先 reshape 再 `ct.transpose`，下层得到的布局质量不稳定。cuTile 原生可以先对三维 tile 使用 `ct.permute` 交换最后两轴，再 reshape 到 MMA operand。这个修改只改变 target spelling，不新增 Plan 字段。

当前结果：

| 设备 | NN | TN | NT | TT |
|---|---:|---:|---:|---:|
| RTX 5090 generated | 0.0917 | 0.0922 | 0.0922 | 0.0938 ms |
| RTX 5090 source | 0.0922 | 0.0922 | 0.0922 | 0.0918 ms |
| H100 generated | 0.0480 | 0.0407 | 0.0489 | 0.0485 ms |
| H100 source | 0.0531 | 0.0540 | 0.0539 | 0.0535 ms |

H100 四种布局全部超过固定 source；5090 只有 TT 仍有约 2.2% 的小差距。

### 7.5 被实测否决并完全撤回的方向

TileLang pointwise single-use expression forwarding 被分别用于 W4A8 与 varlen attention。它没有改善 W4A8，且使 varlen attention noncausal/causal 分别从约 0.497/0.351 退到 0.507/0.361 ms，因此工作树中的这项改动全部撤回。

这两条差距的根因并不在临时表达式数量：

- W4A8 的 packed index `k // 2` 在 shared facts 中被压成普通 data-dependent indexing，Plan 没有保留“单一来源轴 + 正常数除数 + compact physical span”这类 typed coverage；TileLang 只能逐 logical-K 元素加载。正确方向是共享的受限 quasi-affine compact-coverage fact，而不是 W4A8 特判。
- varlen/GQA attention 的算法与上游都是 ragged online attention。差距来自 ordered-ragged stream 的 query/K tile、visible-prefix、mask-contraction fusion boundary 与 target layout 没有形成一个完整的联合物理决定；不是把某条算术 inline 就能解决。

## 8. 当前固定表的真实状态

两张 CSV 仍各有 122 条记录、113 个 kernel runner，20 列结构与 source 数字均未改变。最近一轮只刷新受影响 generated p50/p95。

### 8.1 状态计数

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| RTX 5090 | 120 pass / 1 compile-failed / 1 unsupported | 120 pass / 2 unsupported | 107 pass / 11 unsupported / 3 failed / 1 compile-failed |
| H100 | 120 pass / 1 compile-failed / 1 unsupported | 118 pass / 3 unsupported / 1 compile-timeout | 111 pass / 11 unsupported |

### 8.2 当前赢家分布

按每条记录中通过 provider 的最小 generated p50 计，并对完全相同的并列值均分：

| 设备 | Triton | cuTile | TileLang | 至少一个 provider 通过 |
|---|---:|---:|---:|---:|
| RTX 5090 | 53.17 | 33.17 | 35.67 | 122 |
| H100 | 56 | 30 | 35 | 121 |

三家在两台机器上仍分别拥有独立赢家，且赢家分布随设备改变。这个结果支持“同一 GPU Plan 的多个 surface 让下层各自发挥”，但不等于三家所有格都同样成熟。

### 8.3 还没有追平的格子必须分性质看

**已定位的通用 compiler/Plan 缺口：**

- TileLang W4A8 packed：5090 0.1612/0.0895 ms，H100 0.2356/0.1579 ms；缺 shared typed compact quasi-affine coverage，不是 leaf 名字分支能修。
- TileLang varlen attention：5090 causal 0.3512/0.2535 ms、GQA 9.7634/6.3852 ms；H100 causal 0.4700/0.1695 ms、GQA 9.4668/4.7898 ms；缺 ordered-ragged stream 的联合物理决定与对应投影。
- H100 Triton dense attention：3.5791/3.1012 ms；5090 基本持平，说明候选/descriptor/warp-specialized 下层 realization 仍需从结构上继续核对。

**不是公平的“同一实现更慢”：**

- `online_softmax` DSL 明确写了两个 state-stream pass，上游 softmax 是单次 load/reduce/store；数字可以观察，但不能把差距冒充同算法 leaf 回归。
- cuTile `block_scaled_matmul` 的 DSL 是 E8M0 scale 解码、f32 pointwise scale 后普通 contraction，上游使用原生 `ct.mma_scaled`；直接在 leaf pattern-match 并折叠会改写 Kernel IR。只有未来把 scaled contraction 作为正式算法 primitive，或下层自己完成融合，这个比值才可公平追。
- Cross entropy、部分 grouped/ragged 条目使用 end-to-end 或不同调用编排；必须按各自 scope 解释，不能和 kernel-only 混在一起。

**较小但仍保留在固定表中的差距：**

- 5090 cuTile batched GEMM TT 约慢 2.2%；
- cuTile inverse RoPE 5090 约慢 6.3%，H100 约慢 6.8%；
- cuTile MLA prefill 在两台机器分别约慢 4.9% 和 28.2%；
- H100 cuTile SwiGLU 与若干 provider-specific entry 仍有一成左右差距。

因此，最近一轮确实把 row launch、occupancy 和 transposed-load 三类通用能力推到或超过对应 baseline，但“所有有 baseline 的格子都已不弱于 baseline”还没有完成。剩余项已经从“零散慢值”缩成上述几类可定位问题，不能通过调数字、改测试 scope 或按 kernel 特判宣布结束。

## 9. 当前编译器到底完成了什么

### 9.1 已经闭合的核心

1. Python 只做 source frontend 和 lowering 临时状态，直接构造唯一 canonical Intent Kernel MLIR；没有并行 typed Python Kernel IR。
2. Kernel IR 保存算法、logical region、index/effect、structured primitive、typed closure 与 source callable 合同。
3. Shared facts 只保存可重算的 provenance/use-def/legality；Physical Plan 只保存多个合法物理答案中实际选中的一个。
4. GPU realizer 按逻辑轴组合 parallel、lane、ordered、reduction、contraction、ragged member 等角色，不按 kernel 类别选择入口。
5. Triton/cuTile/TileLang 共用 Kernel IR 与 GPU Plan；leaf 只做 capability、逐 op/概念投影、目标内候选和 runtime 接线。
6. Generic reduce/scan closure、多阶段 execution contract、tail validity、tensor-indexed load/store、复杂 ragged/stream/contraction 组合都有真实 kernel 和双设备数值证据。
7. 122 条记录覆盖逐元素、归约、dense/ragged/stream attention、GEMM、MoE、卷积、扫描、排序、动态规划、量化、反向、multi-stage、in-place 与等价分解。

### 9.2 明确但不伪装成完成的边界

- CPU、RISC-V、RVV target family 尚未接入；当前只能证明 Kernel IR 没把 GPU tile 身份写死，不能证明未来 realizer 已经存在或高性能。
- `partition(count=P)` 有 source 语义但当前 realizer 未实现，frontend fail-closed。
- TileLang 0.1.13 的 generic combiner、CAS、二维联合 checked footprint、若干单行/多轴 contraction 形态没有可机械委托的现行 surface 能力，提前 unsupported。
- Sparse 2:4、E8M0 与 FP8 能力是 provider × device 的真实子集，不被提升成全语言统一假能力。
- 下层编译表示上限、候选超时、NVCC crash 与 layout/resource failure 分别保留其真实状态，不统一包装成 unsupported。

## 10. 当前结论

从第一轮到现在，推进不是简单增加更多算子，而是依次完成了：边界复审与语义收拢、target spelling 解耦、陌生 vendor 算法施压、双设备全量回归、以及面向真实 baseline 的通用性能强化。

当前 GPU 主线已经是一套结构完整、边界显式、能从同一算法和同一 Plan 投影到三个 provider 并真实运行的单算子编译器。它已经具备成熟 compiler 的主要骨架，但性能收尾仍未全部完成：W4A8 compact coverage、ordered-ragged attention realization、H100 dense attention 与若干较小 provider 差距仍需按已定位的共享机制继续推进。准确的状态不是“只剩测试”，也不是“全部 benchmark 已追平”，而是 **correctness 与架构闭环已经建立，当前剩余主线集中在少数可解释的 realization/emission 质量缺口和尚未接入的 CPU/RISC-V/RVV target family**。
