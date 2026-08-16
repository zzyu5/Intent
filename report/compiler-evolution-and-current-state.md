# 从边界复审到决策空间形式化：统一推进报告

## 1. 报告范围

本文统一梳理从用户提出“第一轮：收尾遗留 + 重新自查路径”开始，到本轮“形式化编译器决策空间 + 算法对齐审计”的完整推进过程。它替代这一阶段按轮次产生的零散汇报；固定性能数字仍只保存在：

- `report/baseline/kernel-performance.csv`：RTX 5090；
- `report/baseline/kernel-performance-h100.csv`：H100；
- `report/baseline/compiler-closure.md`：更早的 42-repro 历史快照，不再滚动更新。

本文回答四个问题：每轮要求解决什么、实际怎样修改编译器、每个物理决定为什么属于当前层、以及现有 upstream 数字是否真的与 DSL 算法可比。

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
| 正确性探针 | `2e958e4` | 收拢 domain provenance 与 arg-reduction index 语义 |
| Cross entropy 对齐 | `12b2e8e` | 将 DSL 改成与 Liger 相同的单 kernel fused 算法 |
| Triton row 参数委托 | `ae02092` | 删除 warp 阶梯常量，改由 Triton autotuner 实测 |
| 决策空间代码收敛 | `767c2e5` | 委托三家 row launch 参数并加入 canonical scaled contraction |
| Backward 与 baseline 对齐 | `ba06622` | 使用 forward-saved stats，并直接接上游 backward kernels |
| Scaled spelling 委托 | `31a776a` | Triton 原生/显式 scaled spelling 共同进入 target-local tuner |
| 决策审计收口 | `a0f7de7` | 收拢 domain extent 派生查询，并把 TileLang GEMM warp policy 交给 tuner |
| TileLang 参数空间 | `d82d1f4` | 补齐 contraction 的 K/stage 合法候选 |
| 候选保真 | `fc17060` | profile 只按当前 roles 覆盖度筛选，不再误删旧合法候选 |

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

SwiGLU、softmax、value-select 等 row/pointwise kernel 的生成结构与上游接近，但原 wrapper 把 warp 数钉成一张按 row width 分段的常量表。那张表虽然能跑，却违反“下层能实测的参数不由 Intent 猜”的边界：`num_warps`、`num_stages` 不改变 Kernel IR、grid 结构或 row ownership，只是 Triton launch 参数，属于参数性选择。

当前实现保留共享层给出的 row/persistent 结构，删除 warp 阶梯和 `row_configuration` 的固定 launch 值。Triton runtime tuner 现在联合实测 `num_warps`、`num_stages`、persistent row occupancy 与 pipeline stages；普通 row 的 autotune key 覆盖完整 shape，wrapper 的调优 guard key 另外带 ABI dtype 与 device，不能再把同一 lane width、不同外层 shape 的 winner 错误复用。

调优资格仍由语义安全性限制：含 compiler-private stage、atomic、scatter-reduce 等不可重放 effect 的 kernel 不进入会重复执行 body 的 autotune。这不是性能启发式，而是唯一合法的 replay 合同。作者 body 不变，改变的只是下层 launch 参数。关键结果：

- 5090 softmax：0.3523 ms，对应 source 0.3661 ms；
- H100 softmax：0.1820 ms，对应 source 0.1886 ms；
- 5090 SwiGLU forward：0.2329→0.2265 ms，对应 source 0.2273 ms；
- H100 SwiGLU forward：0.1455→0.1241 ms，对应 source 0.1210 ms；
- value-select：5090 0.0887→0.0800 ms，H100 0.0726→0.0607 ms。

`softmax_backward` 的最终 p50 为：5090 Triton/cuTile/TileLang `0.1347/0.1345/0.1306 ms`，H100 为 `0.0924/0.0782/0.1106 ms`。三个 target 的 winner 不同，说明 launch 参数必须委托给各自下层，而不是共享一张 GPU 型号无关的表。

### 7.3 cuTile row occupancy

cuTile 对同一 row program 可以由下层选择不同 occupancy；TileLang 可以选择 `threads` 与 `num_stages`。这些值不改变算法、ownership 或 tile，只是 provider launch/compiler 参数。因此两个 leaf 都只在可重放 row path 暴露候选，分别交给 cuTile 和 TileLang tuner 实测；cache key 包含完整 shape、输入 dtype 与 device。

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

## 8. 固定表与算法对齐审计

两张 CSV 仍各有 122 条记录、113 个 kernel runner，20 列结构不变。本轮没有全量重跑，只更新真实受影响的 generated 数字，并撤掉无法和 DSL 算法公平比较的 source 数字。空 source 表示没有可比高性能上游，不表示生成失败。

### 8.1 状态计数

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| RTX 5090 | 120 pass / 1 compile-failed / 1 unsupported | 120 pass / 2 unsupported | 107 pass / 11 unsupported / 3 failed / 1 compile-failed |
| H100 | 120 pass / 1 compile-failed / 1 unsupported | 118 pass / 3 unsupported / 1 compile-timeout | 111 pass / 11 unsupported |

### 8.2 当前赢家分布

按每条记录中通过 provider 的最小 generated p50 计，并对完全相同的并列值均分：

| 设备 | Triton | cuTile | TileLang | 至少一个 provider 通过 |
|---|---:|---:|---:|---:|
| RTX 5090 | 54.17 | 31.17 | 36.67 | 122 |
| H100 | 56 | 30 | 35 | 121 |

三家在两台机器上仍分别拥有独立赢家，且赢家分布随设备改变。这个结果支持“同一 GPU Plan 的多个 surface 让下层各自发挥”，但不等于三家所有格都同样成熟。

### 8.3 审计判据

每个 source-bearing provider cell 只能落入三类：

- **A：算法、调用数与计时 scope 一致。** source 数字保留；差距归编译器或 provider。
- **B：算法本应一致，但 DSL 或 adapter 没有照上游写。** 先改 DSL/adapter，再重测；不能继续使用旧数字。
- **C：上游是另一算法或另一调用编排，且不该把 DSL 改成它。** 撤掉 source 数字，不制造“近似可比”。

### 8.4 所有保留的 source cells

下表覆盖更新后 CSV 中每一个非空 source cell；同一行列出的 provider 分别是 Triton/cuTile/TileLang。

| kernel / case | provider | 结论 | 对齐依据 |
|---|---|---|---|
| softmax | Triton、cuTile | A | 单次 row load/reduce/store，kernel-only |
| layer_norm | Triton、cuTile | A | 同一 forward normalization 与 affine |
| layer_norm_backward | Triton | B→A | 现在直接调用上游两个 backward kernel，forward-saved mean/rstd 在计时外 |
| rms_norm | Triton | A | 同一 weighted RMSNorm kernel |
| fused_add_rms_norm | Triton | A | 同一 fused residual + RMSNorm |
| cross_entropy | Triton | B→A | DSL 改成单 kernel fused loss/prediction/in-place gradient，与 Liger 同 scope |
| gemm/base | 三家 | A | 同形状、dtype、单 contraction |
| bf16_gemm | cuTile、TileLang | A | 同一 bf16 GEMM |
| batched_gemm NN/TN/NT/TT | cuTile | A | batch、四种 transpose 与调用数一致 |
| attention | 三家 | A | dense forward attention，同一 QKV/O scope |
| attention_bias | Triton | A | 同一 fused bias attention |
| varlen_attention/causal | TileLang | A | 同一 causal varlen online attention |
| varlen_gqa_prefill | TileLang | A | 同一 varlen GQA prefill |
| paged_attention | Triton | A | 同一 paged decode attention |
| online_softmax | TileLang | A | 两边都是 online/two-pass 算法，而不是普通 stable softmax |
| grouped_gemm/base | 三家 | A | grouped problem 与端到端 list/result scope 一致 |
| swiglu_forward | Triton、cuTile | A | 同一 fused SiLU×up |
| swiglu_backward | Triton | A | 同一 backward kernel |
| rope_qk_full/partial/inverse | cuTile | A | Q/K、旋转维度与方向逐 case 一致 |
| mla_prefill | cuTile | A | 同一 MLA prefill contraction 结构 |
| w4a8_packed | TileLang | A | 同一 signed packed-W4/A8 dequant GEMM |
| embedding_forward_lookup | Triton | A | 同一 embedding gather |
| embedding_backward_atomic | Triton | A | 同一 atomic embedding gradient；只有有实测值的设备填 source |
| block_scaled_matmul | cuTile | B→A | 新 Core 直接表达 E8M0 scaled contraction，投影为 `ct.mma_scaled` |
| splitk_attention_reduce | cuTile | A | 同一 split-K partial reduction |
| fp8_gemm e4m3/e5m2 | TileLang | A | dtype 和单 GEMM scope 分别一致 |
| index_select_rows | Triton | A | 同一 row gather |
| scaled_index_add | Triton | A | 同一 indexed scaled add |
| sparse_2to4_gemm | TileLang | A | 同一 2:4 metadata contraction |

### 8.5 撤掉的对照

| kernel | provider | C 类原因 |
|---|---|---|
| softmax | TileLang | source 是 online/two-pass 结构，不是 stable single-pass 算法 |
| rms_norm | TileLang | source kernel 不含相同权重路径，adapter 还在 kernel 外乘权重 |
| dual_gemm | 三家 | source 是两个独立 GEMM 加外部 epilogue，DSL 是单 fused kernel |
| online_softmax | Triton、cuTile | source 是普通 stable softmax，DSL 明确选择 online/two-pass |
| moe | 三家 | source 是两个 grouped GEMM 加 merge，DSL 是一个 fused ragged FFN kernel |

这些 cell 仍保留 generated 数字，但 source 列为空。没有把某个 provider 的算法反向变成 DSL 的唯一写法。

## 9. 物理决策空间的正式合同

### 9.1 三类决定与判定顺序

分类对象是“多个候选里必须填一个值的物理决定”，不是所有 compiler facts。判定顺序固定为：

1. **下层 provider 能否只用它已经看见的 target source、shape、dtype 与设备信息实测这个选择？** 能，并且候选只改变参数值或等价 API 拼写，就是 P。
2. 不能时，问 **算法语义是否只允许一个答案**。若是，就是 U；错了会改变地址、边界、数值或依赖，属于 correctness。
3. 仍有多个合法答案，但答案会改变 grid、循环、ownership、驻留或 stage 等源码结构，而下层已经看不到 Intent 的 region/use-def/跨 op 结构，就是 S。

正式分类如下：

- **U — 唯一合法解。** 从 Kernel IR 与目标合同只有一个正确兑现结果，错了就会改变地址、边界、数值或依赖；必须在共享 lowering/Plan 形成一处权威来源，leaf 只读取。它和普通 derived fact 的区别是：derived fact 只描述 IR 中已经存在的关系，可随时重算；U 是编译器必须补齐的 correctness 合同。例：索引算术位宽、padding identity、已选 stage 的拓扑依赖与可见性。provenance 本身是 derived fact，不属于 U。
- **S — 结构性选择。** 多个答案都正确，但改变生成源码的结构，而且 provider 无法仅凭最终 target source 前的局部信息重建。由 target-family realizer 选择并写入 Physical Plan。例：program ownership、persistent traversal、lane promotion、private buffer 驻留。
- **P — 参数性选择。** 多个答案都正确，不改变 Kernel IR 和 Physical Plan 的结构，只改变 tile/launch 数值或等价 target API 拼写；交给对应 provider tuner 实测。Intent 只声明合法候选。

一条容易混淆但必须保留的边界是：**作者显式写 `partition` 不是 S，而是算法程序结构。** 它决定 body 看见 scalar 还是 region，编译器无权删除或补造。Plan 里选择这个 region 在机器上用多大 physical extent，才是 S/P 边界里的物理工作。

### 9.2 不属于 U/S/P 的四类内容

以下内容不是“决定”，强塞进三类反而会制造假问题：

1. 可从 Kernel IR 重算的 provenance、use-def、result axes 等派生 facts；它们可以做内存索引，但不能成为第二份 schema 真理。
2. target capability 与 legality，例如 cuTile sm90 不支持 E8M0、TileLang 0.1.13 不接 generic combiner、descriptor 地址上限。
3. canonical op 到目标 intrinsic 名称、dtype spelling、参数顺序等纯语法映射。
4. benchmark warmup、timeout、CUDA graph、cache flush 等测量策略。

本轮没有发现用上述判据无法分类的真实物理决定。真正需要补充的是：P 不仅包括数字参数，也包括 **语义与 Plan 完全相同、只在 target 里有不同等价拼写** 的离散候选。

### 9.3 全量物理决定审计

| 决定或事实 | 分类 | 当前权威层 | 审计结论 |
|---|---|---|---|
| logical index provenance、地址表达式、domain extent、use-def/result axes | derived fact | Kernel IR + shared facts/query | 不属于 U/S/P；按 SSA/use-def 重算，找不到精确来源直接诊断，不按同 extent 猜轴。本轮把三个 leaf 重复的 domain→ABI extent 回溯收拢为一个共享查询，但没有把结果物化进 Plan |
| 地址算术宽度 | U + target capability | shared correctness contract + leaf guard | 语义地址先按 i64；Triton 直接投影 i64，cuTile/TileLang 当前 surface 在 launch 前证明 element offset 不超过 `2^31-1` 后才使用其 32-bit descriptor/bulk-copy 表达，超出就明确 unsupported，不窄化回绕 |
| region argument → logical axis/purpose | derived fact | Kernel IR + shared facts | 不属于 U/S/P；这是作者 region 语义的稳定绑定，不是物理候选 |
| 已选 physical range → region argument/use 的绑定 | U | Physical Plan binding | 正确；range 一旦选定，消费关系只有一个合法答案，leaf 不从 tensor shape 重建 |
| affine access footprint → transfer/source-axis 的 access range | U | Physical Plan access binding | 正确；每个 footprint 精确绑定已有 ownership range，不由 leaf 重猜 |
| validity、reduction identity、padding、tail fill | U | shared proof + Plan padding/transfer | 正确；三个 leaf 只兑现 mask/fill |
| `BlockExtent(power_of_two, zero)` 与 full-domain row/lane 最小合法 extent | U | shared GPU legalization | 不是 tuner tile：作者 body 已要求看见完整逻辑域；GPU fragment 合同唯一补成能覆盖它的最小 2 次幂，并以 consumer identity 中和尾部 |
| stage dependency、输入输出归属、same-stream visibility | U | Physical Plan stage contract | 正确；不是 emitter launch 顺序的隐含偶然 |
| parallel/ordered/reduction/contraction/ragged 等逻辑角色 | derived fact | Kernel IR + shared facts | 不属于 U/S/P；角色由 op、region 和 use-def 唯一推出，而且可以组合 |
| 逻辑角色到 lane/program/worker/stream 等物理用途的分配 | S | shared GPU realizer | 正确；这是多个合法机器映射中的结构性选择，不按 kernel 类别分支 |
| indirect-ragged/serial traversal 是否标量化为 `one` | S | shared GPU realizer | 当前是保守结构选择；它改变循环与搬运形态，不冒充可调 tile 数值 |
| program order、worker folding、program-group membership、reuse-worker | S | shared GPU realizer | 正确；provider tuner 不重新决定 ownership。Plan 中 `group_m` 先记录哪些轴组成一组 |
| program-group 的具体 group width | P | provider runtime tuner | `GROUP_SIZE_M` 等只改变同一 grouped traversal 的数值；`AxisOp.group` 的组成员是 S，SearchSpace 中同名 parameter 是该组宽度，二者不能混为一类 |
| persistent traversal 是否存在 | S | shared GPU realizer | 正确；改变循环/grid 结构，不能下放成普通 launch 参数 |
| pointwise lane promotion eligibility | S | shared GPU realizer | 正确；只对纯逐元素 body，不能自动把 contraction 改成块算法 |
| logical buffer、transfer result、contraction operand、scan result 的驻留 | S | shared GPU Plan builder | 由 owner、lifetime、reuse 与消费结构选择 private scalar/fragment/shared/workspace；leaf 只拼写。Sparse 2:4 的 shared operands 还受当前 target capability 约束 |
| ownership/traversal/reduction 的可分块 concrete tile、scan chunk | P | 三个 runtime tuner | 正确；结构用途已经由 Plan 确定，SearchSpace 只给合法参数轴，不钉死 winner；`fixed_*`、`one`、`row_vector*` 明确不进入这张搜索表 |
| `num_warps`、`num_stages`、TileLang threads、cuTile occupancy/num_ctas | P | provider runtime tuner | 本轮收敛；删除 Triton warp 阶梯与固定 row winner。不能安全 replay 的 effectful kernel 不做 wrapper autotune，这是 runtime legality，不是一个被写死的 Plan winner |
| TileLang `GemmWarpPolicy` | P | TileLang runtime tuner | 本轮从固定 `FullRow` 改为 FullRow/Square/FullCol 离散候选；target compiler 负责过滤当前 tile/device 无法编译的组合，没有架构分支 |
| cuTile masked-gather vs gather+where | P | cuTile leaf tuner | 正确；两台机器实测 winner 不同，不进共享 Plan |
| Triton `tl.dot_scaled` vs 显式 E8M0 decode + `tl.dot` | P | Triton leaf tuner | 同一 scaled-contract op、同一 Plan，由 target tuner 实测 |
| matrix-unit 可用性、worker 轴最多三维、cuTile 32-bit descriptor、Triton scaled group=32 | capability/legality | target-family/leaf | 不属于 P；不支持时在编译或 launch 前明确拒绝。Triton 当前 native/fallback scaled path 只接受 K-group 32，这是 target 子集，不是 shared Core 限制 |
| canonical op kind→`tl.*`/`ct.*`/`T.*`、dtype spelling、参数顺序 | target syntax | 各 `Syntax/Spelling`/op handler | 不属于 U/S/P；leaf 读取 canonical op 选择目标 intrinsic 正是机械投影，不能为消除一次 op-name lookup 而把算法 kind 复制进 Plan |
| row-tuner eligibility、warmup/repetition/timeout/cache key | runtime policy + capability | shared effect query + provider runtime | 不属于物理决定；三家可以因 replay/API 能力而有不同 eligibility，但 cache key 必须覆盖影响 winner 的完整 shape/dtype/device |

审计了共享 realizer 中所有规则、三个 leaf 的常量/阈值和 runtime candidate 表。固定数字现在只承担四种角色：U 的唯一兑现、target capability 上限、S policy 的结构阈值、或 P 搜索空间中的候选值；候选中出现 `8` 不代表 winner 被固定成 `8`。本轮没有发现无法用“derived/capability/syntax/runtime 与 U/S/P”这套顺序归类的残余决定。

### 9.4 本轮实际纠正的错位

1. Triton row `num_warps` 的 `2048/8192/32768` 阶梯、固定 winner `num_warps=8` 和写死的 shared-memory 二选一 stage 被删除；`8` 仍可作为合法候选，最终由 Triton tuner 联合测量 launch 参数后选择。
2. cuTile row occupancy 与 TileLang row threads/stages 统一进入各自 tuner；不可重放 effect 使用一份共享 derived check。
3. Triton 普通 row wrapper 的 cache key 从单 lane extent 扩成完整动态 shape；手工 guard key 另外带 ABI dtype/device。Triton 原生 `@autotune` 的 key 仍来自 SearchSpace，ABI dtype 与 device 在同一 generated module 内固定，二者没有被混写成一套 key。
4. `scaled_contract` 的 target spelling 被证明也是 P：5090 更偏向 native scaled dot，H100 的候选空间必须同时包含单 scale-group 形态。没有写 `if sm90`/`if sm120`。
5. H100 cuTile 的 E8M0 失败被收敛为 capability：当前 API 在 sm90 不能表示该 dtype，launch 在 tuner 前直接 `NotImplementedError`，不再让所有候选逐个失败。
6. 三个 leaf 各自递归解析 ragged/domain extent 的 120 余行重复逻辑被删除，改读一份共享 `logicalDomainExtent` 派生查询。它仍从 Kernel IR 与 ABI shape 现算，不进入 Plan schema。
7. TileLang contraction 的 `GemmWarpPolicy.FullRow` 不再是 leaf 常量；FullRow/Square/FullCol 与 tile、stage、threads 一起由 TileLang tuner 实测。5090 的 dense GEMM 与 sparse GEMM 都选择过非旧默认的赢家，证明它不是纯语法别名。
8. 扩大 TileLang contraction profile 后，最初的 selector 因 profile 含一个当前 kernel 未消费的字段而误删旧候选，造成 H100 sparse GEMM `0.1807→0.2205 ms`。selector 现在只比较当前 roles 的覆盖度；旧候选和新候选都保留，最终恢复到 `0.1815 ms`。这是候选集合 correctness，不是按设备回撤。

### 9.5 Canonical scaled contraction 为什么是 Core，而不是 pattern

旧 `block_scaled_matmul` 把 E8M0 decode、逐元素 scale 和普通 contraction 手写在 DSL 里，而上游 cuTile 调用原生 `ct.mma_scaled`。把这串普通 op 在 leaf 里 pattern-match 成 MMA 会改写作者 Kernel IR，违反编译器边界；继续挂 baseline 又不可比。

本轮确认 Triton `tl.dot_scaled` 与 cuTile `ct.mma_scaled` 都把“带显式 scale tensor 的 contraction”作为原生算法角色，因此增加 canonical `intent.scaled_contract`。它只保存算法语义：四个 data/scale operands、FP8/E8M0 format、两侧 scale group、reduction pair 与 accumulator dtype；没有 warp、tile、layout、storage 或 provider 字段。普通和 scaled contraction 共用 facts、轴角色、padding proof 与 `intent_plan.contract`；三 leaf 逐 op 投影：

- Triton：`tl.dot_scaled` 与显式 decode+`tl.dot` 都作为 P 候选；
- cuTile：支持设备上投影 `ct.mma_scaled`；sm90 的 E8M0 capability 明确拒绝；
- TileLang 0.1.13：没有同级 scaled MMA surface，机械 decode 后调用 `T.gemm`，不向共享层索要 TileLang 专属字段。

定向结果：

| 设备 | Triton | cuTile | TileLang | cuTile source |
|---|---:|---:|---:|---:|
| RTX 5090 | 0.0118 ms | 0.0139 ms | 0.0258 ms | 0.1390 ms |
| H100 | 0.0169 ms | capability unsupported | 0.0280 ms | 不可运行 |

H100 上旧显式算法同机 A/B 为 0.0170 ms，新候选为 0.0169 ms；固定表旧值 0.0164 ms 的差异来自当次环境漂移，不是新 Core 回退。

TileLang warp-policy 委托的定向结果如下；source 数字没有因 generated 侧调优而改动：

| device / case | 旧固定表 generated p50 | 本轮 p50 | 结果 |
|---|---:|---:|---|
| RTX 5090 GEMM/base | 2.1181 ms | 2.0475 ms | 数值通过，快 3.3% |
| RTX 5090 sparse 2:4 | 0.2309 ms | 0.2103 ms | 数值通过，快 8.9% |
| RTX 5090 block-scaled | 0.0279 ms | 0.0258 ms | 数值通过，快 7.5% |
| H100 GEMM/base | 1.7485 ms | 1.7468 ms | 数值通过，未退化 |
| H100 GEMM/tail | 1.8982 ms | 1.8126 ms | 数值通过，快 4.5% |
| H100 sparse 2:4 | 0.1814 ms | 0.1815 ms | 数值通过，0.1 μs 差异；旧 FullRow winner 仍在候选中 |

## 10. 本轮验证范围与真实性

没有做全量。实际执行的受影响 repro 包括：

- RTX 5090：`block_scaled_matmul` × 3、Triton `layer_norm_backward`、Triton `softmax_backward`，以及本轮前半已执行的 `cross_entropy` × 3、cuTile/TileLang LayerNorm backward；决策审计收口后又执行 TileLang `gemm`（base/tail/degenerate）、`block_scaled_matmul`、`sparse_2to4_gemm`；
- H100：`block_scaled_matmul` × 3（cuTile 确认 capability）、`cross_entropy` × 3、`layer_norm_backward` × 3、`softmax_backward` × 3；之后在隔离 worktree 用同一补丁执行 TileLang `gemm`（base/tail/degenerate）、`block_scaled_matmul`、`sparse_2to4_gemm`；
- H100 额外用旧显式 block-scaled 实现做同机 A/B，不使用旧 CSV 猜回归。

所有 pass 项都完成真实 GPU 数值对照。主要结果：

| kernel / device | generated | source | 结论 |
|---|---:|---:|---|
| cross entropy / 5090 Triton | 0.3432 | 0.3421 ms | 同算法，约 0.3% 差距 |
| cross entropy / H100 Triton | 0.1944 | 0.2170 ms | generated 快 10.4% |
| LayerNorm backward / 5090 Triton | 0.0826 | 0.0625 ms | 真正 compiler/physical-reduction 差距，约 1.32× |
| LayerNorm backward / H100 Triton | 0.0809 | 0.0858 ms | generated 快约 5.7% |
| block-scaled / 5090 cuTile | 0.0139 | 0.1390 ms | native scaled Core 明显优于 vendor 固定接线 |
| GEMM / 5090 TileLang | 2.0475 | 2.3229 ms | warp policy 与扩展参数族由下层实测，generated 快 11.9% |
| GEMM / H100 TileLang | 1.7468 | 2.1684 ms | 同一候选机制，generated 快 19.4% |
| sparse 2:4 / 5090 TileLang | 0.2103 | 0.2332 ms | generated 快 9.8% |
| sparse 2:4 / H100 TileLang | 0.1815 | 0.2622 ms | generated 快 30.8% |

5090 LayerNorm backward 的剩余差距已经不能再归因 adapter：两边都读取 forward-saved stats、执行 grouped partial reduction 与第二阶段归约。当前 generated 的 `scatter_reduce` 需要清零两个 f32 partial buffers；上游用 row-group lock 和 bf16 partial buffer。它是一个需要继续决定 collision/partial-storage realization 的通用编译器问题，不是本轮为了表格增加的 kernel 特例。

## 11. 当前编译器到底完成了什么

### 11.1 已经闭合的核心

1. Python 只做 source frontend 和 lowering 临时状态，直接构造唯一 canonical Intent Kernel MLIR；没有并行 typed Python Kernel IR。
2. Kernel IR 保存算法、logical region、index/effect、structured primitive、typed closure 与 source callable 合同。
3. Shared facts 只保存可重算的 provenance/use-def/legality；Physical Plan 只保存多个合法物理答案中实际选中的一个。
4. GPU realizer 按逻辑轴组合 parallel、lane、ordered、reduction、contraction、ragged member 等角色，不按 kernel 类别选择入口。
5. Triton/cuTile/TileLang 共用 Kernel IR 与 GPU Plan；leaf 只做 capability、逐 op/概念投影、目标内候选和 runtime 接线。
6. Generic reduce/scan closure、多阶段 execution contract、tail validity、tensor-indexed load/store、复杂 ragged/stream/contraction 组合都有真实 kernel 和双设备数值证据。
7. `intent.scaled_contract` 使 FP8/E8M0 scaled MMA 成为作者可直接陈述的 Core 算法角色；没有通过 leaf pattern-match 改写普通 op 链。
8. 122 条记录覆盖逐元素、归约、dense/ragged/stream attention、GEMM、MoE、卷积、扫描、排序、动态规划、量化、反向、multi-stage、in-place 与等价分解。

### 11.2 明确但不伪装成完成的边界

- CPU、RISC-V、RVV target family 尚未接入；当前只能证明 Kernel IR 没把 GPU tile 身份写死，不能证明未来 realizer 已经存在或高性能。
- `partition(count=P)` 有 source 语义但当前 realizer 未实现，frontend fail-closed。
- TileLang 0.1.13 的 generic combiner、CAS、二维联合 checked footprint、若干单行/多轴 contraction 形态没有可机械委托的现行 surface 能力，提前 unsupported。
- Sparse 2:4、E8M0 与 FP8 能力是 provider × device 的真实子集，不被提升成全语言统一假能力。
- 下层编译表示上限、候选超时、NVCC crash 与 layout/resource failure 分别保留其真实状态，不统一包装成 unsupported。
- 5090 Triton LayerNorm backward 的 grouped partial reduction 仍比直接上游约慢 1.32×；它是已对齐算法后的真实 realization 差距，不以环境或 scope 遮掩。

## 12. 当前结论

从第一轮到现在，推进不是简单增加更多算子，而是依次完成了：边界复审与语义收拢、target spelling 解耦、陌生 vendor 算法施压、双设备全量回归、面向真实 baseline 的通用性能强化，以及本轮对所有物理决定和所有 source cell 的正式归类。

当前 GPU 主线已经是一套结构完整、边界显式、能从同一算法和同一 Plan 投影到三个 provider 并真实运行的单算子编译器。三类决策合同给出了以后修改的硬边界：U 只能有一个来源，S 只能由看得见 Intent 结构的 realizer 选择，P 必须交给 provider 实测。算法审计也使 source 数字重新只表达可比较事实。

它已经具备成熟 compiler 的主要骨架，但性能收尾仍未全部完成：LayerNorm backward 的 grouped-reduction strategy、W4A8 compact coverage、ordered-ragged attention realization、H100 dense attention 与若干较小 provider 差距仍需按已定位的共享机制继续推进。准确状态是：**correctness、编程模型和决策归属已经闭环；当前剩余主线是少数有真实对照支撑的 realization/emission 质量缺口，以及尚未接入的 CPU/RISC-V/RVV target family。**
