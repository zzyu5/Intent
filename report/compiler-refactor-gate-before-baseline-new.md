# 继续新 baseline 前的编译器重构门槛

## 1. 本报告的目的

这份报告记录一次暂停后的只读审计。它不提出一个新的编译器版本，也不把旧 baseline 与新 baseline 当成编译器版本。

需要同时纠正两个问题：

1. 新 baseline 的接线工作不应成为随意修改编程模型、Kernel IR、Physical Plan 或 target emission 的授权。
2. source inventory 已经有 Triton 41、cuTile 39、TileLang 39 个可独立运行的高性能入口；每家 30 个只是新表的最低要求，不是最终规模。

因此，后续顺序必须调整为：

> 先审计并重构编译器与编程模型，关闭这几轮可能越界或混杂的修改；确认旧能力没有被破坏；然后再继续建立新 baseline。

本报告只登记事实、风险、边界和验收门槛。它不修改 `doc/`，不修改代码，也不把未验证判断写成正式语义。

## 2. 术语必须先纠正

仓库中所谓 baseline-v1 / baseline-v2，只表示两套性能对照口径：

- 旧 baseline：`report/baseline/` 中已经冻结的 RTX 5090D 与 H100 矩阵；
- 新 baseline：计划放在 `report/baseline-new/` 中的六张 provider × device 表。

它们不是 Intent 编译器的版本。建立新 baseline 不意味着：

- 编程模型可以重新设计；
- Kernel IR 可以按新 source 临时扩张；
- Realizer 可以为新 kernel 增加结构分支；
- emitter 可以绕过 Physical Plan 重建算法或物理决定；
- 旧 baseline 已经证明的能力可以被忽略并重新实现一遍。

`report/compiler-space-v1.md` 文件名中的 `v1` 只能理解为第一份架构审计稿，不能理解为编译器版本。后续报告和讨论应使用“当前编译器”“旧 baseline”“新 baseline”，不再用版本词混淆对象。

## 3. 旧 baseline 到底复用了多少

新 baseline 没有重新建立一条编译链。以下核心部分一直被复用：

- `examples/kernels/` 中同一套目标无关 DSL 算法；
- Python frontend 与 canonical Kernel MLIR；
- GPU Realizer 与 Physical Plan；
- Triton、cuTile、TileLang 三个 target leaf；
- `CompiledArtifact`、target source materialization 和 runtime launcher；
- `examples/repro/common/support.py` 中的 CUDA Event、CUDA Graph、L2 flush 与预分配 output launcher；
- 旧 source runtime 中仍符合新 source 合同的入口。

新建的是 baseline 接线层：

- `examples/repro/v2/registry.py`；
- `examples/repro/v2/providers/` 中按 provider 组织的 ABI adapter；
- 多输出结果树比较；
- 统一的 CSV writer；
- `examples/run/baseline-v2.sh`。

单独建立接线层有现实原因。旧 runner 由大型 shell/provider dispatch 组成，同时混有：

- generated-only 行；
- variant 行；
- 不同质量的 source；
- kernel-only、end-to-end、runtime-metadata 等不同计时 scope；
- provider 与 source callable 的硬编码映射。

新 baseline 要求同 provider、同算法、同输入输出、同调用次数、同计时 scope，所以需要一份更严格的 entry inventory 和 adapter。

但是，这只能解释 runner/adapter 为什么新增，不能解释共享编译器为什么要随着每个 adapter 失败继续扩张。baseline adapter 的失败首先应当被视为接线、算法对齐或已有能力复用问题。

## 4. source 数量与“30 项”错误

### 4.1 当前权威 inventory

三个 source README 记录的是 runtime-visible、可独立调用和计时的高性能入口：

| Provider | source inventory | 当前新 registry | 未进入 registry |
|---|---:|---:|---:|
| Triton | 41 | 30 | 11 |
| cuTile | 39 | 30 | 9 |
| TileLang | 39 | 30 | 9 |
| 合计 | 119 | 90 | 29 |

这些 source runtime 路径当前都实际存在。README 是完整 inventory，不是只有源码、没有运行入口的候选占位。

`report/baseline-new/construction.md` 已经明确：每张表至少 30 个 entry，30 只是下限，不是停止扩充现代模型路径的上限。当前 registry 却机械地固定为每家 30 个，而且没有记录 29 个未选项的排除理由。

### 4.2 未选项的真实性质

未选的 29 项中，部分可能是同算法不同 provider source 或实现变体；但大部分没有证据证明应排除，只能确认“尚未建立 DSL/adapter entry”。典型遗漏包括：

- Triton：fused linear cross entropy、split-K paged attention、causal Conv1D backward、两种 MoE expert projection、Mamba3 sequence forward；
- cuTile：dense attention、grouped flash decode、attention-sink decode、Gemma split-K decode、chunk gated delta、NVFP4 quantization；
- TileLang：persistent MLA、fused MoE、top-k selector、linear-attention backward、sparse MLA backward、BitNet 与 FP4 路径。

因此，当前 30 × 3 只能叫第一批 adapter scaffold，不能叫完整的新 baseline。

### 4.3 后续正确完成定义

新 baseline 不应以“每家正好 30 个”为目标，而应对完整 41/39/39 inventory 逐项分类：

1. 算法、调用边界和计时 scope 可对齐：进入 registry 和最终表；
2. 与已有 entry 完全重复：合并，并写清重复关系；
3. source 虽可运行，但和现有 DSL 是不同算法：新增一份真正对齐的 DSL，或明确暂缓；
4. 当前语言或 target 确实不能表达：记录明确能力边界；
5. 仅仅尚未做 adapter：不得伪装成“不需要处理”。

## 5. 最近编译器与 DSL 修改的审计

### 5.1 可以保留的既有能力修复

以下修改没有新增算法模型，主要是把已有 KIR/Plan 信息正确传给 leaf：

| Commit | 性质 | 当前判断 |
|---|---|---|
| `55978cc` | state-stream 已绑定 reduction range，但 leaf 又生成一套本地 reduction loop；同时修正 scalar gather | 通用 Plan 消费缺口，可保留，但需在重构后重验普通 contraction 与 stream-bound contraction |
| `01a2304` | routed staged contraction 的 RHS transpose、多维 unique store、constexpr shape dimension | 已有 canonical op 的 leaf/frontend 缺口，不是新算法，可保留并拆分职责 |
| `57e8d5f` | TileLang 把 traversal region argument 当普通 index | 明确目标投影 correctness 修复，不应放到 adapter 解决 |
| `f4c2c44` 中 ABI owner 部分 | cuTile wrapper 从尚未分配的 Out tensor 读取动态 shape | 明确 ABI 权威来源错误；Out 不能成为自己的动态 shape owner |

### 5.2 真正新增语言能力的修改

`76bb3f5` 新增 `I.sin` 与 `I.cos`，沿现有 canonical unary op 进入三个 target 的原生 spelling。RoPE 需要真实三角函数，旧 Core 无法在不改算法的情况下表达，因此这一项具备真实语言缺口依据。

即使如此，语言能力也必须独立提交、独立说明：

- 哪个真实算法无法表达；
- 为什么已有 Core 组合不成立；
- 三个 target 是否原生支持；
- 是否需要新 Plan 事实；
- 数值语义是否跨 target 一致。

### 5.3 `3a7fb23` 是当前最高风险项

该提交修改 25 个文件，净变化约 `+946/-96`，把以下不同职责混在一个提交中：

1. 新增通用 producer-chain contraction replay；
2. 修改 `SurfacePlan` 与 shared contraction/materialization 查询；
3. 修改 GPU Plan build/decision；
4. 同时修改三个 target emitter；
5. 传播 fixed partition extent；
6. 修改 Mamba 与 selective-scan 示例；
7. 新增 `I.floor` 语言 surface；
8. 改变部分 Mamba 数值表达与 domain 写法。

其中 ContractReplay 的核心不是 kernel-name matcher。它只接受受限 producer 白名单、单 reduction、两侧共享同一 reduction domain，并拒绝多个 contraction 争用同一 deferred transfer。这个方向可能是合理的通用 realization。

但它和以下内容并不构成一个不可分割的改动：

- fixed partition extent；
- `I.floor`；
- Mamba domain/bounds 重写；
- decay clamp 或 modulo 等数值路径变化。

尤其 `I.floor` 和数值表达改写没有在提交中给出独立的语言缺口与 reference 数值证据。它们不能因为 producer replay 需要而被顺带冻结。

重构前必须把这份提交按职责重新审计：

- A：ContractReplay 的 common analysis；
- B：Plan 中需要记录的 selected/derived facts；
- C：三个 leaf 的机械 replay；
- D：fixed extent 传播；
- E：Mamba/selective-scan 算法源码改动；
- F：`I.floor` 语言能力。

每一部分单独决定保留、重写或回退。不能继续以一个大提交整体继承。

### 5.4 不能把尚未定位的问题倒推给某个提交

之前曾怀疑 recurrent gated delta 数值失败由 producer replay 引入。当前代码审计显示 replay eligibility 明确排除了包含同一 reduction domain 的 `state_stream` contraction，因此没有 bisect 与生成源码证据时，不能把该失败归因于 `3a7fb23`。

这条同样是后续纪律：

> 失败只能根据具体 KIR、Plan、目标源码或提交 A/B 定位，不能因为“最近改过附近代码”就建立因果关系。

## 6. `partition` 编程模型：什么能简化，什么不能删

### 6.1 当前语义

`partition` 不是单纯 tile hint。它改变作者 body 看见的对象：

- 没有 partition：body 看见单个 logical element；
- 有 partition：body 看见同一 logical domain 的一个连续 region。

region 可以直接参与：

- contract；
- reduction；
- scan/state stream；
- region mask；
- logical buffer/effect；
- tensor indexing 与 broadcast。

因此，编译器不能因为某个标量循环适合向量化，就自动插入 partition。纯逐点标量程序可以做 lane packing，但 body 的算法语义仍然是标量实例，不应被改写成作者可见 region 程序。

### 6.2 当前 surface 与实现不一致

当前设计同时描述三种 extent 来源和一种 count 模式：

| 形式 | 文档/前端状态 | 当前 Realizer 状态 |
|---|---|---|
| `extent=<fixed integer>` | 接受 | 支持 |
| `extent=I.auto(name)` | 接受 | 支持，由 Plan 选择具体 physical extent |
| `extent=<runtime scalar>` | 文档和 frontend 接受 | KernelFacts/Realizer 拒绝 |
| `count=P` | 类型与文档保留 | frontend 立即拒绝，examples 无真实使用 |

这意味着当前 public model 比可执行 Core 更宽，存在两项“看起来支持、实际不能 lowering”的假能力。

### 6.3 可行的简化方向

当前真正有真实 kernel 证据的 Core 可以先收敛为：

```text
partition(domain, extent=fixed | auto)
```

- partition 是否存在：作者算法决定；
- fixed extent：作者明确决定 region 最大逻辑长度；
- auto extent：作者保留 region 算法边界，把具体 physical extent 交给 Realizer；
- region identity 始终来自原 logical domain，不创造第二个 axis 真理。

`count=P` 与 runtime extent 不应继续保持“名义可用、深层失败”的状态。重构时只有两个诚实选择：

1. 有真实算法与完整 lowering 需要它，补齐语义和实现；
2. 当前没有真实需求，从 public Core 中收回，等真实算法再次逼出时按冻结门槛重新审查。

`count` 不能在后端静默改写成 extent。它暴露 part ordinal，并可能影响 partial buffer shape、host orchestration 和多 kernel 共享 identity，语义上不是 extent 的别名。

### 6.4 不能越过的边界

无论如何重构，编译器不能自动改变：

- partition 是否存在；
- body 是 element 还是 region；
- fixed extent；
- count/part identity；
- logical index relation；
- region 上的 contract/reduce/scan/state/effect；
- wrapper-visible partial ABI。

Realizer 只能在作者给出 `auto` 授权时选择 physical extent；目标 leaf 只能消费这个决定。

## 7. 继续新 baseline 前的重构顺序

### 阶段一：冻结 baseline 接线工作

在本报告所列问题关闭前：

- 不增加新 registry entry；
- 不继续跑六张完整表；
- 不因 source adapter 失败修改语言；
- 不把某个 source 的手写结构直接搬进共享编译器。

source inventory 与 runtime 保持不动，作为后续验证材料。

### 阶段二：按职责拆审最近改动

逐项建立下面的纵向证据：

```text
真实算法需求
  → DSL 是否已有表达
  → canonical KIR 是否保留作者语义
  → derived facts 是否可从 KIR 重算
  → Physical Plan 是否只保存已选物理决定
  → leaf 是否只做目标投影与 capability check
  → 目标源码与数值 repro
```

重点先处理 `3a7fb23`，然后复核 `55978cc`、`01a2304`、`57e8d5f` 与 `f4c2c44` 的职责落点。

### 阶段三：收敛编程模型

对每个 public construct 分类：

- Core：真实算法需要、canonical KIR 完整、至少一个真实 lowering 闭环；
- sugar：frontend 可无损展开到 Core，不进入第二语义体系；
- target capability subset：Core 合法，但具体 target 可明确拒绝；
- 未实现 proposal：不应伪装成 public 可用能力。

`partition` 是这一轮的首个审计对象，但不是唯一对象。`floor`、generic combine、stage execution、sparse contraction 等近期改动也需要用同一把尺子重新确认。

### 阶段四：重构 compiler passes 与唯一权威来源

重构目标不是为了目录更漂亮，而是让每一层只持有自己的真理：

- Kernel IR：作者算法与逻辑 dataflow；
- derived analysis：从 Kernel IR 唯一重算，不序列化第二份真理；
- Physical Plan：多个合法物理方案中已经选中的一个；
- target leaf：目标 API、语法、allocation/copy/sync/primitive composition 与 capability；
- provider tuner：不改变程序结构的参数候选；
- 下层 compiler：layout、寄存器、指令和机器流水。

每个 pass 必须说明输入事实、输出事实、唯一权威来源和拒绝条件。不得按 kernel 名、op 数量或完整形态 matcher 分支。

### 阶段五：先验证旧能力，再继续新 baseline

重构验证范围按事实消费者确定：

- 修改 stream binding：跑所有读取该 binding 的代表 kernel；
- 修改 staged contraction：跑 dense、routed、ragged、transpose orientation 与 tail；
- 修改 region/partition：跑 element、region contract、region reduction、state stream 与 mask；
- 修改 ABI output：跑单输出、多输出和多阶段 pipeline；
- 修改 leaf：同一 KIR 至少比较三个 target 的数值结果。

只保留人工可执行的 DSL → emit → GPU numerical repro，不新建测试目录或 fixture。重构完成节点再做两台机器全量。

### 阶段六：恢复新 baseline

恢复后不再以 30 为完成目标，而是从完整 41/39/39 inventory 出发：

- 能公平对齐的全部进入；
- 重复项明确合并；
- 不同算法分别保留；
- unsupported、编译成本和尚未实现分别说明；
- 六张表只记录完成数值对照和公平计时的 entry；
- source 与 generated 的算法、调用数、dtype、workspace 和 timed scope 必须一致。

## 8. 编程模型冻结后的修改门槛

重构完成后，新增语言构造必须同时满足：

1. 存在真实公开算法，无法用当前 Core 表达；
2. 这项语义不能作为 frontend sugar 无损展开；
3. 不能直接委托给 target 已有能力而无需进入 canonical KIR；
4. 跨目标语义可以定义清楚；
5. 有最小 DSL → KIR → Plan → target source → 数值运行证据；
6. 单独提交，不和算法、adapter、Plan 重构混在一起。

新增 Physical Plan 字段必须满足：

1. 存在多个合法物理答案；
2. 已经选择的是其中一个答案；
3. 不能从 Kernel IR 唯一重算；
4. target leaf 不应重新决定；
5. 至少两个 target 需要消费同一物理事实，或它明确属于 target-local Plan。

不满足门槛的需求应当被拒绝、留在 adapter、作为 derived query 重算，或交给下层，而不是扩语言与 Plan。

## 9. 当前停止线

当前不能宣称：

- 新 baseline 已经完成；
- 90 个 registry entry 都已经实际运行；
- source inventory 只有 30 × 3；
- 最近的编译器修改全部经过冻结门槛；
- `partition` 的所有文档语义都已经实现；
- producer-chain replay 与 Mamba 示例改写是一个不可拆分的能力。

当前可以确认：

- 旧 baseline 仍是有效的历史能力证据；
- 新 baseline 的 source inventory 已经达到 41/39/39；
- 新 runner 复用了原编译链与计时底座，但 adapter 层尚未全部兑现；
- 最近修改中既有真实 emitter/ABI correctness 修复，也有需要重新审计的语言与算法改写；
- 下一步应先完成编程模型与 pass 重构，再继续新 baseline，而不是继续边跑表边修改共享编译器。
