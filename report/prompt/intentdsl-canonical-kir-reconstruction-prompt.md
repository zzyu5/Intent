# 第二轮：重构第一层编译器 IR——Canonical KIR

这一轮重构编译器的第一层：把第一轮已经冻结并迁移完成的作者 DSL 降到 canonical Kernel IR。依据是 `doc/` 中已经确定的编程模型、DSL 和编译器分层设计。先完整阅读这些最终规格和第一轮报告，再检查当前实现：

```text
doc/programming-model/
doc/dsl/
doc/compiler/
AGENTS.md
```

`doc/` 描述目标架构，不以当前代码为标准。当前实现与规格冲突时，应修改实现；但如果真实代码证据表明规格内部存在矛盾，必须停下来指出矛盾，不能靠兼容分支、默认值或弱化 verifier 把它绕过去。

第一轮已经负责public surface、规范示例与`examples/kernels/`源码迁移。本轮不能重新选择算法或修改DSL语义；若发现第一轮源码仍违反最终规格，先按规格修正该遗漏并说明，不能在frontend里为它创造第二种解释。

## 一、本轮边界

本轮只闭合：

```text
作者 DSL
  → frontend 解析与 desugaring
  → canonical Intent Kernel IR
  → KIR verifier 与 canonical analyses
```

Canonical KIR 是作者算法语义的唯一权威表示。它必须完整保存：

- kernel/helper function 边界、参数、结果和 ABI；
- tensor、scalar、record、view、logical buffer；
- domain、subregion 和 typed index relation；
- 普通结构化控制流及其 loop-carried values；
- tensor/value 计算；
- reduce、scan、region fold、region scan、contract、scaled contract、sparse contract、histogram；
- load、store、gather、scatter、atomic 等可观察 effect；
- RNG 的正式数值语义；
- alias、读写归属以及其他作者可观察语义。

它不能包含：

- program ID、grid、warp、lane 或 thread mapping；
- block/tile 数值；
- register/shared/private/global 等存储选择；
- target pointer、descriptor、copy form；
- physical mask、padding realization；
- pipeline、persistent traversal、MMA layout；
- Triton/cuTile/TileLang 专属属性；
- tuner candidate 或设备型号判断。

本轮不设计或完成 shared executable GPU IR，也不实现 GPU physical passes。KIR 后面的 Physical Program 是下一层任务。不要因为下一层目前不完整，就把 physical side records、旧 Plan 字段或 provider metadata 塞回 KIR。

## 二、先建立完整、typed 的 KIR 骨架

不要继续以 opaque string metadata、宽泛 `AnyType` 或“名字加 attributes”作为语义权威。横向梳理并实现最终 KIR 所需的类型和身份系统。

至少闭合：

- scalar 和 tensor element type；
- shape dimension 的静态值、动态身份及其等价关系；
- typed record 及字段投影；
- external view、logical buffer 及 alias/effect 信息；
- domain；
- source-derived subregion；
- typed index relation；
- function、kernel 和 helper symbol；
- 稳定的 value/domain/relation provenance。

动态维度不能因为都被打印成 `?` 就被视为同一个维度。相等必须来自同一语义来源或可证明的关系，而不是字符串、extent 数值或位置巧合。

source location 和 origin ID 只服务于：

- 诊断；
- provenance；
- 验证 lowering 没有丢失作者语义。

它们不能参与物理选择，也不能成为另一份 executable authority。

## 三、结构化控制流必须成为普通 KIR

根据最终 DSL 设计，把普通程序结构直接降成 typed KIR：

- `if`；
- `for`；
- `while`；
- 表达算法独立性的 parallel iteration；
- loop arguments；
- loop-carried scalar/tensor/record；
- `yield`、condition 和 return。

每个 region 的 block arguments、yield schema 和 result schema 必须由 verifier 精确校验。

删除把普通控制流重新包装成特殊编译许可的旧节点。特别检查并收敛：

- `state_stream`；
- physical `partition`；
- public `ordered`；
- 旧 ragged/members 专用执行节点。

如果 surface helper 仍存在，它必须在 frontend 机械展开为 domain、subregion、index relation 和普通控制流；canonical KIR 中不得保留第二套同义语义。

编译器不得把一个 `@intent.kernel` 自动拆成多个 target kernel。多个 kernel 仍由作者分别定义并由 host wrapper 编排。

## 四、结构化张量运算必须有正式语义

### Reduce 与 scan

Reduce/scan 必须是 first-class KIR operation，而不是靠分析普通循环猜出来。

generic combine 应表示成真正的 typed KIR region：

- 参数是两组 accumulator components；
- 每个 component 有明确类型和 identity；
- combine result schema 与 accumulator schema 完全一致；
- closure capture 必须显式成为 operands；
- constexpr 可以作为正式常量；
- closure 内禁止 load/store/atomic/RNG 等 effect；
- verifier 不替作者证明结合律，但必须验证类型、arity、effects 和 region termination。

内建 `sum/max/min/and/or` 可以作为语法糖，但只能 lower 到同一个 canonical reduce/scan 表示，不能保留两套实现。

### Region fold 与 region scan

`region_fold`和`region_scan`必须是独立first-class typed KIR operations，不能退化为旧`state_stream`、opaque callable、metadata或普通循环的pattern tag。

`region_fold`明确保存：

- source components及共同source axis；
- typed pure `summarize` region；
- typed pure `combine` region；
- 逐component typed identity；
- runtime captures作为显式operands；
- 固定summary/result schema。

`region_scan`还必须保存typed `apply`与`emit` regions、initial state、source-aligned output assembly和final-state flow。Verifier检查每个region的arguments、yield/result schema、purity、effects和termination，以及transition identity/action/output等式所需的结构条件；不替作者证明数学同态律。

所有source components必须在该axis上具有同一logical extent，并由同一组segment boundaries同步切片后传给`summarize/emit`。Source component可以在某个helper中未使用，但不能因此从另一helper的参数列表或provenance中消失。

`I.indices(source_axis)`产生的值在slice、helper call、broadcast、reshape/transpose、tuple/record passage和ordinary integer/comparison/select中必须机械保存absolute coordinate provenance；不能重新编号为chunk-local ordinal，也不能从shape、名称或相同extent反猜回来。

Identity必须在canonical数值语义下逐component中立。需要区分“没有成员”时，summary必须显式包含bool validity或等价typed state；禁止用可能在combine中产生NaN、或只在隐含值域内成立的sentinel冒充identity。Empty source返回typed identity。

### Contract 家族

Contract 在 KIR 中保存数学收缩关系：

- operands；
-保留轴与收缩轴关系；
- accumulator；
- dtype 和数值语义；
- broadcasting/batching 等算法关系。

不能保存 MMA shape、warp layout 或 provider primitive selection。

`scaled_contract` 和 `sparse_contract` 需要保留它们不可由普通 contract 丢失的算法语义，例如 scale encoding、scale relation、稀疏格式和 metadata schema。不要把它们退化成某个目标 API 名，也不要用模式匹配从一串普通运算偷偷替换整段算法。

### 其他 tensor/value op

逐项闭合 cast、bitcast、broadcast、reshape、join/interleave、比较、选择、索引和记录操作。每个 op 都要有：

- typed operands/results；
- 明确 shape relation；
- verifier；
- 无歧义的 canonical semantics。

不能让 analysis 和 emitter 分别从 result shape 反猜来源轴。

## 五、数据访问与 effects

把访问语义完整放进 KIR，而不是只留一个“可能读写”的布尔值。

至少需要表达：

- external view load/store；
- logical buffer 的分配语义与生命周期边界；
- gather/indexed load；
- scatter 与 scatter-reduce；
- predicate/validity 的算法来源；
- alias 和 In/Out/InOut；
- atomic family及其返回结果；
- RNG。

Atomic 应是语义明确的 op family，而不是一个带 target scope 字符串的万能入口。它需要表达操作种类、地址、值、返回值和语言层可观察的 ordering；不得在 DSL/KIR 中写死 GPU device/workgroup 等 target identity。

RNG 按最终规格固定为 Philox4x32-10 的逻辑随机位流。KIR 保存 counter、key、lane/result schema 等算法输入，不保存 GPU lane 或 vector packing。

## 六、用第一轮已迁移的 examples 横向闭合 KIR

完整扫描第一轮已经迁移的 `examples/kernels/`，确保每个最终surface都只生成新的canonical KIR。本轮不重新设计example算法，也不把源码改回旧构造来迁就frontend。

这不是挑几个例子改通。若源码仍残留旧surface，应把它视为第一轮遗漏并按最终规格消除；frontend与KIR中则必须清掉全部旧producer，包括：

- `partition(auto)`；
- `state_stream`；
- public `ordered`；
-旧 ragged/members executable op；
- 旧 random；
- 旧窄 atomic；
- 依赖 metadata 字符串传语义的写法。

保持作者算法不变。不要为了适配现有 compiler，把 online algorithm 改成 materialized algorithm、把多 kernel pipeline 合成一个 kernel，或者把普通控制流替换成 compiler 专用构造。

至少用下列不同结构压力检查新 KIR 是否横向成立：

- FlashAttention：region fold、显式valid summary、两个contract、坐标predicate与absolute provenance；
- causal linear attention：region scan、同步source components、transition/apply/emit与final state；
- online softmax：generic reduce或region fold的多component summary；
- two-pass reduction / split-K：显式多 kernel、partial ABI；
- record kernel：typed record carry 和 projection；
- ragged MoE：多个 index relation、contract、scatter-reduce；
- nested ragged pooling：嵌套 relation、tensor carry、多结果；
- scaled contraction；
- sparse 2:4 contraction；
- compare-exchange；
- Viterbi/动态规划：logical buffer、普通顺序循环、动态终止；
- nonzero/compaction：scan、indexed write、多结果；
- fused cross entropy：作者显式编排多个 kernel；
- attention backward：多结果、归约和 effect。

这些只是覆盖类别，不允许根据 kernel 名称添加处理。

## 七、关闭旧 KIR 路径并明确下一层边界

不能只改 `IntentOps.td`，却让frontend、verifier或canonical analyses继续生产和解释旧节点。本轮清理到KIR→GPU边界为止，不顺带重构shared GPU IR与provider leaf。

沿本轮use chain清理：

```text
frontend
→ ODS/types/attributes
→ KIR verifier
→ KernelModel
→ canonical analyses / KernelFacts
→ KIR-to-GPU typed boundary
```

必须删除：

- legacy KIR op/type/enum/attribute；
- 针对 `partition`、`state_stream`、旧 ragged op 的专用 facts；
- 从 op 名、region 名、shape 或 metadata 反查已经存在的 canonical identity；
- 同一个语义事实在 type、metadata 和 attribute 中并存的多份权威来源。

若下游旧GPU代码仍直接引用被删除的KIR节点，只做保证仓库可构建所必需的机械删除或接口断开；不要在本轮重写GPU program、passes或provider lowering。旧Plan mirror、`exec_*`、provider回读和materializer重建由第三轮整体删除，不能在这里再造一层过渡adapter。

Canonical analyses 可以从 KIR 推导 shape、index、effect、def-use 和 dependence facts，但推导结果只能是可失效、可重算的 analysis，不得成为第二份算法语义。

由于下一层GPU Program尚未在本轮完成，新KIR operation在KIR→GPU边界可以精确报`NotImplementedError`。不允许frontend拒绝合法语言，也不允许退回旧KIR或让provider从附近结构猜一个实现。

## 八、禁止的实现方式

- 不保留新旧 KIR 双路径；
- 不添加 compatibility flag；
- 不用默认值补缺失语义；
- 不按 kernel 名、op 数量、固定 shape 或 provider 名称分支；
- 不把 target physical facts 塞进 canonical KIR；
- 不让 emitter 重新解释作者算法；
- 不建立 test 目录、pytest、fixture 或测试脚手架；
- 不顺带实现下一层完整 GPU pass pipeline；
- 不更新 baseline CSV；
- 不把中间审计结果写进 `doc/`。

## 节点二：代码结构形成后的强制横向自查

完成主要代码改造后，先暂停新增实现，做一次内部横向审计。审计不单独产出文档。

逐个 canonical KIR family 检查：

```text
DSL producer
→ frontend/desugaring
→ typed KIR op/type
→ region/result schema
→ verifier
→ parser/builder
→ canonical analysis consumer
→ KIR-to-GPU typed boundary
```

必须回答：

- 是否每个最终语言构造都有唯一 canonical representation；
- 是否仍有旧节点或 compatibility fallback；
- 是否仍有 `AnyType`、opaque string 或 metadata-only semantic truth；
- 是否正确区分不同来源的动态 dimension；
- closure 是否是真正的 KIR region；
- region fold/scan的source axis、source component extent与lockstep slicing是否精确；
- summarize/combine/apply/emit的schema、purity、identity、output assembly与final state是否精确；
- absolute coordinate provenance是否机械传播，是否有helper或shape transform把它降成普通整数tensor；
- ordinary ordered loop、严格recurrence和dynamic stop是否仍是普通control，而未被误写成region operation；
- effect 是否由 op/interface 明确表达；
- loop carry 是否支持 scalar、tensor 和 record；
- 是否有任何canonical analysis通过名字、相同extent或邻近结构猜provenance；
- KIR→GPU边界是否明确消费typed KIR或精确unsupported，而不是兼容旧节点；
- 是否存在只让一个样例通过的 kernel/shape 特判；
- examples 是否覆盖上述不同语义结构，而不是只有 softmax/GEMM 两个点。

发现问题就在本轮修掉，然后重新执行这次横向审计。不能把问题登记为以后处理。

## 节点三：完成后的收尾自查

确认功能闭合后，再做一次纯减法收尾：

- 删除无调用者的旧 builder、helper、enum、diagnostic 和 handler；
- 删除旧KIR producer、consumer与canonical analysis分支；
- 删除重复的 shape/index/effect 推导；
- 检查没有被注释掉的旧实现、过渡开关或 fallback；
- 检查目录层次是否分别表达frontend、canonical KIR、analysis和KIR-to-GPU边界；
- 检查所有 example 已使用最终 DSL，并且没有为了旧 compiler 写的绕行结构；
- 检查 `doc/` 描述的最终 KIR 与实现一致；只修设计事实，不写进展、失败表或当前测试状态。

验证只保留一条可手动执行的端到端 repro：选择一个同时包含动态 subregion、structured reduction 和 tail validity 的真实 kernel，将 DSL emit 成 Triton 源码并实际运行一次数值对照。

这条 repro 只是确认唯一执行链可运行，不代表横向覆盖；横向完整性由前述 schema、consumer 和 examples 审计负责。不要新增任何测试文件或 fixture。

## 九、交付

完成后：

1. 产出一份报告：

```text
report/canonical-kir-reconstruction.md
```

报告写清楚：

- 最终 canonical KIR 的组成和边界；
- 删除了哪些旧语义与双份路径；
- 第一轮examples在新KIR中覆盖了哪些结构；
- 两次强制自查各发现并修掉了什么；
- 哪些 downstream consumer 已切到新 KIR；
- 唯一 repro 命令及数值结果；
- 是否存在因规格矛盾而无法闭合的地方。

2. 把本轮改动整理成一个语义连贯的提交。

3. 确认工作区干净。

不要为节点二、节点三另建报告；它们是本轮内部纪律，最终只保留一份总报告。
