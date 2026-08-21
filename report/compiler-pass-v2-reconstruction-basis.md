# Compiler Pass V2 重构依据：五轮之后的统一判断

## 0. 文档定位

这不是第六轮进度报告，也不是实验计划。它整理的是从重新阅读五轮产出开始，围绕 DSL 表达、Physical Program、compiler pass、provider-local realization、terminal emission、tuner、TMA、语料接入与失败分类形成的统一判断。后续 GPU 编译器重构应当以这里的边界为依据，而不是继续在当前构造路径上逐点修补。

本文只讨论 GPU 路径：Triton、cuTile 和 TileLang。它不讨论图编译、算子融合搜索、算法替换或者 CPU 后端。

为了避免再次混淆，本文严格区分四类陈述：

- **当前事实**：当前仓库的代码确实这样工作。
- **现行语言合同**：当前 DSL 文档已经规定的语义，即使这份文档认为它需要修改，也不能假装它现在不存在。
- **成熟态判断**：下一轮重构应当收敛到的架构与责任边界。
- **未决问题**：已经确认必须解决，但目前还没有足够依据冻结具体表面或 pass 排序的地方。

本文不把“尚未决定”包装成结论，也不设计探针、对照测量或新的测试体系。

---

## 1. 最终结论

IntentDSL 应当是一门 **target-independent、结构化、程序式的算子 DSL**。作者表达 logical domain、普通控制流、region、value flow、state、reduce、scan、contract、indexing、effects 和数值路径；作者不通过额外标志“授权”编译器优化，也不手写 GPU tile、program ownership、storage、copy、pipeline 或 target primitive。

Kernel IR 保存算法和逻辑数据流。GPU 编译器把它 lowering 成一份始终合法、可被逐步改写的 Physical Program。共享 GPU passes 决定跨 provider 成立的物理结构；provider-local passes 决定只有某个目标语言才有意义的原生形态；terminal translator 只把已经合法的 provider program 拼写成 Triton、cuTile 或 TileLang 源码。其后由目标 DSL 自己的编译器继续完成 layout、MMA、寄存器分配、软件流水和机器代码生成。

```text
Intent DSL
    ↓ frontend lowering
Kernel IR                         算法与逻辑数据流
    ↓ construct legal baseline
Physical GPU Program              完整但可继续改进的物理实现
    ↓ shared GPU passes
Refined Physical GPU Program      execution / value / access / structured-op
    ↓ provider-local passes
Provider-legal Physical Program   Triton / cuTile / TileLang 原生结构
    ↓ terminal translation
Provider source
    ↓ provider compiler
Machine code
```

因此，我们“到底在编译什么”的准确回答是：

> 在固定 Kernel IR 算法与逻辑数据流的前提下，编译器确定性地补回被 Intent 抽掉、但一份高质量 target program 必须具有的 physical realization，并把这些决定逐步变成 provider-native program。

这不是 whole-op 模板选择，不是算法搜索，也不是 autotune。它是一条由 typed semantic facts、def-use、region structure、reuse、lifetime、validity、structured-op semantics、device facts 和 provider capability 驱动的 canonical realization pipeline。

对于固定的 Kernel IR、device、provider 和 tuner configuration，结果仍然只有一份 canonical program。存在编译空间，不等于必须枚举这个空间；pass 中的确定性 policy 正是在这个空间里选择一条实现路径。

---

## 2. 五轮工作已经把现状推进到了哪里

### 2.1 已经完成的骨架变化

当前真实主链已经是：

```text
Kernel IR
  → ConstructPhysicalProgramPass
  → VerifyPhysicalProgramPass
  → MaterializeTargetProgramPass
  → intent_plan.target_program(provider, source)
  → terminal translator
```

`ConstructPhysicalProgramPass` 会分析 Kernel IR，创建 `intent_plan.program`，写入 device、launch、axis、range、buffer、padding、transfer、reduction、scan、pointwise、contract、sparse、stage、stream 以及 accumulator/replay 等决定，并把物理入口函数移入 `ProgramOp`。`VerifyPhysicalProgramPass` 对这份 Program 做完整验证。旧的“Kernel IR 函数和 Plan 各自都像 executable authority”的双重权威已经被消掉。

`MaterializeTargetProgramPass` 再调用各 provider materializer 生成源码，最终 translator 只输出 `TargetProgramOp` 中的 source string。这个终端 translator 已经很薄。

五轮里补上的 persistent 全范围核算、accumulator flow、compact coverage 和 scalar lane packing，证明当前构造器并不是空的“填表器”：它已经包含真实、可跨 kernel 生效的结构 policy。normal profile family 则属于显式 tuner policy，不应和这些结构决定混为一谈。

### 2.2 当前 Physical Program 到底是什么

它既不是纯粹的 “Kernel IR + 一张无行为的旁表”，也还不是 Triton TTGIR 那种可以独立承接后续全部物理改写的 lowered dialect。

准确地说，当前处在一个过渡态：

- `ProgramOp` 已经拥有物理入口函数和大量显式 decision op；
- 它有完整 verifier，不再允许两个 executable authority；
- 但大量 operation 仍然保留 Intent/KIR 形态；
- provider materializer 仍重新把物理入口分析成 `KernelModel`；
- materializer 在遍历这些旧形态时重建 index、access、loop 和 provider policy；
- `TargetProgramOp` 只有 `provider + source string`，不存在可继续改写、验证的 provider-legal SSA program。

所以，当前结构更接近“带 executable skeleton 的 decision-bearing physical IR”，还没有完成从 semantic IR 到 provider IR 的整条 lowering。

这点非常重要。下一轮不能只把几个 leaf `if` 搬进 `Build.cpp`，然后宣称已经有 pass pipeline。那只会让单体构造器更大。真正的目标是让 Physical Program 成为后续 passes 实际改写的唯一程序，并让 provider-specific 结构在生成字符串之前也有 IR 中的位置。

### 2.3 五轮没有完成的事情

当前 GPU transform pipeline 里真正注册的物理 passes 仍然只有 construct 和 verify。绝大多数决定在一次前向 `buildPhysicalProgram` 中完成；这意味着：

- 决定虽然被记录了，但没有各自明确的 pass 位置；
- 早期决定与晚期决定的依赖顺序藏在构造调用栈里；
- 后续结构改变很难重新检查并改写前面的决定；
- reuse、lifetime、producer-consumer、device resource 等事实还没有形成可复用的独立 analyses；
- provider-local 选择仍大量藏在源码 materializer；
- provider program 一旦成为字符串，就失去了 verifier、rewrite 和 analysis 的对象。

因此，五轮完成的是 **从散落 leaf constructor 向显式 Physical Program 的第一阶段收敛**，不是成熟 Pass V2 的终点。

---

## 3. “权限”不是正确的 DSL 与编译器模型

### 3.1 编译器不需要用户手动授予实现权限

编译器能否做某个变换，来自程序语义和可证明的不变量，不来自作者额外写一句“允许编译器这样做”。如果一个 construct 的唯一用途是向编译器发放优化许可，而它本身不改变源程序所表达的算法，那么这个 construct 暴露错了层级。

例如，一个 contraction op 的语义本身已经说明结果轴、归约轴、combine 行为与数值合同。编译器可以选择合法的 tile、dot primitive、accumulator 和 traversal，不需要作者再授予“可以分块”或“可以改写”的权限。普通顺序循环的语义本身要求顺序；真正独立的 parallel loop 语义本身允许并行实例之间无顺序依赖。两者都是语言含义，不是授权标志。

即使是 fast-math 一类表面，也应当理解为切换了数值语义合同，而不是给同一语义下的编译器发许可证。

所以，后续语言与文档应当停止用“权限边界”解释 DSL construct，改成下面四层：

1. 作者写出的 **算法语义**；
2. 编译器从语义中证明的 **derived facts**；
3. compiler passes 选择的 **physical realization**；
4. tuner 或下层编译器绑定的 **数值参数和机器细节**。

### 3.2 这不等于删除所有结构信息

去掉“授权式 construct”不等于把 DSL 退化成 TVM TE 式的纯数学声明。IntentDSL 的核心价值恰恰是保留一个真实算子程序，而不只是输出张量公式：

- 显式 logical domain 与索引关系；
- region 和算法可见的分段；
- 普通控制流和循环嵌套；
- state、carry、stop 与 recurrence；
- value flow、buffer、effects 和 mutation；
- reduce、scan、contract 等 structured operation；
- ragged、sparse、validity 和数值路径。

应当删除的是物理实现泄漏，不是算法结构。成熟的 DSL 仍然是程序式的、可读的、能直接表达真实 kernel 算法的，只是不要求作者把同一算法手工改写成 GPU blocking skeleton。自然语法本身应当承载语义：普通 `for`、value use、state update、structured op 和 effect 足以让 frontend 建立内部 facts，而不是让作者在程序旁边再写一组给 compiler 读取的声明标签。

### 3.3 判断一个表面是否该保留的标准

对每个 public construct，应问：

> 如果换一种合法 GPU 实现，这个 construct 所表达的算法可观察行为是否仍然必须保留？

- 如果必须保留，它是语义，应留在 DSL/KIR。
- 如果不必保留，只是为了当前后端获得 tile、worker、storage 或 pipeline，它应进入 physical pass。
- 如果可以从其他语义唯一推出，它应成为 analysis fact，而不是要求作者重复标注。
- 如果只对某个 provider API 有意义，它应成为 provider-local IR/pass，而不是 public Intent surface。

这个标准比“它现在被哪个函数读取”可靠，也比“它看起来像算法还是优化”更可审计。

---

## 4. `partition`、`ordered`、`parallel` 与 `state_stream` 的重新定性

这里必须同时保留当前合同和成熟态判断，不能把拟议修改写成当前事实。

### 4.1 当前合同中的 `partition`

现行文档把 `partition` 定义成算法可见的 region：body 看到一个子区域，`extent` 也进入当前语言合同。因此，直接删除所有现有 `partition` 会改变一部分程序的算法结构。

当前合同和实现自身也没有完全闭合：文档描述了 runtime/`I.Constexpr` extent 以及 `partition(count=P)`，而 Python frontend 目前只接受 `(axis, extent)`，并要求正的 compile-time integer 或 `I.auto(...)`。这不是 compiler 可以默默猜测的细节；在重做 surface 前必须明确哪些 dynamic partition 是真实算法需求、哪些旧文档表面从未实现。

但是当前表面混合了两种完全不同的东西：

1. **逻辑分段**：segment、window、page、算法可见 chunk，body 的语义确实依赖这个区域；
2. **物理 blocking**：只为了让同一个完整逻辑域分块执行，分多大和如何嵌套不属于算法。

成熟态必须拆开这两者。逻辑分段应当保留为自然的 region/segment/window 语义；物理 blocking 应由 compiler pass 从完整 logical domain、structured op、device 和 provider facts 推导。当前所有 `partition` 使用点都必须逐项做语义审计，不能用一次全局替换完成。

`I.auto` 一类“作者声明这里有 tile，但大小交给编译器”的表面尤其值得删除。它仍然迫使作者先替编译器决定“这里必须有一个 physical partition”。成熟的 compiler 应当自己决定是否引入 physical region、region 的嵌套关系，以及哪个 logical axis 进入 launch 或 kernel 内循环。

### 4.2 `ordered` 不应成为普通顺序程序的额外标志

现行 DSL 把 `ordered` 冻结成独立 construct，当前 compiler 也只从它建立特定的 ordered physical facts；因此，在当前实现里它不能无条件改名后删除。

但成熟表面不应要求作者为了表达普通顺序循环而额外写“ordered”。自然的语言规则应当是：

- 普通 `for` 就是顺序循环；
- state/carry/effect 直接属于这个循环的语义；
- frontend 可以在内部 lowering 成显式 sequential/ordered KIR op 或 fact；
- compiler passes 读取内部事实，不要求用户再次授权。

也就是说，内部 IR 仍然需要可验证的顺序信息，public DSL 不一定需要一个生硬的 `ordered` 标志。删除当前 `ordered` 表面之前，必须先让普通 `for` 完整承接它的 carry、effect、stop 和 lowering 语义；否则只是删掉了信息。

### 4.3 `parallel` 的情况不同，但也不应承担 physical mapping

`parallel` 可以是一种真实的程序语义：不同逻辑实例独立，不存在由 source order 定义的 carry。这不是“允许 GPU 并行”的授权，而是 parallel loop 本身的含义。

因此，第一阶段不能把所有 `parallel` 也机械删除，然后假定 compiler 总能从普通循环证明独立性。更稳妥的成熟边界是：

- public `parallel` 只表达逻辑实例独立；
- 它不表达 program id、lane、warp、worker、tile 或 launch shape；
- 这些 physical mapping 全部由 execution passes 决定；
- 将来如果依赖分析足够强，可以让普通循环自动转为内部 parallel fact，但这不是本轮已经成立的事实。

### 4.4 `state_stream` 中应保留什么

state、carry、递推顺序、stop 条件和可见输出是算法语义，必须保留。一个 state stream 是否按 persistent program、按多少元素一块、采用怎样的 worker/fold/reuse、是否分 stage，是物理实现。

当前 DSL 文档和 frontend 都只允许 compile-time integer 或 `I.auto(...)` 作为 `state_stream.extent`；dynamic logical endpoint 由 `stop=I.end(...)` 表达。固定整数在 KIR 中被物化成 operand，C++ schema/诊断却把这个 operand 称为 “runtime extent”，随后又强制它来自正的 `intent.constant`。这首先是内部表示与命名没有闭合，不是 public DSL 已经支持 runtime segment extent。下一轮仍需明确：stream/carry/stop 属于算法，而 auto/fixed segment chunk 在 Physical Program 中如何表示和改写。

### 4.5 `reduce`、`scan`、`contract` 不是优化许可

这些是高层 structured semantics。作者应当表达：输入、输出、逻辑轴、归约或扫描关系、identity、carry、数值顺序要求，以及算法真正需要的中间值。作者不应表达 GPU tile、shared allocation 或 provider primitive。

尤其对于 contract，compiler 应能从完整逻辑 contraction 的结果轴和归约轴引入必要的 physical region 与嵌套，而不是要求 DSL 作者先写一个 GPU blocking skeleton。这是一个真实的新 compiler decision，不是简单语法糖；但它也不要求把 DSL 退化为只剩一个数学表达式。

---

## 5. Intent 到 target source 之间真正存在的编译空间

这个空间不是无限的，也没有必要夸张成算法搜索。对 Triton 来说，它确实比一开始想象得窄，因为 Triton compiler 会继续承担 layout、coalescing、dot operand、MMA、warp specialization、software pipeline 和低层代码生成。但“较窄”不等于“只填空”。

对于固定的 Kernel IR，尚未唯一确定而又会改变 target program 结构的决定，可以稳定地归入四组。

### 5.1 Execution realization

它决定完整 logical domain 如何成为一个 GPU program：

- 哪些 logical axes 形成 launch/grid，哪些成为 kernel 内循环；
- 是否以及在哪里引入 physical region；
- physical region 如何嵌套；
- lane、worker、group、fold 和 reuse 的关系；
- persistent traversal 与普通 traversal；
- loop placement、range 和 launch ownership；
- compiler-private stage 的 execution dependency。

这里不是简单给 `program_id` 换个名字。`program_id` 只是选定 execution decomposition 后的一种 target spelling。真正的决定是：一个逻辑工作域被拆成哪些 program-local 工作、哪些工作在程序内部迭代、哪些值跨这些迭代复用。

### 5.2 Value realization

它决定一个逻辑值在物理程序中以什么方式存在：

- inline、replay、materialize 或 workspace；
- program-local residency、shared residency 或跨 program storage；
- accumulator、carry 和 compiler-private temporary；
- producer-consumer 之间是否保留、重算或搬运；
- lifetime 与 reuse scope；
- 多个 use 是否共享同一物理值。

中间值本身当然来自 DSL/KIR；compiler 决定的不是“凭空创造另一个算法中间值”，而是同一个逻辑值的物理存在方式。这个差异会直接改变 load 数量、live range、register pressure、shared traffic 和 workspace traffic。

### 5.3 Access and validity realization

它决定逻辑索引和有效域怎样落到目标程序的访问面：

- logical range 与 selected physical range 的关系；
- boundary/validity 在哪一个 access 上兑现；
- mask、padding、guard 和 neutral value 的组织；
- contiguous、gather、scatter、bulk copy 或其他 provider access form；
- 哪些 validity 可以下推、合并或由后续 use 吸收；
- copy 是由哪些 value forms 之间的差异产生。

mask 的某些部分是确定性 correctness lowering，但 mask 放在哪里、是否重复、能否变成 padding 或 provider-native access，仍会改变代码质量。不能把“最终一定要正确处理边界”误解成“所有 access form 都已由 KIR 唯一决定”。

### 5.4 Structured-operation realization

它决定 reduce、scan、contract、state stream、sparse operation 如何成为物理骨架：

- reduction/scan 的 program-local 与跨阶段结构；
- contract 的 operand form、accumulator flow 和 primitive composition；
- state stream 的 persistent traversal、carry placement 和 stage；
- structured op 与 surrounding loop、reuse、materialization 的结合；
- provider capability 不同导致的合法 native form。

这不是替换算法。attention 仍然是原来的 attention，contract 的逻辑轴和数值路径不变，state recurrence 也不变。变化的是同一 structured operation 如何被组织成 target tile program。

### 5.5 tile 参数处在什么位置

tile 不能成为整个 compiler space 的代名词。它只是上述决定的一部分数值化结果。

- 某个 physical region 是否存在、属于哪个 loop、与哪些值共享，是结构决定；
- tile/warp/stage 的具体数值可以由 device rule 固定，也可以暴露给 provider tuner；
- provider tuner 绑定数值，不应重新决定算法、value flow 或 physical skeleton；
- 下层 compiler 继续决定它自己拥有的 layout 和机器级 tile。

因此，V2 可以既是 deterministic pass pipeline，又保留 tuner configuration。两者不冲突，也不能互相替代。

这些 policy 也必须是 per-device 的，而不只是 per-provider 的。相同 Triton target 在 H100 与 RTX 5090 上可以根据 device resources、capability 和 tuner configuration 得到不同的 canonical physical realization；这种差异来自硬件事实，不来自按 kernel 名称选择另一份模板。

---

## 6. 为什么这不是模板选择器

如果 compiler 只按 kernel 名称或 whole-op signature 选择一份 GEMM、stream 或 ragged 模板，那么它确实没有形成成熟算子编译器。

V2 的规则必须按 typed KIR structure 组合生效：

- 一个 pass 读取 axis、region、def-use、structured-op、reuse、lifetime 和 device facts；
- 它只改写满足这些结构条件的局部 physical program；
- 同一规则可作用于多个 kernel family；
- 一个 kernel 的实现由多个局部规则组合，而不是一次 whole-op 命中；
- 不允许用 kernel name、op count、完整 region matcher 或 legacy fallback 代替语义条件。

deterministic canonical path 也不等于模板。Triton 的许多优化 passes 同样用确定性规则把很大的合法实现空间压成一条 canonical path。区别在于规则的输入是 IR facts，输出是可继续改写的 IR，而不是从库里取出一份完整源码。

---

## 7. 成熟 Physical Program 的形态

### 7.1 一份不断被改写的 IR，而不是 KIR 与 Plan 的长期同步

成熟态应当保留 Kernel IR 作为不可变语义输入，同时转换出一份新的 Physical Program 承接物理决定。后续 passes 改写的是 Physical Program，不是在 Kernel IR 旁边持续填一张注解表。

Physical Program 可以保留到 KIR value/op 的 provenance，analysis 也可以借助这些引用重新取得逻辑语义；但 provider lowering 不应再靠遍历旧 KIR 重新发明 execution structure、index form 或 value placement。可执行结构必须已经存在于 Physical Program 自身。

Triton 的具体实现不是长期同步两份 module：`make_ttgir` 在同一个 `mod` 上运行 `convert_to_ttgpuir`，TTIR operation 被转换/替换，后续 passes 直接改写已经携带 layout 的 TTGIR。真正值得借鉴的是 lowered dialect 承接决定、旧 IR 不再作为第二个 executable authority，而不是“必须创建另一个 module”。Intent 可以保留不可变 KIR 作为语义/provenance 来源，但物理执行结构不能继续由 KIR 与旁表共同解释。

### 7.2 IR 在每个 pass 之间都必须完整合法

不需要 partial Plan，也不需要“某字段目前尚未决定”的特殊生命周期合同。baseline constructor 应当先生成一份保守但完整的合法 Physical Program；后续 passes 用更好的合法实现替换它。

例如，一个 value 一开始可以有保守 residency，一个 axis 可以有保守 blocked execution；后续 pass 根据 reuse、lifetime 或 provider capability 重写。变化的是实现质量，不是 IR 是否有含义。

每个 pass 后运行完整 verifier。verifier 只检查明确的不变量：SSA、region/range 一致性、value placement 合法性、structured-op operand/result 合同、provider capability 和 executable completeness。它不判断某个 policy 是否足够快。

### 7.3 不需要字段唯一写入权限

成熟编译器不靠“这个字段只能由某 pass 写”维持纪律。后面的 pass 可以重写前面的 execution、residency、range 或 structured form，只要改写后 IR 仍然合法。

真正的纪律来自：

- pass 顺序被显式设计；
- 每个 pass 后 verifier；
- analysis 与 IR 分离；
- 默认保守失效 analysis，只有明确保持时才声明 preserved；
- 必要的 canonicalization 或 refinement pass 可以在固定位置重复运行，而不是发明全局收敛协议。

这也意味着 persistent pass 晚些时候改写 worker/fold/reuse 是正常编译器行为，不需要“耦合决策族”或字段所有权机制。

### 7.4 decision、derived fact 与 emitted fact 的边界

- **Decision**：KIR 无法唯一恢复、且改变 physical program 的选择，应写进 IR。
- **Derived fact**：能从当前 IR 唯一重算的 def-use、provenance、index relation、lifetime summary，应由 analysis 提供。
- **Emitted fact**：只是在 provider syntax 中唯一拼写已有结构，不应再存第二份真理。

当前 Plan 已记录许多 decision，但 materializer 仍重算并重新选择一部分结构。下一轮应当以这个分类逐个审计 target-source 分支。

---

## 8. Pass pipeline 应当承担什么

成熟 V2 不是“增加很多名字叫 pass 的函数”，而是把真实物理决定变成可验证、可组合、可重写的 IR transformations。

### 8.1 共享 GPU pipeline 的责任槽位

下面是稳定的责任槽位，不是已经冻结的 C++ pass 名单：

1. **Baseline construction**：从 KIR 生成完整、合法、保守的 Physical Program。
2. **Region and execution formation**：引入 physical region，决定 launch/loop/traversal/worker/persistent/fold/reuse。
3. **Value realization**：根据 def-use、reuse、lifetime 和 structured consumption 决定 replay、residency、materialization、workspace、accumulator 与 carry。
4. **Access and validity realization**：形成 range、padding、mask、transfer 与 validity placement。
5. **Structured-op realization**：形成 reduce、scan、contract、state、sparse 的共享物理骨架。
6. **Reconciliation/canonicalization**：在后续结构改变后重新检查 range、value placement、reuse、validity 和 structured form。

这些槽位之间存在真实依赖，不应假装一次线性填字段就能解决。structured op 会影响 value form，value form 会影响 copy/access，persistent traversal 又可能回头改变 range 和 reuse。因此成熟 pipeline 可以固定地重复某些 refinement/canonicalization pass；不需要 cost model，也不需要运行到不动点。

最终的具体 pass 拆分和顺序应由即将修改的 IR 定义决定。这里只冻结责任边界：这些结构变化必须发生在 terminal translation 之前，并且每次变化都写回 Physical Program。

### 8.2 shared pass 读取什么事实

共享 policy 不应只依赖 op kind 和几个常量。它应能读取：

- typed axes、logical ranges、region nesting；
- def-use 和 producer-consumer structure；
- reuse count、reuse distance 与 lifetime；
- effect、state carry 和 validity；
- reduce/scan/contract 等 structured semantics；
- device resources 与跨 provider 通用的 GPU capability；
- 当前 Physical Program 已经选择的 execution 和 value facts。

只有这样，同一 policy 才会在不同 Kernel IR 上产生不同 physical program，而不是退化成按 role 组合展开局部模板。

### 8.3 provider-local pipeline 同样是 pass pipeline

进入 Triton、cuTile 或 TileLang 后，只有该 provider API 才有意义的结构选择仍然应当是 IR pass：

- provider form selection；
- provider bufferization/storage form；
- native copy/gather/collective 选择；
- provider-specific loop/pipeline surface；
- provider legality 和 capability legalization。

这些 pass 可以向同一 Physical Program 引入 provider-specific op/attr/type，也可以转换到同一体系下的 provider dialect。它们不应藏在 string emitter 的临时分支里，也不需要另造一个不可验证的 side mechanism。

这不是在 GPU Realizer 与 leaf 之间再发明一个产品层级。成熟的 provider leaf 本身就应由“provider-local passes + terminal translator”组成；前者仍属于该 leaf 的实现责任，只是它们对 IR 作出并记录结构决定，后者才负责最终拼写。

### 8.4 terminal leaf 应该剩下什么

terminal leaf 仍然很重要，但它的“性能责任”应被精确定义为 provider-native 拼写和合法 translation：

- SSA name、pointer expression、mask expression；
- 目标 API 的调用语法；
- decorator、signature 和 launch wrapper；
- 已选 loop、allocation、copy、primitive 的确定性展开；
- provider source 的 syntactic legality。

如果一次修改改变了 loop nesting、ownership、storage form、copy topology、materialization、workspace、structured traversal 或 primitive selection，那它就不是 terminal spelling，应前移到 shared 或 provider pass。

“leaf 写不好会造成数量级差距”仍然可能成立，但这通常说明所谓 leaf 其实仍在做结构决定。架构闭合后，纯 translator 的错误主要应表现为错误拼写、非法 API 或没有忠实投影既定 IR。

---

## 9. 性能为什么主要应从 pass 中产生

### 9.1 pass 形式不是性能，pass policy 才是性能

把当前 `if` 从 materializer 搬到 Realizer、生成完全相同的 source，只完成了架构闭合，不自动提高性能。性能来自 pass 读到了哪些事实、选择了怎样的 execution/value/access/structured form。

但是，没有 pass/IR 边界，结构 policy 无法组合、重跑和审计，所有优化最终只能继续堆进 leaf。因此，对于后续可持续优化，pass pipeline 是必要条件，不是充分条件。

### 9.2 “以后 90% 优化在 pass”应该怎样理解

这不是一个已经测量出的性能百分比，而是一条架构责任原则：

> 绝大多数会改变 target program 结构的优化，都应表现为 shared 或 provider-local physical pass，而不是 terminal source emitter 修改。

仍然留给 provider compiler 的 layout、MMA、register allocation、instruction scheduling 和 software pipeline，不是我们 leaf 的工作；仍然留给 translator 的 API 拼写，也不是 shared pass 的工作。所谓“主要在 pass”，指的是 Intent 到 provider-legal program 之间由我们拥有的结构空间。

### 9.3 当前五轮优化应如何重新理解

五轮中的这些改动已经是未来 pass policy 的原型：

- persistent 全范围核算属于 execution/range reconciliation；
- accumulator flow 属于 value 与 structured-op realization；
- compact coverage 属于 access/validity realization；
- scalar lane packing 属于 execution/access realization；
- normal profile family 属于 tuner search-space declaration，不是结构 pass 本身。

它们目前集中在 construct/materializer 路径，是因为成熟 pass 边界还没有建立，不是因为性能天然属于 leaf。

### 9.4 可跨 kernel 复用的优化点

当前真正值得形成通用 policy 的内容包括：

- 从 logical domain 与 structured op 形成 physical region 和 loop nesting；
- 根据 reuse/lifetime 决定 replay、residency 和 materialization；
- persistent traversal 与 worker/fold/reuse 的一致重写；
- accumulator、carry 和 state 的物理放置；
- validity 下推、padding、mask 和 transfer form；
- structured op 与 surrounding loop/value form 的组合；
- provider-local storage/copy/collective/primitive form selection。

这些规则不依赖 kernel 名称，能够在不同 family 之间复用，才构成真正的算子编译器性能知识。

---

## 10. Triton 路径：空间较窄，但绝不是没有工作

Triton 的 Intent leaf 相对可以更薄，是因为 Triton compiler 自己有完整的 TTIR→TTGIR→LLVM pipeline：TTGIR tensor 类型从进入 GPU dialect 起就带合法 layout encoding，后续 coalesce、accelerate matmul、optimize dot operands、remove layout conversions、warp specialization、latency scheduling、software pipeline、TMEM/TMA/MMA 等 passes 会继续改写它。

因此 Intent Triton 路径不应重复决定：

- 最终 tensor layout encoding；
- MMA instruction 细节；
- register allocation；
- 下层软件流水和 instruction scheduling；
- 能由 Triton IR/target backend 合法推导的机器级 lowering。

但 Intent 仍必须决定并正确表达：

- program decomposition 与 kernel 内 loop；
- persistent traversal、reuse 和 value lifetime；
- inline/replay/materialization/workspace；
- pointer/index/mask 的高质量 logical form；
- reduce、scan、dot、state 等 primitive composition；
- 哪些结构成为 Triton 编译器能够继续优化的 surface。

一个糟糕的 Triton source 即使交给同一个下层 compiler，也可能因为 loop、live value、mask、load form 或 primitive composition 不同而产生完全不同的代码。下层能推 layout，不代表它会替上层重建被错误组织的算法程序。

### 10.1 当前 TMA 的准确状态

当前 Intent Physical Program 没有 descriptor/TMA transfer 形态，Triton materializer 主要发射普通 pointer、mask、`tl.load`/`tl.store`、dot/reduce/scan surface。它不能有意识地选择一条与手写 tensor descriptor/TMA source 相同的路径。

这有两种合法处理：

1. 如果普通 Triton IR 足以让下层 backend 自己选择 TMA，就继续把决定留给 Triton compiler；
2. 如果某类高质量 source 必须显式构造 descriptor/TMA surface，增加 Triton-local transfer form/op 和对应 pass，再由 translator 发射。

不能把 TMA 放进跨 provider shared Plan，也不能把它当成 V2 宏观问题的答案。它只是 provider-local access realization 的一个具体缺口，是否需要显式拥有取决于目标 Triton surface 与下层能力边界。

---

## 11. TileLang 路径：显式投影更多，provider passes 也更多

TileLang 要显式表达 allocation、`T.copy`、synchronization、`T.Pipelined` 和 `T.gemm`。因此它需要比 Triton 更多的 provider-native materialization，但“要写的投影更多”与“存在更多独立算法决定”不是一回事。

前一轮“`T.gemm` 一确定，alloc/copy/barrier 全部唯一推出”的判断只对了一半：

- 一旦 operand storage form、copy topology 和 primitive variant 已经选定，许多 allocation、copy 与 synchronization 的确是确定性派生；
- 但 `T.gemm` 本身支持不同 operand/storage 组合，SS、SR、RS、RR、TS 等 native form 并非由逻辑 contract 唯一决定；
- bulk copy、parallel element transfer、shared staging、fragment form 和 capability constraint 也可能有多个合法 provider realization。

因此 TileLang 不需要一个脱离上下文、单独搜索“内存放哪里”的全局 pass，但需要清晰的 provider-local pipeline：

```text
shared physical facts
  → TileLang form selection
  → storage-form/bufferization
  → copy and synchronization materialization
  → pipeline/primitive legalization
  → terminal TileLang translation
```

`alloc_shared`、`alloc_fragment`、`T.copy` 和 barrier 在 form 已选后可以是派生 lowering；选择哪一种 form、是否 staging、哪个 loop 可 pipeline，则是 provider-local physical decision。当前 materializer 中按 rank-reducing chain 构造 symmetric tile、选择 bulk copy/parallel elements、延迟 pointwise、组合 allocation/copy/gemm 的逻辑，应从 string construction 中分离出来。

`T.Pipelined` 包裹哪个 loop 是结构事实；pipeline stage 数值可以交给 tuner 或下层 policy。不能把二者都称为一个 config。

---

## 12. cuTile 路径：native tile/access form 也需要显式位置

cuTile materializer 当前会从 physical/KIR facts 推导 load、store、gather、scatter、bounds 和 collective surface，同时还会在本地添加 `GATHER_SPELLING`、row occupancy 等选择。

如果不同 gather spelling 会改变访问形态、tile shape 或 provider primitive，它就不是纯字符串拼写，应进入 cuTile-local form-selection pass；只有在 provider IR 已经唯一指定 gather form 后，API 名称和参数组织才属于 translator。

cuTile 与 TileLang/Triton 的原则相同：共享 semantic/physical facts 不重复选择，provider-specific form 有显式 IR 位置，terminal source emitter 不临时决定结构。

---

## 13. Provider tuner 到底是什么

当前 provider-specific Python/C++ 生成路径中按 GEMM、stream、ragged 等 role 组合准备的经验配置，本质上和手写 Triton 的 `@triton.autotune(configs=[...])` 是同一类东西：它们是候选数值配置和适用范围，不是 compiler pass，也不是 Intent 相对于手写 source 的本质性能知识。

它们可以包含：

- tile size；
- warp/thread 数；
- pipeline stage 数；
- 某些 provider numeric knob；
- 一组被允许尝试的组合。

它们不应决定：

- logical region 如何嵌套；
- value 是 replay、materialize 还是 workspace；
- persistent traversal；
- shared value 与 copy topology；
- structured-op skeleton；
- provider native form 的语义选择。

结构 pass 可以声明“这里有一个可调参数”和它的合法约束，provider tuner 再绑定数值。手写 source 同样可以提供 configs，所以不能靠 tuner 解释 Intent compiler 的价值。

当前一些 provider 文件硬编码 role-based config，最多说明已有经验搜索空间；把这些 config 从 leaf 整理到清晰的 SearchSpace 只是责任闭合，不等于新的性能优化。

---

## 14. JIT 时间为什么看起来很长

当前测量链实际包含多个阶段：

```text
intent.compile
  → Python frontend lowering
  → 外部 intent-compile C++ 子进程
  → 生成 provider source
artifact.run（第一次）
  → Python compile/exec wrapper
  → provider JIT
  → provider autotune / exhaustive search
  → launcher preparation
后续 timed runs
  → p50 runtime
```

重要事实是：provider JIT/autotune 通常发生在第一次 `artifact.run`，不在 C++ Physical Program construction 本身。Triton 生成 `@triton.autotune/@jit`，cuTile 使用 configs/exhaustive search，TileLang 使用 `@autotune` 与 `@tilelang.jit`。source baseline 也有自己的 JIT/autotune。

五轮里出现的 63 candidates、单 candidate 约十余秒的问题，是两部分叠加：

- 我们的 provider search-space policy 生成了过大的 Cartesian config family；
- provider compiler 为每个 candidate 支付真实 JIT/benchmark 成本。

所以它既不是“下面 compiler 自己 tune，和我们完全无关”，也不是 C++ Intent compiler 对同一 config 重复运行 63 次。我们负责候选集合为何如此大，provider 负责每个候选的编译与测量。

当前 `intent.compile` 每次还会创建临时目录并启动外部 compiler 进程，没有 compilation cache；这是一次固定的自身开销。但没有分段证据时，不能声称它占总 JIT 时间的主要部分。

当前 p50 测量发生在第一次 run 之后，通常不包含前述 JIT/autotune 时间。因此“编译很慢”和“steady-state kernel 很慢”是两个不同问题。

### 14.1 当前失败分类会掩盖真实阶段

现有 measurement 把 Intent compile、第一次 artifact run、provider JIT 和 launcher preparation 的异常较宽地包装为 generated compilation error，最终 CSV 统一显示 `compile_failed`。因此这个标签不能证明失败发生在 C++ compiler，更不能自动证明是 DSL、provider 或环境中的哪一层。

成熟工具链应当至少在产物与错误语义上区分：

- frontend/KIR failure；
- Physical Program construction/verification failure；
- provider materialization/legalization failure；
- provider JIT failure；
- launch/runtime failure；
- numerical mismatch；
- source adapter 或环境不可用。

这不是要求本轮新增探针，而是重构 error boundary 时不能继续把多个阶段折叠成同一个“compile_failed”。

---

## 15. 语料接入：28 个缺口必须真正闭合

五轮报告中的真实 inventory 是 119 个 runtime-visible source entries，registry 已接入 91 个，仍缺 28 个：Triton 10、cuTile 9、TileLang 9。

“给出状态”不算接入。每个 source entry 最终必须进入真实 registry 和 baseline 路径；如果现有 DSL 已经表达同一算法，可以复用 DSL 并新增 provider/source adapter；如果算法结构尚不存在，就必须创建 DSL，而不是只记录 unsupported。

### 15.1 Triton 缺口（10）

1. fused cross entropy
2. fused LayerNorm family
3. fused linear cross entropy
4. split-K paged attention
5. xFormers RMSNorm
6. causal Conv1D backward
7. modern FlashAttention forward
8. MoE split-K expert projection
9. MoE column-major expert projection
10. Mamba3 SISO sequence forward

### 15.2 cuTile 缺口（9）

1. official fused MoE
2. dense attention forward
3. grouped flash decode
4. TileGym dense GEMM
5. attention sink decode
6. Gemma split-K decode
7. chunk gated delta
8. fused linear cross entropy
9. NVFP4 quantization

### 15.3 TileLang 缺口（9）

1. persistent MLA decode
2. fused routed/shared MoE
3. DeepSeek V3.2 top-k selector
4. fused chunk linear attention backward
5. attention-sink backward
6. sparse MLA backward
7. BitNet int8 × packed-int2
8. BF16 × FP4 dequant GEMM
9. block FP4 activation quantization

### 15.4 “接入完成”的含义

一个 entry 只有在下面这条链真实成立时才算接入：

```text
source inventory entry
  → 对应 Intent DSL 算法
  → Kernel IR
  → Physical Program
  → provider source
  → provider compile/JIT
  → GPU 实际执行
  → 与 source/reference 做数值比较
  → 进入同一 baseline 记录
```

forward/backward、多 kernel pipeline 和不同 runtime-visible callable 必须分别保留，不能因为名字属于同一 family 就合并掉。按照项目验证纪律，每次纵向接入只保留一条可手动执行的 repro 命令，不另建 test/fixture 体系。

---

## 16. 28 个 inventory 缺口与 35 个 `compile_failed` 不是一回事

28 表示 source entry 根本尚未进入 registry；35 表示固定矩阵中已经尝试的 case 在宽泛的 generated path 中失败。二者不能相加，也不能互相解释。

五轮固定矩阵共有 182 条，147 pass、35 compile_failed。当前记录的失败分布是：

### 16.1 Triton

- RTX 5090：`flash_attention_backward`、`block_sparse_gqa_decode`
- H100：`flash_attention_backward`

### 16.2 cuTile

- H100：`block_scaled_gemm`、`sparse_mla_prefill`

### 16.3 TileLang

两台机器各有同一组 15 项：

- `block_sparse_gqa_decode`
- `gqa_decode`
- `varlen_gqa_decode_logits`
- `paged_mla_decode`
- `conv2d`
- `deepgemm_fp8_2xacc`
- `linear_attention_forward`
- `retention_forward`
- `mhc_pre`
- `varlen_block_causal_attention`
- `native_sparse_attention_forward`
- `native_sparse_attention_decode`
- `gqa_attention_backward`
- `fp8_lighting_indexer`
- `grouped_gemm_backward`

这些标签中可能混有 provider capability、环境/adapter、JIT、launcher 和真实 generated program 问题。当前错误边界不足以据此断言“V2 compiler 回归了 35 项”，也不能反过来把它们一概解释成环境问题。

如果记忆中某些 V1 case 曾成功，只有在相同 entry、相同 provider、相同 device、相同 input 和相同执行阶段下才能构成回归证据。五轮矩阵扩大了 source/registry/device 范围，本身不能与旧的成功总数直接比较。

另外，第五轮做过定向 A/B，但没有把这些定向结果回填成一份新的全量固定 CSV。后续文档必须继续区分“固定矩阵事实”和“之后的定向结果”，不能用后者改写前者的统计口径。

---

## 17. 当前 leaf 中已经确认的边界泄漏

下面这些不是抽象猜测，而是当前 provider materializer 中已经存在的现象：

| 当前行为 | 现位置 | 成熟责任位置 | 判断 |
|---|---|---|---|
| Triton row-vector/occupancy 重新判断 | Triton materializer | shared execution fact 或 Triton-local pass；数值候选进 tuner | 不是纯 source spelling |
| cuTile `GATHER_SPELLING` 与 row tuning | cuTile materializer | cuTile form-selection pass + tuner config | 访问形态变化时是结构决定 |
| TileLang rank-reducing chain 的 symmetric tile | TileLang materializer | shared/provider structured realization | materializer 在重建 physical shape |
| TileLang bulk copy 与 parallel element transfer 选择 | TileLang materializer | TileLang access/form-selection pass | provider-local physical decision |
| TileLang pointwise defer、shared/fragment allocation、copy、sync、gemm 组合 | TileLang materializer | value form、bufferization、copy/pipeline legalization passes | 不能只作为字符串构造 |
| Triton TMA/descriptor 缺席 | Physical IR 与 Triton path 均无对应 form | 需要时由 Triton-local access pass 引入 | 不是 shared decision |
| provider role-based configs | provider source construction | 显式 SearchSpace/tuner declaration | config，不是通用 compiler policy |

审计原则不是“leaf 是否读取 Kernel IR”。translator 可以通过 Physical Program 中的显式 provenance 查询类型或常量；真正的问题是它是否重新遍历旧 KIR 来选择一项尚未在 provider program 中记录的结构。

---

## 18. 下一轮重构必须保持的架构不变量

### 18.1 DSL/KIR 不变量

- DSL 保留完整算法、控制、state、effect、structured operation 和数值路径。
- public surface 不用手动授权 compiler 做物理改写。
- 算法可见 region 与 physical blocking 明确分开。
- 普通顺序控制使用自然的程序语义，内部 KIR 可以显式保存 sequential fact。
- physical tile、ownership、storage、copy、pipeline 不进入 target-independent DSL。
- 不以删除结构为代价退化成纯数学表达或图级 op 集合。

### 18.2 Physical Program 不变量

- 它是唯一可执行的 lowered GPU program，不是长期旁表。
- 每个 pass 边界都有完整合法含义并通过 verifier。
- baseline 使用合法保守值，不使用 partial/unknown hole 表示尚未优化。
- 任何 pass 可以合法重写现有决定；不引入字段写权限。
- decision 写进 IR，derived facts 由 analysis 重算。
- KIR provenance 可以保留，但 provider path 不靠 KIR 重建 executable structure。

### 18.3 Pass 不变量

- 所有 target-source 结构变化必须能定位到 shared 或 provider-local pass。
- pass 依据 typed structure、def-use、reuse、lifetime、validity、device/capability，不依据 kernel 名称。
- analysis 默认保守失效，明确 preserved 才复用。
- pass 顺序与必要的重复 canonicalization 是显式 pipeline 设计。
- 不用全局 cost model 或 runtime structural search 替代 deterministic policy。

### 18.4 Provider 不变量

- provider-specific op/type/attr 与通用 physical facts 存在于同一可验证 lowering 体系中。
- Triton 不重复下层 layout/MMA/register/pipeline 决定。
- TileLang 显式 storage/copy/pipeline 通过 provider passes 形成。
- cuTile native access/collective form 在 emitter 之前确定。
- TMA 只有在 Intent 必须有意识选择该 surface 时才成为 Triton-local form。

### 18.5 Terminal emission 不变量

- translator 消费 provider-legal program，不重新选择 ownership、range、residency、copy、workspace、stage 或 structured traversal。
- source string 是 pipeline 的终端产物，不是中间优化载体。
- decorator、signature、syntax 和 wrapper 可以由 translator 生成；结构候选与 tuner config 必须已有显式输入。

### 18.6 语料与诊断不变量

- 28 个缺口必须实际接入，不以状态表代替实现。
- inventory missing、provider unavailable、compile/JIT failure、runtime failure 和 numerical mismatch 分开。
- 固定 baseline 与后续定向结果分开陈述。
- 不用旧矩阵总数证明新矩阵回归或成功。

---

## 19. 建议采用的重构顺序

这里给的是依赖顺序，不是提交清单。

### 第一层：先闭合语言语义

先审计 `partition`、`ordered`、`parallel`、`state_stream extent` 和 contract surface，明确哪些是算法可观察结构，哪些是 physical hint。没有这一步，compiler pass 不知道自己是在推导实现，还是在偷偷改写算法。

这一步不要求一次设计出最终最漂亮的语法，但必须让 public surface、frontend lowering、KIR op 与 verifier 的语义一致。

### 第二层：让 Physical Program 真正可被改写

把当前 executable skeleton 扩展成能够直接表示 execution、value、access/validity 和 structured-op realization 的 SSA program。baseline constructor 先产生完整合法的保守 program；materializer 不再把旧 Intent op 当作主要 executable source。

### 第三层：拆出 shared analyses 与 passes

先从当前已经存在且跨 kernel 生效的 policy 拆起：range/persistent reconciliation、accumulator/value flow、compact validity、lane/execution packing，再逐步让它们读取独立 def-use、reuse、lifetime 和 device analyses。

目标不是移动代码，而是让 pass 前后 Physical Program 真的不同、后续 pass 能看见并修正结果。

### 第四层：建立 provider-legal IR 与 provider passes

将当前 Triton/cuTile/TileLang materializer 中的结构选择前移为 provider form、bufferization、copy/access、primitive 和 legality passes。provider-specific facts 与 shared facts 同处可验证的 IR 链，不另建隐藏配置层。

### 第五层：收缩 terminal translator

只有当 provider program 已经完整时，才生成 source string。此时 leaf 修改若改变结构，应被视为 boundary regression；纯 source syntax 和 wrapper 修改才留在 translator。

### 第六层：闭合 inventory 与错误阶段

28 个 entry 按真实 registry 接入；每个新算法缺口先补 DSL/KIR，再走同一 Physical Program 和 provider pipeline。与此同时让错误阶段能够区分 compiler、provider JIT、launch、数值与环境，使 baseline 的失败具有可解释性。

这六层并不意味着必须一次性大爆炸式重写。可以按纵向 kernel/provider slice 逐步迁移，但每个 slice 都必须向这条主链收敛，不能新增 legacy fallback 或第二套 executable authority。

---

## 20. 对关键反驳的回答

### 20.1 “去掉 partition/ordered，会不会只剩 TE？”

如果机械删除当然会。正确做法不是删除语义，而是把算法可见 region、state、effect、control 和 structured operation 保留，把 physical blocking 移出 public surface。普通 `for` 承担顺序语义，逻辑 segment/window 保留，只有 GPU tile 被 compiler 推导。这样仍然是一门程序式算子 DSL。

### 20.2 “算法到 Triton 不是几乎确定的吗？”

低层 layout/MMA 等大量事情确实由 Triton 决定，但 program decomposition、loop structure、value realization、validity placement 和 primitive composition 不会由 Triton 从任意糟糕的上层程序中重新发现。空间比 TileLang 显式层窄，但足以造成显著结构和性能差异。

### 20.3 “既然 deterministic，和模板有什么区别？”

模板按 whole-op 选择完整实现；pass 按 typed local facts 改写可组合 IR。deterministic 只说明固定输入有一份 canonical 输出，不说明内部没有编译决策。

### 20.4 “TileLang 的 alloc/copy 都能推出来，为什么还要 pass？”

选择了 storage/primitive form 后，具体 alloc/copy/barrier 多数可以确定性 materialize；但 storage/primitive form 并不总由逻辑 op 唯一决定。前者是 lowering pass，后者是 provider realization pass。两者都不应藏在 terminal string emitter。

### 20.5 “性能现在还是靠改 leaf，说明 pass 没用？”

这说明当前 leaf 仍然拥有结构，而不是结构天然属于 leaf。最近改动中 persistent、accumulator、compact、lane packing 都能归入通用 physical policy。重构的目标正是让以后同类优化变成可复用 pass，而不是继续按 provider source 分支修补。

### 20.6 “TMA 加上不就行了吗？”

TMA 只修补一种 Triton access surface。它不能回答 region、execution、value、validity、structured op、provider IR 和 terminal boundary。需要时应加，但它不是编译器骨架。

### 20.7 “配置就是性能知识吗？”

配置包含经验值，但它不是 compiler transformation。手写 source 也能拥有同样的 autotune configs。Intent compiler 的核心价值必须来自结构 policy；config 只绑定剩余数值自由度。

### 20.8 “35 个失败是不是环境没配好？”

可能有一部分是，但当前 `compile_failed` 折叠了多个阶段，证据不足以逐项归因。也可能有 provider capability、adapter、JIT 或 generated program 问题。正确结论是错误边界需要重构，而不是先把失败统一归给环境或 compiler。

---

## 21. 仍然没有冻结的具体问题

这些问题不能靠本文凭空给出最终 API，但它们已经有清楚的判断边界：

1. 当前每个 `partition` 到底是算法可见 region 还是 physical blocking，需要逐使用点审计后才能决定替代表面。
2. 普通 `for` 如何完整承接当前 `ordered` 的 carry/effect/stop，并与内部 KIR sequential op 对齐，需要 frontend/KIR 一起设计。
3. `parallel` 在第一阶段保留显式逻辑独立性，还是同时引入 dependence inference，尚未冻结；不能在没有证明能力时默认并行。
4. state stream 的 public extent 已是 compile-time/auto，但 KIR schema 仍把固定 extent operand 称为 runtime extent；应统一内部表示与命名，并明确 physical segment chunk 的落点。
5. provider-specific IR 是作为 Physical Program 的扩展 op/attr，还是阶段性 provider dialect，需要结合当前 `PlanOps` 与 materializer 的实际改写方式决定。
6. shared structured-op realization 与 execution/value passes 的最终顺序可能需要固定重复 refinement；本文只冻结责任槽位，不伪造一份未经实现验证的完整 pass list。
7. Triton 哪些 workload 必须显式 descriptor/TMA surface，哪些可交给下层 compiler，必须按真实 provider capability 和目标 source contract 判断。
8. 35 个失败的逐项根因仍未由当前宽泛错误标签证明。本轮不设计探针或对照测量。

这些未决点不会动摇主骨架：语义属于 DSL/KIR，物理决定属于可验证 passes，provider structure 属于 provider IR/pass，terminal leaf 只做 translation。

---

## 22. 对后续实现的最终判据

后续每一项重构或性能修改，都应当能够回答下面六个问题：

1. 它保持的 Kernel IR 算法语义是什么？
2. 它决定的是 execution、value、access/validity 还是 structured-op realization？
3. 它读取了哪些 typed semantic、analysis、device 或 provider facts？
4. 这个决定写入了哪一层可验证 IR，后续 pass 能否看见并改写？
5. 它怎样改变最终 provider program 的结构，而不是只换一个 config？
6. 为什么这项规则能够跨 kernel 结构复用，而不是一次 whole-op 模板命中？

如果一个修改回答不了这些问题，却在 leaf 中新增 loop、tile、buffer、copy、workspace、stage 或 primitive branch，它就是继续扩大当前边界泄漏。

如果一个 public DSL construct 只能回答“它让 compiler 可以这样优化”，却说不出新的算法可观察语义，它就是不该暴露给作者的物理授权。

如果一个 provider-specific structure 在 source string 生成时才第一次出现，它就是尚未完成的 provider pass。

最终，成熟 Compiler Pass V2 应当让我们对任意一个生成 kernel 清楚地说出：

> 作者表达了什么算法；Intent 抽掉了哪些 target-program decisions；每项决定由哪一个 deterministic shared/provider pass 补回；pass 读取了什么真实事实；决定如何进入可验证的 Physical Program；terminal translator 如何忠实投影；剩下哪些数值和机器决定交给 provider tuner 与下层 compiler。

这才是一条真正的算子编译器主链，也是接下来重构不能再偏离的边界。

---

## 23. 主要事实依据

### 当前项目

- `report/compiler-space-v1.md`：V1 的有限编译空间、性能来源与 leaf 边界。
- `report/compiler-pass-v2.md`：executable Physical IR、shared/provider passes、verifier 与 terminal translator 的成熟态定义。
- `report/compiler-pass-v2-five-round-progress.md`：五轮真实实现、固定矩阵、定向 A/B、inventory 与当前缺口。
- `doc/dsl/model.md`、`doc/dsl/domains-and-control.md`、`doc/dsl/tensor-flow.md`：当前 DSL/KIR 语义合同。
- `lib/Target/GPU/Transforms/Passes.cpp`、`Construct.cpp`、`Build.cpp`：当前 construct/verify pipeline 与 monolithic physical construction。
- `include/intent/IR/PlanOps.td`、`lib/IR/PlanOps.cpp`：当前 Physical Program、decision ops、TargetProgram source string 与 verifier。
- `lib/Target/Common/Lowering/Driver.cpp`：materialization 与 terminal translation 边界。
- `lib/Target/Triton/Lowering/Program.cpp`、`lib/Target/CuTile/Lowering/Program.cpp`、`lib/Target/TileLang/Lowering/Program.cpp`：当前 provider-local 重判、tuner 与 source construction。
- `examples/repro/v2/measurement.py`、`python/intent/compiler/pipeline.py`、`python/intent/compiler/toolchain.py`：compile、first run、provider JIT 与失败包装边界。
- `source/triton/README.md`、`source/cutile/README.md`、`source/tilelang/README.md`：119/91/28 inventory 口径与缺口。

### 参考编译器

- `/home/kingdom/phdworks/ref/triton/third_party/nvidia/backend/compiler.py`：TTIR→TTGIR 与 NVIDIA provider pass pipeline。
- `/home/kingdom/phdworks/ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`：layout encoding 作为 IR/type fact。
- `/home/kingdom/phdworks/ref/tilelang/tilelang/cuda/pipeline.py`：TileLang CUDA provider pipeline。
- `/home/kingdom/phdworks/ref/tilelang/tilelang/language/loop.py`、`copy_op.py`、`gemm_op.py`：`T.Parallel`、`T.Pipelined`、`T.copy`、`T.gemm` 的真实语言与 lowering 表面。
