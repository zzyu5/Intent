# CPU executable programs

## 1. Execution family 与 provider

CPU construction 在任何 GPU construction 之前消费 immutable canonical KIR。GPU 与 CPU 共享作者语义及可复用分析，不共享 GPU program id、warp 或 fragment ownership。CPU 程序可以由 Mojo、Weft 等 provider 消费；RVV 是 CPU hardware capability，不是另一套从 KIR 出发的 Intent execution family。

```text
canonical KIR
    → shared executable CPU Program / analyses / passes
        ├─ Mojo legalization / serialization → Mojo / LLVM
        └─ Weft legalization / serialization → Canonical Weft IR → Weft compiler
```

Provider 与 hardware 分开选择。x86 SIMD、RISC-V RVV、可选 matrix extension 不改变 Intent source/KIR 的算法、dtype、控制或 effects。具体能力参与 legality，不能把 AVX2/AVX512 或未知运行时向量长度变成整个 CPU family 的必要条件。

CPU 模型是非 SIMT 的完整计算程序，不以硬件 vector 或 matrix tile 作为作者根对象。一个 kernel 调用中的任务分区由 CPU 程序与 host/runtime 承载；scalar、vector、register replica、matrix fragment 和局部存储是其实现表示。

## 2. 唯一程序与 typed IR

CPU dialect 拥有本 family 专属的 program、ABI、轴/访问关系、任务/资源归属、数值义务、capabilities 和 physical parameters 的 typed 表达及 verifier。合适的 `func/arith/math/scf/memref/vector/linalg` operations 可以复用；不能以标准 dialect 已验证为由跳过 CPU 自有语义，也不重复定义无必要的标准算术。

CPU dialect 的 IR、current-program analyses 与 transforms 在同一职责边界内组织；provider legalization/serialization 和 native runtime 分属各自模块。模块数量不代表 IR 层数，混用标准 operations 不产生另一份 executable authority。

Construction 产生完整独立程序，至少显式承载：

- scalar/shaped values 的 dtype、logical axes、result/broadcast relations；
- 外部 views、访问坐标、有效域、fill、读写方向和 alias；
- task work domain、captures、唯一写入、依赖和完成边界；
- physical loops、bounds、steps、tails 与 carries；
- reduce/contract 的输入、轴、identity/combine、accumulator 与数值保持条件；
- 本层形成的 blocking、reuse、materialization、资源 owner、初始化与 lifetime；
- 已声明的 target facts 和约束实际 types/loops/accesses 的参数。

后续 transformations 只改这份程序。Origin、分析缓存和 tuning 结果不参与执行解释；provider 或 runtime 不回读 KIR、解析 role 名称或从 shape 猜回丢失的结构。

## 3. Shaped computation 与机器表示

共同 CPU value 保留 logical element/axis identity。Shaped computation 不等于已选 AVX/RVV register；可以沿 physical traversal、vector、多个 accumulator 或合法 matrix 表示实现同一数值 operation。

Reduce 和 contract 在消费者仍需要其结构时保持显式 typed schema，不先无差别展开为 scalar loops。细化到 SIMD loop、归约合并图或 matrix operation 是有 legality 的实际 rewrite，而不是后端重新发现算法。

矩阵表示不是另一种算法。其输入/输出 dtype、shape、axes、accumulator、tail、packing/handoff、资源和 effects 必须保持对应的 closed operation。没有合法 matrix realization 时只能使用本次明确选择的合法表示或诊断 unsupported，不在 emission 失败后隐式换程序。

共享层保存本层负责的完整计算、访问与复用结构。具体 matrix fragment、ISA-specific layout、配置状态及机器资源由需要它们的 provider-local lowering 或外部 compiler 拥有；只有真实 consumer、表示差异和独立 legality 要求时才引入 local extension。不得为将来可能有的硬件增加空字段或占位 executable path。

Scalar/vector/matrix 协作不意味着存在 Ascend 式两个执行上下文。同步 CPU 指令流可通过 SSA、内存 effects、显式转换与资源生命周期表达；异步 engine 或跨上下文同步只能由真实能力和完整依赖语义引入，不能默认添加 barrier/queue 协议。

## 4. Access、任务与资源

ABI 保存 element type、rank、静态 extents、动态维度身份、offset/strides、access 与 alias；native lowering 机械展开为 pointer/descriptor/scalar 参数。资源与 interface 的关系由 typed schema 验证，不能由 serializer 与 runtime 各自猜字段含义。

Contiguity、alignment 和 disjointness 若是所选实现的前置条件，必须在 physical entry 中明确，并由实际 view 验证；它们不自动成为作者承诺。不支持的 view/alias 组合明确拒绝，不暗中 copy、假定 noalias 或覆盖尚需读取的输入快照。

Task partition 由 workset、dependence、访问域和唯一写入证明形成。程序保存每个任务的工作范围与 captures，并在 host-visible 调用返回前完成依赖与 join。线程池只执行已经声明的任务，不替编译器决定算法或复用结构。

局部资源保存大小、对齐、owner、初始化和 lifetime。Forwarding/rematerialization 必须保持坐标、dominance、effect order 与来源值稳定性。Invocation-local packing 是同一值的表示，不自动成为跨调用缓存或另一份 workspace ABI。

## 5. Analyses、passes 与 verifier

CPU current-program analyses 只从当前 IR 重算 coverage、axis/access relation、dependence、reuse、buffer lifetime、资源估计和 provider eligibility。相关 mutation 后失效，只有证明保持的结果可以复用。

每个完整 transformation group 声明输入事实、legality、改写的 IR、保持语义和失效 analyses，并在完成后验证同一程序。内层 repair helper 不是可独立运行的完整 pass；注册名、旁表标签或字符串变化不构成优化能力。

共同变换包括：

- producer/consumer fusion、SSA reuse、合法 rematerialization 与中间存储消除；
- task partition、grain 和覆盖关系的细化；
- 保持 logical axes 的 blocking、vector/structured computation 与 tail realization；
- 保持数值条件的局部 reduction/accumulator graph 和 consumer 遍历；
- contraction 的 cache/register blocking、多行/多列累加与数据供应；
- 当前程序中显式的 packing、转换和资源生命周期维护。

Provider legality 检查每个 operation/type/access/interface 是否有合法表示，local extension 是否完整。进入 serialization 时不留需要 emitter 自行作出的执行决定；verifier 只检查、不修程序。

## 6. Capabilities、参数与选优

Hardware/provider capabilities 有 typed schema，包含实际消费者所需的 dtype、向量粒度、资源、memory forms 和可选矩阵表示条件。Passes 消费 feature/资源事实，不匹配设备名或把某个固定宽度集合当作所有 CPU 的定义。

Physical bindings 直接约束 task grain、block/register extents、accesses、向量 realization 或明确的 provider options。共同与 provider 参数由各自 consumer 拥有，有限候选文件与对应 transforms/provider 相邻；参数角色与合法域必须可追溯到真实程序实体。

配置入口保留有限 override 能力。候选的字段/schema 由实际支持的具名角色定义并验证，不使用 kernel 名称、算法开关、第一份 GPU 默认表或完整笛卡尔积。Width 的改变不能替代缺失的 reuse、blocking 或 accumulator 结构。

每个候选是完整合法 binding，在同一语义程序上产生具体 IR 与编译产物。Tuner 只测合法候选并保存 winner，不发明 source tree，不把失败后的隐式改选留给 emitter。产物缓存与 winner 缓存分开，其身份包含各自实际依赖的 specialization、view facts、hardware/provider/options 和 bindings；动态 shape 可以由 ABI 表达。

## 7. Provider lowering 与 serialization

Mojo 接收已形成的任务、循环、shaped/向量 operations、访问和资源。其 legalization 形成支持的 API/type/native ABI；serializer 只机械拼写。Mojo/LLVM 继续负责机器 lowering、寄存器分配、指令选择和调度，不能反过来承担共同 CPU 程序中缺失的任务或复用结构。

Weft 的外部输入为 Canonical Weft IR。Intent 将当前 CPU 任务内计算的 typed axes、operations、views、控制和生命周期映射到该输入；显式保留与外部任务调度的接口，不向 Weft 语言引入隐式 hart/grid。Weft 的 RISC-V physical layout、RVV/IME fragment、packing、schedule/resource 和 intrinsic emission 仍由 Weft compiler 完成。

外部 provider 内部的 IR 层次不改变 Intent 的程序权威。Provider 不能按 Intent kernel 名选择整算子 std tree，不能回到 KIR 重建任务、循环或数值结构；不支持的接口/数值组合在具有足够事实时诊断。

普通浮点保持 Intent 默认语义；Mojo 全局隐式 contraction/fast-math/FTZ 不直接继承。只有对应 operation 允许的 FMA、近似或重结合可以使用。若任务运行时改变 FP environment，合法化与 runtime 须维护所需语义及调用方状态；矩阵扩展不能隐式改变外部 dtype 或中间精度。

## 8. Artifact 与执行边界

可执行 artifact 的 ABI 包含所需 views/scalars 和明确的私有资源要求。初始化、编译、加载与普通调用分离；一个 host-visible entry 的所有内部任务在返回前同步完成。Runtime 只绑定资源和调度已声明的任务，不执行 Python/Torch 算法替代品或隐藏的输入 repacking 缓存。

只生成产物的模式与 native materialization 分开。生成的 provider source/IR 和接口不是可调用 artifact；未接入的调度、下层编译或设备执行保持明确，不以 source generation 声称运行支持。

Native 算子计时包含该次调用所需的 packing、内部 materialization、任务派发和同步；编译、调优、加载及外部输出分配独立处理。生成能力、native 执行、数值容差和相对成熟库的性能分别陈述，运行 case、具体阈值和性能结果不进入本设计规格。
