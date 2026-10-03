# IntentDSL v1 基础设施收口路线图

调查日期：2026-10-03。实现基线：`main / 07911f53`。本次原位替换旧路线图，只更新调查和计划；未修改 compiler、正式规格、作者程序或实验结果，未编译或运行 benchmark。

**目标：先完成 v1 的编译器与产品基础设施，使后续性能轮次能够在稳定接口内新增、改进可复用 pass。基础设施重构和性能优化严格分轮。** 本文的 v1 是能力与架构完成状态，不新增软件版本号或承诺所有未来硬件、语言扩展永远不改基础层。

## 1. 已确定的方向

用户已经明确：当前发现的结构问题必须先处理完整；不能一边做性能，一边临时改 IR 合同、construction、公共分析、provider 或 runtime。优化能够沉淀，要求其读取的事实、改写的程序和调用的接口先稳定。

本轮审查得到的结论是：主干已经建立，剩余结构工作可以集中完成。当前有五个明确定位的结构缺口，另有一项历史数值失败需要结算，以及有限的产品说明与发布事项。它们组成下文的 v1 剩余计划；不把已完成的安装、ABI、配置、CSE 和执行模型分层重新列为建设任务。

### 1.1 v1 的完成状态

在现有语言、执行模型和已声明目标能力范围内：

1. 当前 IR 独立保存执行事实；正常 construction、文本重读和阶段续编译遵守同一合同。
2. 公共分析、实际改写与 verifier 对同一事实采用一致定义，分析结论可以由已有变换接口兑现。
3. 一个 pass 可以通过明确入口完成改写、关系维护和分析失效处理；调用者不需要知道它内部的修复顺序。
4. 公共调用、ABI、配置绑定、产物与 provider/runtime 边界稳定；增加普通优化不牵动这些层。
5. 已发现的同责旧路径已经迁移并删除；CPU、GPU、DSA 各自保留真实的执行模型差异。
6. 开发者能从目录和已有贡献指南找到事实查询、合法性证明、改写与目标消费入口。

**v1 的支持范围按语言构造、dtype、effects、layout 和目标能力描述，不能按“通过了哪些算子”定义。** 合同允许且目标可以实现的程序，不能因为未命中特定写法就被归为作者问题。真实目标能力限制继续明确说明。

### 1.2 架构轮与性能轮的修改边界

| 内容 | v1 基础设施收尾轮 | v1 完成后的纯性能轮 |
|---|---|---|
| DSL/KIR 语义、公共调用行为 | 保持现有正式规格；真正设计变化另行确认 | 冻结 |
| 执行事实的 IR 表达、公共 op/interface、公共分析合同 | 处理本路线图的已证缺口，优先复用现有 carrier | 冻结 |
| 公共 schema 运输、重放、descriptor、effect/lifetime 机制 | 完成统一并迁移消费者 | 使用既有接口 |
| construction、ABI、provider/runtime、产物格式、安装链 | 只做本轮明确需要的收口 | 冻结 |
| pass 的匹配、变换、收益策略 | 迁移现有能力，修复合同不一致；不混入新的性能策略 | 主要修改对象 |
| pass 私有分析、代价模型、既有参数域内的策略数据 | 保持原策略，除非兑现合法性所必需 | 可随 pass 修改，不承担 IR 外的执行语义 |
| benchmark 与证据 | 原入口确认能力与正确性保持；时间变化如实记录 | 原入口比较物理结构与完整调用性能 |

纯性能轮若发现基础层缺少必要能力，应把该问题交回独立基础设施任务，再恢复优化；不能以 pass 私有 metadata、serializer 特例或 runtime 分支绕过。新增 provider、SDK/ABI 适配、新语言构造也属于独立演进，不混入纯性能轮。

Pass 私有分析只推导当前 IR 的临时匹配、合法性或代价事实，不能另行定义公共轴、effect、lifetime 合同；IR 改写后按依赖失效，选定的执行决定仍写回已有 IR。把公共证明复制到私有目录不属于允许的性能修改。

架构收尾可能自然改变生成程序或耗时，这不要求人为维持低效代码；但本轮不以加速比选取额外任务，不增加候选、调整 tile 或改变算法来修饰结果。

## 2. 从旧待办中移除的已完成工作

以下只说明本轮所依赖的已有基础，不再作为待办展开。旧调查和当时观察保存在 Git 中。

| 已有基础 | 当前依据 |
|---|---|
| 数值许可、online 开关、remarks 与普通后续路径 | [CompileOptions](../python/intent/compiler/options.py):17–49；[RealizeOnlineReduction](../lib/Dialect/GPU/Transforms/Reduction/RealizeOnlineReduction.cpp):420–471；[正式合同](../doc/compiler/passes-and-analyses.md):18–22 |
| execution-group owner、普通坐标计算、op canonicalization 和标准 CSE | [GPUOps.td](../include/Intent/Dialect/GPU/IR/GPUOps.td):78–109；[RefineProgramMapping](../lib/Dialect/GPU/Transforms/Mapping/RefineProgramMapping.cpp):156；[EliminateCommonValues](../lib/Dialect/GPU/Transforms/Value/EliminateCommonValues.cpp):192–223 |
| GPU 完整变换边界、公共轴关系与 schema 运输 | [Passes.cpp](../lib/Dialect/GPU/Transforms/Passes.cpp):77–83；[FragmentOpInterface](../lib/Dialect/GPU/IR/FragmentOpInterface.cpp):317–420；提交 `94809e84`、`0ce1875a`、`959f2a10`、`ca8cd919` |
| CPU 单一 pipeline、实现选择和输入供应 | [CPU Passes](../lib/Dialect/CPU/Transforms/Passes.cpp):29–81；[Implementation](../include/Intent/Dialect/CPU/Transforms/Implementation/Implementation.h):102–166；[ImplementationInputs](../include/Intent/Dialect/CPU/Transforms/Implementation/ImplementationInputs.h):13–33 |
| DSA 单一 construction 和当前 IR 上的 collective/matrix realization | [KIRToDSA](../lib/Conversion/KIRToDSA/KIRToDSA.cpp):7–21；提交 `d0d03657`、`ac57d279`；旧 whole-source matrix 分叉已删除 |
| 公共 product、标量数值、存储、整数范围、逻辑尺寸和索引物化 | 提交 `7b964318`、`0954dd9d`、`a47db332`、`aafcb750`、`d74877d8`、`c71a2845`；[Analysis](../include/Intent/Analysis/)、[Conversion](../include/Intent/Conversion/) |
| 候选约束、资源来源、实际配置检查与 workspace ABI | [GPU Resources](../lib/Dialect/GPU/Analysis/Resources.cpp):325–367；[配置处理](../lib/Dialect/GPU/Transforms/Configuration/Resources.cpp):22–61；提交 `a1f52433`、`2531f421`、`b7249fe2` |
| 公共 Out/InOut 返回、named alias、prepared 调用与产物合同解析 | [authoring](../doc/dsl/authoring.md):24、129；[invocation](../python/intent/runtime/invocation.py):25–56；[ProgramContract](../python/intent/runtime/contract.py):73–119；提交 `95b4557d`、`22906555` |
| 作者 backward、mutable Torch 调用、native 编译/调优/执行分层 | [forward/backward 示例](../examples/softmax_forward_backward.py):11–27；[Torch adapter](../python/intent/runtime/torch.py):27–93；提交 `a4dc11b1`、`095a051a`、`3aa4151b` |
| 安装包、独立工具/MCP、源码包构建与可下载 CI 产物 | [pyproject](../pyproject.toml):15–28；[分发工作流](../.github/workflows/distribution.yml):29–100；[现有构建与安装说明](../environment/README.md):104–122 |

这些是当前源码和提交可确认的完成状态，不表示本次重新执行了所有平台验证。历史结果继续按各实验组记录解释。

main 的 GPU 产品只推进 Triton/cuTile。`archive/tilelang-backend` 分支仍存在；旧 TileLang 后端任务和 RMSNorm 失败不进入 main 的 v1 待办。外部 `../ref/tilelang` 继续作为职责与机制参考。

## 3. 本次调查范围与判断方法

核查覆盖：canonical/shared 分析、GPU/CPU/DSA 的 construction 和 transforms、helper/控制/存储关系、provider legalization/serialization、公共 ABI/runtime/产物，以及已有安装和框架入口。源码理解使用现有 codegraph，并对未覆盖代码、规格和参考实现补充直接阅读。

正式依据为 [doc/index](../doc/index.md) 及 compiler 的 [边界](../doc/compiler/README.md)、[passes](../doc/compiler/passes-and-analyses.md)、[CPU 程序](../doc/compiler/cpu-program-ir.md)、[参数](../doc/compiler/physical-parameters.md)、[目标 lowering](../doc/compiler/target-lowering.md)。本路线图不改写这些规格。

本地参考快照：Triton `ef08c7f`、TileLang `f354430`，均为 2026-08-19 提交。MLIR 对照使用本地 LLVM 20.1.8 源码；引用其仓内相对路径，实施环境可从 `/tmp/intentdsl-llvm-20.1.8-src/mlir/` 定位。参考说明职责和实现机制，不把本地快照称为外部最新发布，也不照搬下层 layout/ISA。

发现分三类：

- **已证结构缺口**：当前源码可定位的职责分裂、独立 IR 验证遗漏或表示形式限制；必须在 v1 中处理。
- **待结算的已有问题**：已有失败记录，但本次没有重新运行或证明根因；必须得到明确结论，不能直接算修好或算新的架构 bug。
- **性能与能力扩展**：收益策略、新设备和新表达范围；排在 v1 之后，不能借它们扩大基础收尾。

## 4. 已定位的五个结构缺口

| 编号 | 问题 | 后果 | 责任范围 |
|---|---|---|---|
| S1 | GPU helper 的逐 lane 资格、标量化、升维和重建判断分散 | 同一合法 helper 因变换入口不同得到不同支持；新表达需同步多个名单 | GPU Analysis、Value/Reduction transforms、两个 provider callback 消费者 |
| S2 | CPU producer payload 存在三套资格和重放实现 | scope、读取稳定性、控制流、坐标绑定和 clone 的支持域漂移 | CPU 公共分析/改写机制及三个 fusion/replay 消费者 |
| S3 | DSA 协作执行的变换和 verifier 各自证明 uniform/control | 变换认可和最终验证依据不一致，供数变换重复识别算术树 | DSA 执行关系分析、协作供数与 IR verifier |
| S4 | Weft task view capture 仍以特定 producer 形态识别来源 | 可表达的 descriptor 可能因标准 IR 写法或 forwarding 被拒 | CPU 存储/控制事实与 Weft capture 正规化边界 |
| S5 | DSA full-extent 入口义务缺少 IR 侧完整验证 | serializer 假定属性存在；独立 IR 的非法入口事实可能迟至发射或 runtime 才暴露 | DSA 入口约束查询/verifier、BANG serializer 消费 |

这五项是当前审查确认的剩余结构工作，不是根据文件大小或历史怀疑推导出来的项目。下面分别给出完整收口范围。

### S1：GPU helper 的证明与转换同源

**证据。** [ReductionValues.cpp](../lib/Dialect/GPU/Transforms/Reduction/ReductionValues.cpp):137–159 用 opcode 白名单判断能否 lift；[ValueMaterialization.cpp](../lib/Dialect/GPU/Transforms/Value/ValueMaterialization.cpp):173–228 用另一名单 scalarize，另行判断 Broadcast/Reshape；[ExecutionSchema.cpp](../lib/Dialect/GPU/Transforms/Value/ExecutionSchema.cpp):120–122、185–205 又描述逐元素运算。实际消费者包含多轴 reduction、嵌套 reduction、access/reduction composition，以及 [Triton callbacks](../lib/Target/Triton/Transforms/Collective/Callbacks.cpp):37–68 和 [cuTile compute forms](../lib/Target/CuTile/Transforms/Compute/ComputeForms.cpp):234–273。

**要完成的修改：**

1. 从实际 helper 的 formal/result、record 字段、operand slot 与 `FragmentOpInterface` 关系证明逐 lane 对应。Effect-free 不自动等于逐 lane，Scan prefix、归约成员轴、captures 和空 identity 各守原语义。
2. 将资格查询与可实施转换连接起来，复用 `cloneWithSchema`、`rewriteClonedPhysicalTypes` 等现有机制。Analysis 不产生第二份可执行图，transform 必须生成真实 SSA。
3. scalarize、lift、相关 projection 迁移到共同语义入口；保留各消费者的目标类型、reassociation 权限和收益选择。
4. 精确 binary combine 查询返回足够的原操作和 operand 对应事实，保留顺序、dtype 与数值属性；简单 kind 匹配只作为确实需要它的消费者派生查询。
5. 删除重复的语义白名单和只认某种等价投影拼写的局部递归。Triton/cuTile 的真实 native primitive 能力、identity/NaN 处理仍归 provider。

**现有问题的边界。** [Replay.cpp](../lib/Dialect/GPU/Analysis/Replay.cpp):96–114 目前只返回 BinaryOperator，并接受两种参数顺序。已检查其消费者：按 kind 重建的路径限于交换类操作；其它路径 clone 原 helper。本次未证明数值错误，不把接口信息不足写成已发生 wrong-code。

**参考。** Triton [Ops.cpp](../../ref/triton/lib/Dialect/Triton/IR/Ops.cpp):580–621 验证 combine 的 `2N→N` element 合同，:661–679 返回真实 combiner，参数反序只在 `IsCommutative` 成立时接受；[Utility.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Utility.cpp):584–640 区分 Elementwise 语义与形状关系。Intent 的输入更高，需要完成 fragment helper 到目标 callback 的转换，但不应在每个消费者重写一次语义判定。

**关闭条件。** 对相同目标 schema 和转换前提，已支持 helper 的资格证明与实际转换域一致；新增普通逐元素表达不需修改多套名单；现有 reduction/scan/provider 消费者全部迁移，旧实现删除。用原 GPU registry 的 `flaggems_batch_norm_training`、`flaggems_max_pool2d_with_indices`、`flaggems_cumsum`、`fused_softmax` 中实际相关入口检查能力保持，不预写性能收益，也不要求本轮支持任意纯 region 向量化。

### S2：CPU producer payload 的证明、重放与控制保持

**证据。** [IntegerSources.cpp](../lib/Dialect/CPU/Transforms/Structure/IntegerSources.cpp):24–109、[FuseStructuredComputations.cpp](../lib/Dialect/CPU/Transforms/Structure/FuseStructuredComputations.cpp):24–75、111–209、[FuseIntermediateBuffers.cpp](../lib/Dialect/CPU/Transforms/Storage/FuseIntermediateBuffers.cpp):158–235、299–398 分别维护 producer 资格、依赖递归、读取稳定性与 clone。已共享 StorageAnalysis，但 payload 支持域仍受当前阶段和局部代码限制；其中多处直接拒绝带 region 的生产者。

**要完成的修改：**

1. 建立一份 CPU payload replay 合同，明确原执行位置、目标插入位置、scope/frontier、已绑定值、外部 SSA 可用性和读取稳定性；复用现有 dominance、Storage、ControlFlow 分析。
2. 同一合法域对应同一重建机制：控制区域整体保持、captures 显式绑定、clone 后结果通过 IRMapping 返回。不得把 masked load 的地址或读取提前到无效分支之外。
3. 隐式 linalg input reads 和 payload 内显式 reads 都进入证明；原子、未知 effects、不能证明稳定的快照不能作为普通纯值重放。
4. 三个消费者分别提供 linalg indexing-map 组合或 SCF 坐标绑定，保留自己的融合收益、重算预算和 scalar/vector 实现选择；公共层不接管调度。
5. 迁移三个消费者并删除重复 proof/clone 递归。保持 CPU ReduceOp 现有 scalar combine 合同；[CPUOps.cpp](../lib/Dialect/CPU/IR/CPUOps.cpp):130–138 同时禁止 effects 和嵌套 region。公共 replay 可以在允许的 generic/SCF 位置保留纯条件控制，不代表可以把读取或纯 scf.if 塞进 Reduce combine；消费者的表示资格仍由当地检查。

**参考。** LLVM 20.1.8 `mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`:46–72、138–176、216–247 分开处理坐标组合、融合资格和 index 重绑；`mlir/include/mlir/Analysis/SliceAnalysis.h`:24–54 明确 slice scope/frontier。Triton [RemoveLayoutConversions.cpp](../../ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp):895–959 将 dominance 重用、backward slice 和 rematerialization 资格分开。借用职责边界，不复制另一套 CPU scheduler。

**关闭条件。** 三条已有路径使用同一基础合同并保持条件访问、快照、dtype 和执行顺序；失效分析不跨改写保留；不同层的表示适配留在各自模块。原变长卷积的 masked 中间张量是定位该分裂的实例，删除多少物化、选择怎样向量化属于后续优化任务，不作为本轮必须达到的加速目标。

### S3：DSA 协作执行关系在 pass 与 verifier 间一致

**证据。** [CollectiveGather.cpp](../lib/Dialect/DSA/Transforms/CollectiveGather.cpp):52–120 自行递归分析 uniform/control/variation；[DSAOps.cpp](../lib/Dialect/DSA/IR/DSAOps.cpp):583–602、631–643 为 collective/barrier 合法性另写 uniform。[LocalValues.cpp](../lib/Dialect/DSA/Transforms/LocalValues.cpp):255–262 还自行查询相对 row-IV 的依赖，需检查能否复用同一事实。该文件 :325–329 的 stride 系数匹配，以及 [MatrixSupply.cpp](../lib/Dialect/DSA/Transforms/MatrixSupply.cpp):20–29、60–87 的 slice/grid 匹配属于物理选型，不能仅因也含乘除就认定同责。

**要完成的修改：**

1. 在 DSA family 内集中表达当前 task/group/local 坐标下的一致性和依赖查询，使用现有真实 SSA 与 GroupId/LocalId/SCF；不新增旁路执行计划。
2. 基础整数表达与范围复用公共 IntegerRelations/IntegerRanges；只读来源、存储稳定性复用 BufferStorage。参与者和同步 scope 留在 DSA，不能混入跨 family 标量算术规则。
3. CollectiveGather 的资格与 group 表达式物化、barrier verifier 使用同一定义；其它消费者逐项确认，只迁移确属同一事实的判断。
4. 明确未知、可证明一致和依赖本地参与者的区别；未知不能被当作 uniform。在现有支持范围内，控制合流、loop carry 与 memory read 的依据闭合；不要求为本次收口新增全部 SCF 数据流能力。
5. 删除已确认同责的 uniform/control/依赖证明。保留 MLU370 的四参与者合同、SRAM/WRAM 约束、slice/grid/stride 的物理匹配及各 pass 的策略。

**参考。** Triton [AxisInfo.h](../../ref/triton/include/triton/Analysis/AxisInfo.h):216–244 将分析放在共享 dataflow 框架；[AxisInfo.cpp](../../ref/triton/lib/Analysis/AxisInfo.cpp):1269–1338 集中 visitor、join 与 SCF IV 处理。Intent 不照搬 GPU lane 属性，而是让自己的 DSA 执行事实也拥有唯一证明入口。

**关闭条件。** CollectiveGather 与 verifier 共用一致性查询，其它确实查询同一事实的消费者全部迁移；实际 group rewrite 可由 verifier 独立检查。原 `paged_gqa_decode`、`dense_gemm`、`relu`/`fused_softmax` 选择受影响者确认现有能力；不把增加协作覆盖率或更快供数作为架构完成条件。

### S4：Weft task descriptor 的来源与正规化边界

**证据。** [Weft Views.cpp](../lib/Target/Weft/Transforms/Views.cpp):30–89 的 CaptureReifier 沿私有 operation 名单追溯，view 链遇 loop-carried/opaque descriptor 在 :82 统一拒绝，尚未区分可证明的 SCF forwarding 与真正 opaque 来源；:103–168 的 queryAxisView 要求 reinterpret 的 source 恰为 ExtractStridedMetadata.basebuffer，并限定现有 axis permutation/unit-axis 能力。收口对象是当前目标表示域内可证明的等价形式，不能把所有 opaque 或不支持布局的拒绝都当作语法缺陷。

**要完成的修改：**

1. 以现有 CPU Storage origins、标准 view operands/metadata 和精确 SCF forwarding 作为来源依据，区分“无法证明来源”和“目标无法表达布局”。同一 storage origin 不等于同一 view；控制合流必须逐 incoming 证明 base、offset、sizes、strides、owner 与 lifetime 可由当前目标表示，不能仅凭 alias root 重建 descriptor。
2. 在 task capture 边界把可证明的 base、offset、shape、stride 正规化为当前目标已能消费的 descriptor，保留实际 allocation 和借用 lifetime。
3. 让 host capture、task ABI 与目标 view lowering 消费一致的事实；删除 CaptureReifier 独立的来源资格递归。
4. 保持 Weft 的 Canonical Weft IR 路线和已有 axis permutation/unit-axis 实现范围，不强制经过 Mojo SIMD，不把未知 stride 默认为 contiguous。

**参考。** LLVM 20.1.8 `mlir/lib/Conversion/MemRefToLLVM/MemRefToLLVM.cpp`:1029–1093 从标准 reinterpret operands 构造 descriptor，而不要求固定的定义链；:1337–1344 明确其它 view 的正规化责任。TileLang [CPU pipeline](../../ref/tilelang/tilelang/cpu/pipeline.py):57–87 将 storage、memory verification、host/device 与 ABI 分阶段处理。

**关闭条件。** 现有可表示布局不因等价 descriptor 定义形式或可证明 forwarding 被拒；真正不支持的布局明确诊断。沿已有 Weft/RVV/IME 原入口完成相关 native 路径；设备不可用时保留待完成状态，不把 source generation 当作设备运行。

### S5：DSA 入口容量义务进入完整 IR 验证

**证据。** [Construction.cpp](../lib/Conversion/KIRToDSA/Construction.cpp):234–258 的 requireFullExtent 记录实际公共维度义务，:144 写入 `intent_dsa.full_extent_dimensions`；[BANG Serializer.cpp](../lib/Target/BangC/Serialization/Serializer.cpp):78–89 直接读取该 DenseI64ArrayAttr。现有 DSA `verifyProgram` 检查 public interface/configuration，但未核这项属性；[BangCFacts](../python/intent/runtime/bangc/contract.py):17–25 才检查其公共维度身份，[runtime](../python/intent/runtime/bangc/program.py):261–264 执行容量检查。

这属于静态确认的验证遗漏。本次未构造非法 IR 反例、未运行 crash 复现。**DenseI64ArrayAttr 本身是正常 MLIR 属性；问题是生产、验证和消费没有形成完整合同，不是它的名字或外观。**

**要完成的修改：**

1. 把该义务纳入现有 DSA entry requirements 的正式查询与验证边界：存在性、类型、唯一性、合法公共维度身份和已绑定容量来源均明确。动态输入是否超过容量仍由 runtime 检查，不要求静态证明未知运行值。
2. 优先保留并验证已有 carrier，或在确有职责收益时并入已有 EntryRequirementsAttr；不为一个字段发明新的 schema/plan。
3. 正常 construction 显式写出义务，空集合也明确；独立 current IR 的缺失/非法事实在 verifier 拒绝，不能在 serializer 填默认值。
4. serializer 只读取已验证结果；runtime 继续约束调用者实际输入。无需为了本任务重做 NativeABI slots、公共 View 语义或整个 runtime metadata 格式。
5. 保留现有 integer-range 对内部 bounded extent 的消解，不把局部 capacity 或同名 dimension 猜成外部调用义务。

**关闭条件。** 从 construction 和 shared IR 续编译进入的程序接受同一验证；发射不再承担首次发现缺失入口事实的职责。参考既有 CPU [EntryRequirements](../include/Intent/Dialect/CPU/IR/CPUAttrs.td) 与 [CPUBackends.cpp](../lib/Compiler/CPUBackends.cpp):66–94 的消费边界，以及 Triton [compiler.py](../../ref/triton/python/triton/compiler/compiler.py):466–488 对实际加载资源的最终检查：编译事实的结构验证和调用时资源检查各自完整。

## 5. 三个完整里程碑与实施顺序

五个问题按结果组成三个里程碑，不把每个 helper 提取或每个提交称作里程碑。

| 里程碑 | 包含工作 | 完成后得到的能力 |
|---|---|---|
| V1-A：执行边界完整 | S4 Weft descriptor + S5 DSA 入口义务 | 现有物理程序的来源、入口事实与目标消费能独立闭合，不依赖固定 construction 拼写 |
| V1-B：可复用变换机制完整 | S1 GPU helper + S2 CPU replay + S3 DSA 执行一致性 | 证明、实际改写与验证一致；新增普通优化通过既有接口实现，不再复制基础合法性机制 |
| V1-C：集成、正确性与产品收尾 | 下文有限集成审查、历史数值问题结算、使用说明与公开分发决策 | 清楚的支持范围和稳定扩展边界，可以结束架构建设阶段并进入独立性能轮 |

建议从 V1-A 与 V1-B 的独立部分并行推进；各 family 分文件实施，共用基础接口先明确再迁移消费者。S4 可以复用 CPU 当前 Storage/ControlFlow，不必等待 S2 的全部融合迁移。S3 与 S5 修改相邻 DSA verifier 时需协调所有权。

每个工程包都完成“统一机制 → 迁移全部已列消费者 → 删除旧路径 → 原程序能力确认”。不能只提交一个公共接口，让旧消费者继续自行判断；也不能通过大量文件移动、注释或 CSV 更新凑重构规模。

v1 收尾不设加速比目标。性能策略、收益模型、tile/config 调整、额外融合规则均留给后续性能轮；合法性收口所必需的实际改写仍必须完成。

## 6. V1-C 的有限收尾范围

### 6.1 横向扩展链闭合

这是一轮对现有入口的集成核查，不新增“万能 scheduler”“统一所有 IR”或第二套 verifier。每项核查以当前消费者清单结束；没有发现缺口的模块直接结算。

| 检查边界 | v1 要确认的结果 | 当前已有入口 |
|---|---|---|
| operation 语义到分析 | 同一 operand slot、轴关系、effect 和数值属性不会在 helper/普通操作之间失真 | FragmentOpInterface、StructuredOpInterface、公共 scalar/product/control 分析 |
| 分析到改写 | 能证明的已支持形态有实际转换；不能转换时不先破坏原程序 | ExecutionSchema、SchemaMutation、ValueMaterialization、CPU/DSA 本轮共同机制 |
| 改写到后置检查 | 完整 group 维护自己的 value/access/control/aggregate 关系；失效分析重算 | GPU finishTransformation、CPU PassSupport、DSA group verifier |
| IR 到 provider | provider 读取当前 IR；支持判断与发射同源，数值属性和 ABI 不靠默认值补齐 | 已有 terminal source registry、NativeSource、ProgramContract |
| 参数到执行 | 既有 typed binding、候选约束与 runtime 使用一致；资源估计和原生观察分开 | Resources、ConfigurationAssessment、NativeObservation |
| 开发者入口 | 公共接口与私有实现边界清楚，贡献者知道该改哪个 family/层 | 现有 CONTRIBUTING 与 include/lib 职责目录 |

此次审计已确认 GPU owner/CSE、workspace ABI、候选约束、CPU pipeline/implementation 注册、公共 invocation/产物解析等主体存在且职责合理。集成检查不能成为把这些模块再重构一遍的理由。

目录原则：共享头文件位于 `include/Intent/...`；只供一个实现组使用的私有头可以与 `lib/...` 实现相邻。按稳定职责组织子目录，避免新增平铺名单；不按文件行数机械拆分，不强迫 CPU/GPU 共用物理拓扑。

### 6.2 结算已有正确性欠账

[mojo-x86.csv](../experiments/cpu/results/mojo-x86.csv):40 仍记录 `fp8_gemm` 的 `numerical_failed`。这是已有未结事项，本次没有重新运行，不能把旧失败直接认定为当前 HEAD 的同一 bug，也不能在 v1 完成声明中忽略。

收尾动作：使用该原入口、原输入与既定容差重新定位外部存储、source 运算、accumulator、结果转换和 reference 合同。若是编译实现错误，修复对应合法性/数值实现；若存在参考合同差异，以实际证据说明。禁止改作者算法、放宽容差或把该条移出支持范围来取得“通过”。这属于正确性收尾，不是性能调优。

同表的 `run_only` 条目按原原因保留，例如缺少同合同 reference。它们不自动升级为数值通过，也不要求为 roadmap 新建参考算法或测试矩阵。当前仅 source generation 的能力同样不能改写成 native 运行通过。

### 6.3 对齐现有产品入口

已发现具体说明漂移：[README.md](../README.md):83 仍称 Torch adapter 不支持 InOut；当前 [artifact API](../python/intent/runtime/artifact.py):164–176 和 [register_operator](../python/intent/runtime/torch.py):27–93 已支持受限的 mutable 调用，并明确 mutable alias 与 autograd 限制。

本轮先记录，实施 V1-C 时只同步这类已证差异到现有 README/用户指南/MCP 材料，保持与已实现 API 一致，不再另造教程体系或重复公共 binder。MCP 手册继续只提供通用语法与语义，不加入完整算法、题解或调优建议。

[分发工作流](../.github/workflows/distribution.yml):29–100 和 [环境说明](../environment/README.md):114–122 已提供 Ubuntu 22.04/x86-64、Python 3.10、LLVM/MLIR 20 的产物构建入口。本次未查询远端工作流执行成绩，不把“工作流存在”等同于某次发布成功；也不重新立项建设打包系统。

公开产品收尾仍需：维护者选定项目许可证和实际分发方式；现有支持范围与可取得产物一致；安装后的原公开调用能完成。许可证决定影响公开开源发布，不阻断 A/B 的基础设施实现。本路线图不代选许可证、不发布、不增加版本号、CHANGELOG 或迁移文档。

## 7. v1 冻结后的优化扩展方式

后续性能工程应沿下表的现有边界进行。v1 收尾负责让这些边界可用；后续性能轮不再建设它们。

| 优化需要 | 已有事实来源 | 性能轮的正常落点 |
|---|---|---|
| 改 blocking、ownership、遍历 | 当前轴/坐标、control、typed physical parameters 和目标能力 | family pass 内形成新的当前 IR |
| 减少中间物化与重复读取 | def-use、坐标对应、alias/effects/lifetime、重放合法性 | fusion/rematerialization pass，调用共同改写机制 |
| 优化 reduce/scan/helper 执行 | 原 structured schema、identity、combine、顺序和数值许可 | collective pass；provider 继续使用自己的 native primitive |
| 改供数与局部复用 | 当前存储、访问窗口、执行 scope 和实现要求 | CPU/DSA 对应 pass，不修改公共调用 ABI |
| 改候选质量 | 已声明参数域、合法性约束与当前资源事实 | pass 的策略与既有候选数据，不新增 runtime 特例 |
| 解释性能结果 | 当前 IR/source、实际选中配置、native observation、原 benchmark | 使用已有诊断和实验入口，不再建设一套测量框架 |

每项优化必须保持明确的合法性前提，产生真实 IR 改写，并能够服务满足同一条件的不同程序。通用 pass 不等于所有程序都执行同一种策略；CPU 与 GPU 也不必使用同一份物理变换。

Triton 的 [Combine.cpp](../../ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp):135–192、249–282 在已有语义内把 multiply/reduce、dot/add 改成更合适的当前 IR；TileLang 的 [CUDA pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py):100–142 在明确阶段形成 pipeline、storage 和 tile lowering。它们证明 compiler 可以主动改变物理结构；Intent 应在稳定的自身层次内沉淀这些优化，不退化为算子名模板，也不复制下层 lane layout、机器 allocator 或 ISA。

## 8. v1 的完成判据

以下结果同时成立，才结束本路线图；不以一个 commit、一个新接口或少量运行代替整组完成。

- [ ] S1–S5 的全部列明消费者已迁移；同责旧实现已删除，合理的目标差异保留。
- [ ] 对现有支持范围，IR 表达、资格分析、实际改写与 verifier 的责任闭合；不存在依赖固定前端拼写才能成立的已知同类缺口。
- [ ] 正常 construction 与独立 IR 入口使用同一约束；serializer 不负责猜补缺失执行事实。
- [ ] 横向集成检查已结算到现有入口，后续普通优化可以使用既有接口完成，不需要改 construction、公共 ABI、provider/runtime 或产物合同。
- [ ] 原生产程序的相关编译、native 运行与既定容差检查完成；FP8 历史失败得到明确结论，run_only/source-only 状态保持真实。
- [ ] 已发现的公开说明漂移同步，贡献者能够从现有目录和指南找到正确扩展点。
- [ ] 对外开源交付前，许可证、支持组合与实际分发渠道明确；已有安装路线和原公开调用可用。

未来新增语义、执行事实、硬件或 ABI 变化单独立项。这不把当前已知结构问题留给性能轮，也不要求为未知未来优化提前造字段、pass 或框架。

## 9. 实施与验证纪律

1. 在正式规格和当前代码上实施；结构问题按本表收口，新增事实必须有真实消费者。报告不是修改语言合同的授权。
2. 只使用已有 compiler/build、公开示例与 production registry；不创建测试目录、pytest、fixture 或临时数值/边界脚本，不扩大矩阵。
3. 对真正受影响的原程序执行必要验证，保持作者算法、输入规模、外部 dtype 和原容差；不要求每个内部提交全量重测。
4. 生成、native 编译、运行、数值和性能分开报告。架构轮记录实际耗时变化，不同时追加性能修复和调优任务。
5. 结果回写各实验组既有输出；不建立新的平行汇总表。中间 IR/source/cache 留在仓库外。
6. 里程碑可以由多个连贯提交组成；如实报告实现规模和删除路径，不用目录移动、报告长度、测试数量或行数填充代替进展。
7. 提交本轮自己的文件，保留他人修改，不自动 push 或发布。后续纯性能轮严格按 §1.2 执行。

## 10. 本轮调查结论

当前需要完成的是 **五个已定位结构缺口及其集成、正确性、产品收尾**。旧 roadmap 的核心基础建设已大量完成，应正式移出待办；本轮未找到理由重新设计整套 IR、pipeline、ABI 或 runtime。

完成 V1-A、V1-B、V1-C 后，进入独立的性能优化阶段。此前讨论的 CPU masked producer 融合收益、GPU helper 带来的 blocking 机会、分页 Triton 的历史时间差、QR 性能和跨架构候选质量，都放到该阶段；本路线图不把它们混入基础设施完成标准。
