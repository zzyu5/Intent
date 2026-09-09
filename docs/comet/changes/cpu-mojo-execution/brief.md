# 目标

建立第一条真实的 CPU execution family：同一 Intent DSL 与 canonical KIR，经 CPU physical program 和语义保持的优化 passes，lowering 到 Mojo SIMD，再由 Mojo compiler 生成可在 Linux x86-64 上调用的 native artifact。交付必须包含正式编译与 runtime 路径，以及逐点融合、行归约、dense GEMM 三类算子的 generated/source 性能结果；仅安装 SDK、打印 Mojo 模板或跑通单个算子不算完成。

# 范围

- 共享 frontend、canonical KIR 及可复用的 shape/index/effect/dependence 分析；在 KIR→GPU 之前选择 CPU family，不让 CPU 经过 GPU topology。
- 建立可独立解释的 CPU physical program，显式表达任务分区、循环、SIMD、访问与尾部、归约累加器、cache blocking、必要 packing 和中间值生命周期。优先复用合适的 MLIR 标准 operations，仅为缺失的 CPU 执行事实补最小表达。
- 增加 Mojo provider、x86 hardware capabilities、native shared-library 编译、C ABI 和 CPU tensor runtime。保留 specialization、artifact 复用及有限参数 tuning。
- 首批正式 benchmark 为 f32 batched row affine、weighted RMSNorm、ordinary GEMM。前两项复用现有 Intent 作者 kernel；GEMM 在现有 contraction 模块内补硬件无关的 f32 dtype 变体，不改原 GPU variants。
- 小型手写 Mojo SIMD baseline 与相邻 runtime 按 `source/` 的语言、真实来源和算子职责组织；接入现有 `examples/repro/v2/` 生产入口和 CSV，不另建测试体系。
- 在 Build 中补齐 `doc/compiler/` 的稳定 CPU 执行模型及必要入口说明；本 change 的完整目标规格见 `specs/cpu-mojo-execution/spec.md`。SDK、插件、缓存与机器专用设置留在项目之外。

# 非目标

- 本轮不移植全部 GPU registry，不实现 AMX、RVV/ARM、跨 NUMA 调度、GPU Mojo、分布式或完整 dtype/atomic/sparse 支持。
- 不替换作者算法，不从 baseline 选择 whole-operator 模板，不把已有 GPU IR 改名作为 CPU IR，不在 serializer 中补优化结构。
- 不直接依赖 Mojo 私有 MLIR 作为稳定输入 ABI，不复制 Mojo/LLVM 的寄存器分配、指令选择或机器调度。
- 不承诺未测得的 CPU 性能，也不沿用归档 cuTile change 的 1.05 门槛。第一轮要求真实 SIMD/分块/并行编译能力和可比较的性能结果，不以标量串行占位路径冒充完成。
- 不做 benchmark 之外的运行测试，不新建临时或永久测试脚手架；不改无关 GPU 实现、模型配置或用户文件。

# 验收示例

- A1：三类作者程序从 canonical KIR 构造完整 CPU physical program；源码及同次 benchmark 的编译产物显示任务分区、向量访问、归约累加器、GEMM 分块/复用由真实 passes 产生，且 Mojo serializer 不读取 KIR 或按 kernel 名称补这些结构。
- A2：正式 Python 编译与 artifact 调用入口接受 CPU tensors，产出并复用 Mojo native shared library；shape/stride/offset、读写方向、alias、输出与 scratch 生命周期明确，单次调用同步完成全部 CPU 工作，不进入 CUDA allocation/launch 路径或隐式调用另一算法。
- A3：规格中的三条 CPU registry benchmark 均得到 generated/source 算子执行时间、G/S ratio 及同一次运行的容差内数值结果。双方使用同算法、f32、同 shape 和线程预算；计时包含算子必需的 packing、任务调度及同步，不包含编译、调优、输出分配或 Python 循环派发开销。
- A4：CPU 有限候选绑定真实 physical parameters；至少一条上述 benchmark 在同一程序结构内实际编译、计时并选择两个或更多合法候选的 winner，后续调用复用产物和 winner，不恒取默认表首行、不搜索算法、不另跑调优验证矩阵。
- A5：稳定 CPU 规格与实现边界一致；最终只读审查给出 Intent 与 `ref/triton` 或 `ref/tilelang` 的对应 file:line、具体差异及后果，并结合 `ref/modular` 的 SIMD、并行和 GEMM 实现确认优化归属。未支持能力明确诊断，不能静默变算法或伪装支持。

# 约束与不变量

- `AGENTS.md` 和 `doc/` 为权威；本 change 补 CPU 设计，不把当前 GPU-only 行为写成跨 family 约束。
- target 在作者 source/KIR 之外选择。CPU physical program 是该 family 唯一 executable authority；analysis cache、origin、候选记录不代替 executable SSA、regions、types 和 effects。
- 所有结构变化有明确 legality、实际 IR rewrite 和保持条件；普通运算保持数值语义。Mojo 默认跨语句 FMA contraction 不直接继承；仅在 Intent 语义允许的位置显式实现融合或重结合。
- 一个 kernel specialization 对应一个 host-visible artifact entry。线程任务和 kernel-local packing 属于同次调用，不能变成隐藏的跨 invocation workspace。
- 初期性能预算固定在同一 NUMA 节点的少量物理核，generated/source 使用相同预算；编译可有限并发，计时互不干扰。不占用 H100，也不全程串行等待每个候选准备。
- 数值检查仅在正式 benchmark 中执行一次；沿用既有 case 容差，新增 f32 GEMM 的容差在规格中预先固定，不因失败放宽。编译器自身的 legality/verifier 是生产编译步骤，不另建运行测试。

# 决策

- 用户已授权安装相关工具并创建 CPU/Mojo change 与真实分支。本 change 绑定 `comet/cpu-mojo-execution`，目标分支为 `main`，使用独立 worktree 隔离主目录中的用户内容。
- 采用单个普通 Native change：CPU modeling、passes、provider 和 runtime 共同决定同一执行链，三类 benchmark 横向约束它；拆为可独立宣称完成的 emitter/单算子子任务会破坏该边界。
- Mojo 是首个 CPU source provider，x86-64 是首个 hardware target，两者不混为作者语义。优先采用 native shared library + C ABI，不以 Python importer 的自动编译代替正式 artifact/runtime。
- 工具使用用户级隔离环境、官方编辑器扩展和需要的官方 skills/docs MCP，不修改系统 Python 或原 GPU 环境。安装结果不等于 CPU backend 已完成。
- 当前 `tools/intent-compile/intent-compile.cpp:132–183` 无条件先进入 GPU construction，最后才分 provider；`ref/triton/python/triton/compiler/compiler.py:289–327` 由所选 backend 提供实际 stages。具体后果是 Intent 的 CPU 分叉必须前移，不能只给末端 switch 增加 Mojo serializer。
- `ref/modular/Mojo/stdlib/std/algorithm/backend/vectorize.mojo:115–155` 显式处理 SIMD 遍历与尾部；`ref/modular/max/mojo/max/algorithm/backend/cpu/elementwise.mojo:77–99` 组合任务分区与向量化；`ref/modular/max/kernels/src/linalg/matmul/cpu/impl.mojo:328–495` 显式形成 packing、分块与 microkernel 调用。这些是实现参考，不是让 emitter 自动补全结构的授权，也不是需求来源覆盖清单。
- 用户已确认完整 Shape 并授权进入 Build：首批 f32 三类程序、独立 CPU physical program、native C ABI、有限 tuning、三条正式 benchmark 及本轮不设历史性能 ratio 门槛全部保持本规格范围。

# 待解决问题

无。完整 Shape 已获用户确认。

# 验证预期

只复用现有生产 benchmark 的编译、运行、必要预热、调优与计时，在同一次 benchmark 中检查约定容差。三条 case 和计时口径以完整规格为准，实际结果保存到项目正式性能 CSV，失败如实保留并修复受影响路径。

架构自查消费生产编译产物及源代码，不为 IR、ABI、数值边界或兼容性另外写运行测试。对照先检查 CPU current program 的 partition、loop、SIMD、reuse 和 materialization，再检查 Mojo legalization/serialization 与外部 compiler；未测性能不得推断已达标。
