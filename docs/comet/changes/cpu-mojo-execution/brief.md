# 目标

建立一套可由 Mojo 与 Weft 消费的共同 CPU execution family：同一 Intent 作者算法，经 canonical KIR、正式 CPU dialect/current-program analyses 和语义保持的 passes，形成完整的 CPU 程序。Mojo 首先交付 Linux x86-64 native 执行与成熟库性能对照；Weft 本轮交付生成端，不完整接入 RISC-V runtime。

CPU 不是仅含 AVX SIMD 的专用路径。共同模型须保持 scalar、shaped/vector computation、matrix-eligible contraction、访问、复用与生命周期的关系；具体 ISA、向量宽度、矩阵 fragment 和指令实现由能力与 provider lowering 决定。作者不新增 vector/cube、AMX/IME 或 hart 分工。

# 范围

- 保留同一 frontend、immutable canonical KIR 及可复用分析；在 GPU construction 前选择 CPU family。Mojo 与 Weft 共用 CPU 程序，不各自从 KIR 重建一套执行结构。
- 建立正式 CPU dialect、typed ABI/capability/parameter 表达、current-program analysis、transformation groups 与 verifier。复用合适的标准 MLIR operations；CPU 自有语义不得只靠任意字典、属性存在性或 serializer 私约定维持。
- 共同程序显式保存任务分区、逻辑轴、循环与 carries、访问和尾部、归约/contract 数值 schema、分块、复用及资源生命周期。保留 structured computation 到合适的 lowering 层，不先压成固定 AVX 循环再要求另一 provider 反推。
- 通过公共 passes 推进逐点融合、行归约和 GEMM 的任务粒度、cache/register blocking、数据供应与 materialization；根据真实 consumer/capability 处理 vector 或 matrix 表示，不按算子名称选模板。
- Mojo 保留正式 native shared-library、C ABI、CPU tensor runtime、按需编译、产物复用与有限有效 tuning；本轮实跑范围仍是原有三条 f32 benchmark。
- 三项主要性能基线采用 Modular/MAX 的成熟 CPU 实现：GEMM、weighted RMSNorm 使用对应库实现；affine 使用同算法的库级 elementwise/fused expression。手写 SIMD 版本只作辅助归因，不代替成熟库验收。
- Weft 从共同 CPU 程序生成三类程序所需的 Canonical Weft IR 与明确接口；只生成产物不提供伪造的可调用 artifact。任务内计算与尚未接入的 CPU 调度/runtime 边界必须明确。
- 在现有 `doc/compiler/` 中统一 CPU 设计，正式代码按 execution family、analysis/pass、provider 和 runtime 职责组织；本 change 沿用 `main/current`，不重新创建 worktree。

## 参考范围

`../TianchenRV`、`../ref/modular`、`../ref/triton`、`../ref/tilelang` 及官方 TileLang Ascend 源码是只读的实现/架构参考，不是要求移植全部语言、算子或硬件能力的需求来源。Weft 的既有未提交内容不在本 change 修改范围内。

# 非目标

- 不要求本轮执行 RISC-V、完整接入 Weft JIT/设备 runtime，或实跑 Intel/Apple AMX、IME、ARM、跨 NUMA、异步多引擎与分布式。
- 不把三条 f32 结果当作 AMX/IME 的执行证据；不为使用矩阵扩展隐式改 dtype、精度、算法或调用次数。
- 不修改 Intent 作者编程模型为 Ascend 式 C/V 双执行上下文；不复制 Weft 的 RVV/IME compiler、Mojo/LLVM 的机器 lowering，或引入没有实际 consumer 的占位 extension。
- 不扩展 GPU registry、dtype/shape/线程数矩阵，不改无关 GPU 实现；不保留新旧 CPU executable path、fallback 或兼容开关。
- 不写或运行 benchmark 之外的测试，不建立临时/永久测试脚手架；不更新 SDK、插件或机器配置来代替编译器实现。

# 验收示例

- A1：同一份作者算法形成独立、typed、可验证的 CPU 程序；三类计算的轴、任务、访问、累加器、blocking/reuse 与生命周期由当前 IR 和真实 passes 承载。CPU 公共层不硬编码 AVX2/AVX512 或回读 KIR 补结构，provider serializer 只消费已形成的程序。
- A2：三条既定 f32 CPU benchmark 经正式 Mojo native artifact 执行，分别对照成熟 Modular/MAX 基线得到算子时间、G/S 与同次容差内数值结果；每项满足完整规格第 8 节的用户确认性能门槛。C ABI、CPU views、输出/scratch 生命周期和同步调用明确，双方线程预算与计时范围一致。
- A3：同一 CPU 程序的三类任务内计算可生成带合法 typed axes、数值 operations、访问和生命周期的 Canonical Weft IR；生成接口保留所需任务边界，不重读 KIR、按 kernel 名选 std 模板或绕过 Weft 直发 RVV/IME。生成产物明确不等于 native 调用、RISC-V 运行或性能通过。
- A4：有限候选绑定真实 CPU/provider 参数，至少一条正式 benchmark 实际编译、计时并选择两个以上合法候选；除向量宽度外，本轮已有 task/cache/register 分块参数也有真实 IR consumer。产物与 winner 分别复用，候选先满足数值、访问和资源合法性，不恒取首行或搜索算法。
- A5：CPU dialect、analysis、passes、provider 与外部 compiler 的职责可由源码和生成程序核对；scalar/vector/matrix 表示边界、转换、资源与不支持能力明确，未复制外部机器 compiler 或引入 emitter 优化路径。对照当前 Intent 与 Triton/TileLang、Modular/Weft 的具体实现说明差异及后果，稳定规格与该边界一致。

# 约束与不变量

- `AGENTS.md`、`doc/` 和当前确认需求为权威；旧 CPU 候选及其手写 baseline 数字是实现起点，不是新验收已通过的依据。
- CPU family、provider、hardware target 是不同维度。CPU/RVV 不再作为两套平级的 Intent execution family；实际后端能力不扩张作者语义。
- 每个 transformation 声明输入事实、legality、实际 rewrite、保持语义与失效 analysis；完整 group 后验证同一当前程序。
- 作者 dtype、显式数值属性、允许的重结合、effects、alias、ordered control 与外部 ABI 保持不变。比较允许细微数值差异，不授权 compiler 任意变算法。
- 一个可执行 specialization 对应一个 host-visible entry，内部任务同步完成；workspace 不隐藏跨 invocation 的输入预处理或另一 kernel。
- 生产 benchmark 使用同一 NUMA 的 8 个物理核预算；准备/编译可有限并发，调优和正式计时避免资源争抢。只在同次 benchmark 做一次既定容差检查。
- 缓存、机器路径、工具环境和生成临时产物不进入项目；CSV 只记录真实性能与必要说明，不承担候选耗时或历史审计。

# 决策

- 用户确认将已有 CPU 分支合入 `main` 并清除 worktree，并明确允许仅修正当前 change 的 workspace 绑定为 `main/current`。本次合并不代表验收或归档。
- 用户确认统一 CPU 模型、Mojo/Weft provider 分层、vector 与可选 matrix 的物理表达；不引入 Ascend 式作者引擎分工。原先独立 RVV family、仅 AVX 的公共模型、以手写 SIMD 为主要性能标尺的设定由本 Shape 替代。
- Weft 本轮最多到生成端；不改 Weft 仓库、不接 RISC-V 调用环境、不用 f32 三例声称 AMX/IME 已支持。
- 采用单个普通 Native change：公共 CPU IR、两种 provider 输入边界与 Mojo 性能共同约束同一核心，拆成独立 emitter/算子子 change 没有独立验收价值。
- 当前 `lib/Transforms/CPU/Passes.cpp:78` 的 verifier 和 `:113` 的 AVX 限制，缺少 GPU `include/Intent/Dialect/GPU/IR/GPUAttrs.td:119`、`include/Intent/Dialect/GPU/Analysis/PhysicalProgram.h:302` 对应的 typed/current-program 边界；本 change 修的是该职责缺口，不以 pass 名称或文件数量判断成熟度。
- `../ref/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:684` 使用 type conversion、legality 和真实 patterns；`../TianchenRV/lib/Target/RISCVCompiler.cpp:45` 以实际 passes 形成 layout、memory、schedule、leaf 和资源。这些机制用于约束本轮 CPU 分层，不复制其 ISA 实现。
- `../ref/modular/max/kernels/src/linalg/utils.mojo:493` 与 `:583` 展示 cache 与多列 microkernel 结构；`../TianchenRV/lib/Target/IME/FragmentMaterialization.cpp:97` 展示 typed pack/MMA/unpack。优化和表示转换在 IR 中形成，serializer 不首次创造它们。
- 用户本轮授权修订 Shape，不授权直接继续旧 Build；性能门槛和最终完整 Shape 确认完成前，不进入实现。

# 待解决问题

- [blocking] Q1：三条 f32 benchmark 相对成熟 Modular/MAX 基线的逐项 G/S 上限定为多少？建议以 1.05 为最终目标；1.10 可作为较宽的首轮门槛。该数值尚未获得用户选择，不沿用旧 cuTile 门槛，也不把“仅报出数字”视作性能完成。

# 验证预期

Mojo 只复用现有 `examples/repro/v2/` 的三条性能 benchmark，在同次运行检查原容差；实际结果进入 `report/baselinev2/mojo-x86.csv`。旧手写基线数字不改名冒充 MAX 结果，受影响条目有新测量才更新。

Weft 只执行生成所需的生产编译与编译器自身 legality/verifier，不建立独立测试或运行矩阵；生成文件用于核对真实 lowering，不计作设备执行通过。架构自查为只读代码/产物审查，并给出当前/ref 对应位置、差异与后果。
