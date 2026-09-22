# TritonBench：当前问题、JIT 缺口与下一步

> 本文保留写作时的调查与测量快照；当前运行入口和结果见本实验组的 [README](../README.md)。

2026-09-16。依据当前 checkout、原始候选及既有测量结果。编译器代码基线为 `8706c2c5`，另有尚未提交的 gather workspace 与参数处理修改；当前不是已冻结的实验版本。本次只做调查和报告，不修改编译器、不启动新一轮生成。

**核心判断：现在同时存在编译器实现缺陷、运行准备不完整、作者程序错误，以及尚未归因的数值差异。不能把它们统称为“JIT 没写好”。统一的 Intent JIT 用户入口确实还没有完成，但显式编译入口和若干后端自身的 JIT 已经存在。**

**我们要完成的结果仍然是第一阶段：新生成的 Intent 程序在多轮中达到约 95% 正确率，再使完整算子性能达到或超过 agent Triton 和原 ref。** 目前最近三轮首次成绩为 89/88/89；最新一批旧程序开发复测为 93/100，不能算成新生成已经稳定达到 95%。93 个正确程序中 57 个快于 ref，但 Intent/ref 耗时比的几何平均仍为 1.43，整体性能尚未达标。[最新完整复测](/home/kingdom/.local/share/intentdsl/agent-evaluation/first-stage-high-100-intent-stability-r26-recheck/workspace-and-public-math-closure.csv)

**先区分四层职责，避免把修复放错位置。** 下图以 Triton 路径为例：

```mermaid
flowchart LR
    A[作者的 Intent 程序] --> B[Intent 编译：分块、访问与源码生成]
    B --> C[Triton 编译：线程布局与机器资源分配]
    C --> D[运行准备：资源检查、配置测量与选择]
    D --> E[GPU 执行，与 ref 比较]
```

| 层次 | 应当负责什么 | 当前暴露的问题 |
|---|---|---|
| 作者程序 | 算法、循环、多个 kernel 和 host 编排 | 有的公式错误，有的 shape/type 表达非法 |
| Intent 编译器 | 保持上述语义，形成合法的分块、访问和目标程序 | 坐标/依赖处理曾出错；最近资源策略又生成了非法展开 |
| Triton 等后端 | 实际线程布局、共享内存分配、寄存器分配与机器代码 | 有真实资源结果可用；不应在 Intent 中复制其分配器 |
| Intent 运行准备与统一入口 | 调用编译链、准备可运行配置、缓存并启动 | 预编译未覆盖实际 shared 检查；统一延迟 JIT 和字符串 target 尚未接齐 |

编译器正确性比较的是“生成程序是否保持作者 Intent 语义”；实验正确性比较的是“最终结果是否符合 ref”。作者公式写错时，编译器忠实执行错误公式，实验照样失败。

**shared memory 问题有两个不同原因，只有其中一个属于 JIT/运行准备。**

第一，Intent 的目标实现选择有错误。最新 `matrix_power_eig` 候选本来可以编译运行，但我们新增的策略认为某个配置需要太多共享内存，就改用展开的 FMA 实现，却没有检查展开后的张量是否合法：

- `BK=64` 时，两块 f32 输入的大小估算为 `2 × 256 × 64 × 4 = 131072` 字节。
- 设备每 block 的 opt-in 上限为 `101376` 字节，于是代码选择展开路径。
- 展开产生 `256 × 64 × 256 = 4194304` 个元素，超过 Triton 的 `1048576` 元素上限，编译失败。
- 同一份原程序此前在 `BK=16` 下能编译并运行到数值比较。因此这是我们选择了非法物理配置/实现，不能据此说作者的整个矩阵乘法不受支持。

这里的 `131072` 是输入块大小估算，不是 Triton 最终共享内存分配结果。“原生实现可能放不下”也不证明“展开实现一定合法”。应修目标 pass 的合法性判断，优先保留已知合法配置，不为每个坏配置增加替代算法。[实现选择](/home/kingdom/phdworks/intentdsl/lib/Target/Triton/Transforms/Legalize.cpp:2113)、[展开的 FMA](/home/kingdom/phdworks/intentdsl/python/intent/runtime/triton_math.py:39)

第二，运行准备缺少编译后的资源可运行性检查。当前实验的预编译主要调用 Triton `warmup`，得到编译产物后就返回；实际 shared 检查在 Triton 准备启动 kernel 时才发生。这解释了之前 QR 的“预编译无失败，最终启动却 shared 超限”。

| 我们知道的信息 | 当前情况 |
|---|---|
| 设备硬件容量 | 真正查询 CUDA Driver；当前每 SM 为 102400 字节，每 block opt-in 上限为 101376 字节 |
| Intent 预测的 tile 资源需求 | 根据 shape、dtype 等静态估算，不能冒充后端实际分配 |
| 每个编译产物要求的 shared 字节数 | Triton 提供 `CompiledKernel.metadata.shared`，并在 `_init_handles()` 中检查 |
| Intent 是否主动消费该结果 | 当前没有把它接入统一准备流程，也没有反馈给 Intent 的资源决策 |
| 当前 SM 实时剩余多少 shared | 不按这种方式分配；编译产物声明每 block 的需求，由 GPU 据此调度 |

QR 曾报 `Required: 524288, Hardware limit: 101376`。这里的 524288 来自 Triton 对编译产物的检查，是真实的编译资源需求，不是 Intent 的估算。另需区分：`torch.empty` 创建的 invocation workspace 是全局显存，不是 SM 共享内存。

Autotune 负责比较配置速度，不负责修复程序或重新设计内存分配。当前 Triton 会把资源失败的配置记为无穷大耗时；全都失败时仍可能选出一个，最终启动再报错。另一方面，张量 shape 非法产生的普通 `CompilationError` 不属于这类可跳过的资源失败，会直接中断准备。不能通过捕获所有异常来掩盖 compiler bug。

依据：[设备查询](/home/kingdom/phdworks/intentdsl/python/intent/targets/gpu/device.py:34)、[现有预编译](/home/kingdom/phdworks/intentdsl/experiments/agent_tritonbench/program.py:106)、[Triton 实际 shared 检查](/home/kingdom/.local/lib/python3.10/site-packages/triton/compiler/compiler.py:436)、[QR 资源失败记录](/home/kingdom/.local/share/intentdsl/agent-evaluation/first-stage-high-100-intent-stability-r25-recheck/fused_qr_solve/intent/fixed-collective-warp-distribution/measurement.json:12)。Triton 官方已有 `warmup → _init_handles → metadata.shared/n_regs` 的用法，可直接参考其职责边界。[官方示例](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

**为什么反复生成，还有问题：既有覆盖缺口，也有回归和工作重点扩大。**

新生成的程序会改变循环、切片、归约和中间值组织，触发不同的 pass 组合；这确实暴露了现有实现的不成熟。但每轮失败不是一批全新的独立 bug，长期未解决的作者错误和数值差异也会重复出现，修复之间还会引入回归。

我们没有理由让 reshape 改变元素数，或让普通循环的 scalar carry 自动变成 vector。这样的拒绝应保留。另一方面，合法程序缺少某种目标实现，是编译器侧的能力缺口，需与已有能力被改坏区分，不能把所有能力扩展都当成本轮达标的前置任务。

此前已经知道某份 QR 候选的作者公式有错，却仍花较多时间补其大矩阵存储路径。它可以是合理的编译器能力工作，但不会直接增加该候选的通过数，不应长期占据“先提高正确率”的主线。

最新 93/100 中剩余的 7 题应这样处理：

| 剩余问题 | 现在能下的结论 | 应做的事 |
|---|---|---|
| 3 个 frontend 类型/shape 失败 | 程序本身非法，不应放宽规则迁就 | 保留失败；只修确有缺失的通用语法说明 |
| QR 数值失败 | 作者把反射向量分量当成 R 的对角线回代 | 不改候选，不让 compiler 自动修算法 |
| LDL 数值失败 | ref 使用 compact 因子和 pivot 构造另一个矩阵；首次分歧未定位 | 查因子、pivot、重建矩阵的首次差异，证据不足时保持未归因 |
| 卷积＋BN 数值失败 | 数学公式和坐标基本一致，浮点实现差异尚未唯一归因 | 区分卷积累加与 BN 舍入，不能只说“浮点误差”就结束 |
| 矩阵乘法后端编译失败 | 本次资源策略引入的回归 | 修配置/展开合法性，恢复原程序可运行 |

LDL 的原 ref 把 `ldl_factor` 返回的 pivot 信息当作对角数值使用，不是通常的 LDL 求解；继续遵守用户已确认的“原 ref 实际行为为准”，但不要求 compiler 猜测并替换作者算法。[原 ref](/home/kingdom/phdworks/ref/tritonbench/data/TritonBench_T_v1/solve_symmetric_ldl.py:9)

卷积＋BN 这题使用固定 running mean/variance，不涉及训练时方差计算；TF32 已关闭。当前最大误差约 0.000696，仍超出既定容差。旧 agent Triton 也有类似误差，只能作为调查线索，不能证明我们的编译器无错。[当前测量](/home/kingdom/.local/share/intentdsl/agent-evaluation/first-stage-high-100-intent-stability-r26-recheck/relu_batch_norm_conv2d/intent/workspace-and-public-math-closure/measurement.json:15)

**统一 JIT：用户提出的方向需要做，但不能说现有基础完全没有。**

现在公开可用的组织形式是：

```python
artifact = intent.compile(
    kernel,
    compiler=compiler_path,
    target=intent.TritonTarget(),
)
artifact(...)
```

`intent.compile` 当场完成 Intent lowering 和源码生成；随后各 provider 在自己的阶段编译机器代码。`@intent.kernel` 定义本身不能直接运行，没有首次调用自动完成 Intent 编译的公共 JIT 包装。`target` 当前必须是具有 `.resolve()` 的对象，`target="triton"` 尚不能直接使用。[公共编译入口](/home/kingdom/phdworks/intentdsl/python/intent/compiler/pipeline.py:16)、[kernel 直接调用的拒绝](/home/kingdom/phdworks/intentdsl/python/intent/api/definitions.py:53)

| 用户关心的后端 | 当前接入情况 | 统一 JIT 尚需补齐的部分 |
|---|---|---|
| Triton | [TritonTarget()](/home/kingdom/phdworks/intentdsl/python/intent/targets/triton.py:38)；显式 Intent compile 后，生成程序使用 Triton JIT/autotune | 统一首次调用入口、编译结果缓存和准备期资源检查 |
| cuTile | [CuTileTarget()](/home/kingdom/phdworks/intentdsl/python/intent/targets/cutile.py:38)；已有源码生成和运行 materializer | 接到同一 host JIT 生命周期，验证实际可运行范围 |
| Mojo | [MojoTarget(...)](/home/kingdom/phdworks/intentdsl/python/intent/targets/mojo.py:45)；materialization 时编译 native library | 将现有编译/加载接入统一入口，保持 CPU 配置和 ABI |
| Weft | [WeftTarget(...)](/home/kingdom/phdworks/intentdsl/python/intent/targets/weft.py:25) 当前是 source target；可 `intent.generate`，另有 AOT 导出路径 | 公共 `intent.compile` 尚不能返回 Weft runnable artifact，需接通 native materialization/加载 |

这张表是当前代码接入情况，不是本轮对四后端的运行认证，更不表示这 100 题在四个后端都已通过。Mojo/Weft 的 CPU 参数与工具链要求也不能被字符串名称隐藏。

当前 Mojo native 路径面向 Linux x86-64；Weft 的现有生产路径使用显式 RISC-V Linux profile 和独立 AOT/部署流程。它们需要在相应执行环境中接通，不能默认在当前 GPU 主机上互换运行。[Weft AOT 导出](/home/kingdom/phdworks/intentdsl/python/intent/runtime/weft/compilation.py:70)

已有缓存也是分散的：Triton 等 provider 有自己的 JIT/cache，Mojo 有进程内 native library cache；当前 Intent 公共入口没有统一的首次调用 specialization 与持久 artifact cache。动态 shape 已可作为运行时元数据传给后端，不应为了新增 JIT 一律按完整 shape 重编译；外部 dtype 仍受 kernel 声明约束，不能静默改类型。

**已记录的产品需求：提供 host 侧统一 JIT，支持 `target="triton" / "cutile" / "mojo" / "weft"` 这样的选择。** 下式仅表示期望的使用方式，尚不是现有 API；具体装饰器或工厂函数形式在实现时收敛，不在本报告冒充已定稿规格：

```python
entry = intent.jit(kernel, target="triton")
entry(...)
```

该入口应复用现有 Target 对象、compiler pipeline 和各后端 materializer。首次需要时完成编译和可运行性准备，随后复用结果；specialization/cache 必须区分必要的 constexpr、ABI、provider、硬件和编译选项。补 Intent 层的复用，并继续使用已有 provider 缓存，不另造一套底层缓存。保留高级 target 配置方式。target 仍在 host 选择，不进入 kernel 的算法分支；不会自动换算法或增加隐式 kernel。

这四个名称选择的是 provider，GPU device、CPU 架构与工具链配置仍需单独表达。成熟参考也在 host 侧处理这些区别：Triton 从 active driver/device 建立 target，TileLang 分开 target、target_host 与 execution_backend；不应把这些选择塞进 DSL 运算或另一个 compiler IR。[Triton 入口](/home/kingdom/phdworks/ref/triton/python/triton/runtime/jit.py:686)、[TileLang 入口](/home/kingdom/phdworks/ref/tilelang/tilelang/jit/kernel.py:65)

Weft 接通 runnable artifact 是实际缺口，不能仅增加字符串别名就宣布四后端 JIT 完成。统一入口也不会自行修好错误的分块、QR 公式或数值差异。

**下一步按具体交付推进，不再把所有失败都变成新能力开发。**

| 顺序 | 工作 | 完成条件 |
|---|---|---|
| 1 | 修矩阵乘法非法展开；把真实资源检查接入现有准备流程 | 原程序恢复到可运行；坏配置被准确拒绝；没有可运行配置时明确报错，不吞未知编译错误 |
| 2 | 收束 LDL、卷积＋BN 的归因 | 在原生产 profile 的开发诊断中找到支持归因的证据；只修确定的 compiler 违约，不改候选、ref 或容差 |
| 3 | 实现统一 JIT 的最小调用闭环 | 复用已有 pipeline；先接通 Triton，再逐项接 cuTile/Mojo/Weft，分别验明编译、加载、调用和缓存；不新增实验矩阵 |
| 4 | 冻结 compiler/manual/suite，重新并发生成 Intent100 | 分开保留各轮首次成绩，检验多轮约 95%；没有提交后反馈修复；不新增 Triton 对照组 |
| 5 | 正确率稳定后批量处理性能差距 | 按共同物理结构问题优化，并交付完整算子对 agent Triton/ref 的实测比较；之后才进入第二阶段 |

第 3 项是明确产品待办，不应阻塞第 1、2 项；现有显式 compile 入口可继续承担实验。只有在新的 JIT 入口已经实现并验证后，才将其作为当前能力写入公开手册。准备与编译按资源并发，性能计时串行；不增加独立测试框架、题目或中间管理层。

本报告只记录需求、证据和行动顺序，不修改正式 DSL/IR 规格。已有编译器改动、作者原始程序、原 ref 和历史首次成绩均保留。
