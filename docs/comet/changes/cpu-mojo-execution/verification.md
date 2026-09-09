---
generated_from_state_version: 7
---

# 验证

## 当前结果

- 结果: **验收通过，需要你确认**
- 验证情况: **已完成检查，但需要你确认验证结果**
- 目标周期: 2
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-09T17:43:07.810Z
- 摘要: 独立只读核对当前三类 f32 CPU 程序、Mojo native 生产证据、Weft 生成产物、有限候选与缓存及 Triton/TileLang/Modular/Weft 对照；A1-A5 全部成立。范围明确限于本 change：Weft 仅 generation，非目标 dtype、AMX/IME 与设备 runtime 未作为通过依据。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：同一份作者算法形成独立、typed、可验证的 CPU 程序；三类计算的轴、任务、访问、累加器、blocking/reuse 与生命周期由当前 IR 和真实 passes 承载。CPU 公共层不硬编码 AVX2/AVX512 或回读 KIR 补结构，provider serializer 只消费已形成的程序。 | 直接证据：lib/Conversion/KIRToCPU/KIRToCPU.cpp:423-438 在 immutable KIR 分析后创建独立 CPU module；CPUAttrs.td:13-44 与 CPUOps.td:11-31 提供 typed ABI、view/scalar、Tasks、Reduce、capability/configuration；lib/Dialect/CPU/Transforms/Passes.cpp:78-150 形成 fusion、blocking、task partition/isolation，并由 PhysicalProgram.cpp:135-209 校验 ABI、访问、生命周期和 materialization。BlockContractions.cpp:44-149 实际形成 tile/K-block/panel/microtile/尾部。生成的 affine.cpu.mlir:3-36 与 gemm.cpu.mlir:5-100 可见任务、访问、尾部、packing 和结构化 contraction。公共层不做 AVX 选择；Mojo 的 AVX 限制位于 lib/Target/Mojo/Transforms/Legalize.cpp:67-74，serializer 只消费已形成函数。 |
| A2 | passed | brief.md | A2：三条既定 f32 CPU benchmark 经正式 Mojo native artifact 执行，分别对照成熟 Modular/MAX 基线得到算子时间、G/S 与同次容差内数值结果；每项 generated/MAX 执行时间比均不超过 1.05。C ABI、CPU views、输出/scratch 生命周期和同步调用明确，双方线程预算与计时范围一致。 | 直接证据：report/baselinev2/mojo-x86.csv:2-4 的 generated/MAX 分别为 Affine 2.186536/2.766295 ms ratio 0.790421、RMS 7.853616/8.864000 ratio 0.886013、GEMM 2.248740/2.151451 ratio 1.045221，均 pass 且不超过 1.05。examples/repro/v2/providers/mojo/common.py:14-52 固定单 NUMA 8 个物理核并使用 native benchmark；measurement.py:287-318 先做同次容差比较再取 generated/source p50。Mojo C ABI 位于 lib/Target/Mojo/Serialization/Serializer.cpp:90-93，view/stride/output/alias 与 winner 输入校验位于 python/intent/runtime/mojo/program.py:81-150。该性能结论复用既有正式 runner/CSV，未由本 verifier 重跑。 |
| A3 | passed | brief.md | A3：同一 CPU 程序的三类任务内计算可生成带合法 typed axes、数值 operations、访问和生命周期的 Canonical Weft IR；生成接口保留所需任务边界，不重读 KIR、按 kernel 名选 std 模板或绕过 Weft 直发 RVV/IME。生成产物明确不等于 native 调用、RISC-V 运行或性能通过。 | 直接证据：lib/Target/Weft/Transforms/Legalize.cpp:51-109 将 CPU Tasks 的 coordinate/captures/访问和 AxisRelations 映射到 typed Weft Kernel/View；:352-410 映射 pointwise、contract、reduce 数值操作；:420-535 仅消费当前 CPU IR；:550-572 明确要求 Tasks、运行 Weft verifier 并返回 native=false metadata。生成物 /mnt/hdd/tmp/intentdsl-weft-generation.Q8wbG6/affine.weft.mlir:2-40、rms.weft.mlir:2-34、gemm.weft.mlir:2-118 保留 typed axes、views、commit、reduce/outer_contract 和任务边界；gemm.weft.mlir:2、438、874、1310、1745 还显示五个独立候选任务。Weft serializer 仅在 lib/Target/Weft/Serialization/Serializer.cpp:5-11 验证后打印 IR，没有 KIR/template/RVV/IME 旁路。 |
| A4 | passed | brief.md | A4：有限候选绑定真实 CPU/provider 参数，至少一条正式 benchmark 实际编译、计时并选择两个以上合法候选；除向量宽度外，本轮已有 task/cache/register 分块参数也有真实 IR consumer。产物与 winner 分别复用，候选先满足数值、访问和资源合法性，不恒取首行或搜索算法。 | 直接证据：lib/Dialect/CPU/Transforms/Passes.cpp:16-66 校验有限候选 schema，:78-150 建立每个 Configuration 并分别 clone、blocking、partition、isolate；ConfigurationAttr verifier 位于 lib/Dialect/CPU/IR/CPUDialect.cpp:71-78。taskGrain 由 PartitionTasks.cpp:83-120 消费，tile/cache/register 参数由 BlockContractions.cpp:44-149 消费，vectorWidth 由 VectorizeLoops.cpp:185-189 消费。gemm.cpu.mlir:6、667、1328、1989、2648 展示五个真实 bindings，最后一项为 <8,1,128,128,128,4,2>；对应 gemm.weft.mlir:2、438、874、1310、1745 均生成。Mojo runtime 的 NativeCall.choose 在 python/intent/runtime/mojo/program.py:33-49 对全部候选实测后选择最小值，产物缓存与 winner/timing 缓存分列于 :54-79，编译身份在 compilation.py:57-83。既有正式 GEMM benchmark 已实际编译、计时并完成候选选择；未把候选恒取首行。 |
| A5 | passed | brief.md | A5：CPU dialect、analysis、passes、provider 与外部 compiler 的职责可由源码和生成程序核对；scalar/vector/matrix 表示边界、转换、资源与不支持能力明确，未复制外部机器 compiler 或引入 emitter 优化路径。对照当前 Intent 与 Triton/TileLang、Modular/Weft 的具体实现说明差异及后果，稳定规格与该边界一致。 | 直接证据：CPU/Mojo/Weft职责边界由 lib/Target/Mojo/Transforms/Legalize.cpp:67-111、Mojo serializer:364-399、Weft legalizer:352-365/550-572 分开承载；当前 scalar/vector/matrix 边界由 MaterializeRegisterContractions.cpp:51-146 与 VectorizeLoops.cpp:114-181 形成，Weft 只发 canonical OuterContract。对照差异：Triton 的真实转换在 ref/triton/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp:684-719，依赖 target、type converter、legality 和 GPU rewrite patterns，后果是其 warp/GPU conversion 不能替代 CPU program authority；TileLang 的真实同类实现 ref/tilelang/tilelang/tools/pass_visualizer/examples/gemm_relu.py:29-46 直接建立 GPU grid、shared buffers、fragment 和 T.gemm，proxy.py:58-72/95-147 处理其 buffer shape/strides，后果是这些 GPU roots 不进入 Intent CPU family；Modular CPU 在 ref/modular/max/kernels/src/linalg/utils.mojo:493-544、581-595 以 cache/ISA heuristic 选择 tile/microkernel，而 Intent 用经 verifier 约束的有限 bindings；Weft 外部机器 lowering 在 TianchenRV/lib/Target/IME/FragmentMaterialization.cpp:97-140 与 lib/Target/RISCVCompiler.cpp:45-66 才形成 fragment/layout/resource/pass pipeline，后果是本 change 的生成端不冒称 RVV/IME 执行。源码、生成 IR 与稳定规格边界一致。 |

## 检查

_没有记录 Runtime 检查。_

## 阻塞项

- **user**: The generic Skill bridge cannot prove an independent Verifier execution; user confirmation is required before Archive. — next: `await-user`

## 风险与跳过的工作

_未报告风险。_

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 0 | recovery | — | Native confirmed acceptance criteria changed | 2026-09-09T11:17:22.937Z |
| 2 | 1 | 1 | pass | — | 独立只读核对当前三类 f32 CPU 程序、Mojo native 生产证据、Weft 生成产物、有限候选与缓存及 Triton/TileLang/Modular/Weft 对照；A1-A5 全部成立。范围明确限于本 change：Weft 仅 generation，非目标 dtype、AMX/IME 与设备 runtime 未作为通过依据。 | 2026-09-09T17:43:07.810Z |



## 结论

独立只读核对当前三类 f32 CPU 程序、Mojo native 生产证据、Weft 生成产物、有限候选与缓存及 Triton/TileLang/Modular/Weft 对照；A1-A5 全部成立。范围明确限于本 change：Weft 仅 generation，非目标 dtype、AMX/IME 与设备 runtime 未作为通过依据。
