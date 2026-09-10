---
generated_from_state_version: 5
---

# 验证

## 当前结果

- 结果: **验收通过，需要你确认**
- 验证情况: **已完成检查，但需要你确认验证结果**
- 目标周期: 1
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-10T13:53:46.474Z
- 摘要: 独立只读验收覆盖 A1-A5；候选 17d0182→031ab52、授权外部修复 ea4026b21 与正式 Weft/Mojo CSV 证据均满足规格，建议 Runtime 接受并进入 Archive。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：现有 CPU 程序保留完整 task/block、访问、依赖与 structured compute；共同 blocking/参数不再无条件绑定特定 f32 向量微核的内部组织，Mojo/Weft 从同一共同程序边界继续 lowering。 | 直接证据：CPU construction 保留 typed ABI、domain/task 与 structured ops（lib/Conversion/KIRToCPU/KIRToCPU.cpp:26-100、347-449）；共享 pass 在绑定后调用 implementation formTile，而非预置 f32 微核（lib/Dialect/CPU/Transforms/Passes.cpp:75-149、BlockContractions.cpp:20-89）。Mojo/Weft 均从同一 CPU boundary 分流（tools/intent-compile/intent-compile.cpp:154-179）。对照 TileLang lower_tile_op.cc:967-985、src/op/gemm.cc:198-236，职责差异明确且无 emitter 算法旁路。 |
| A2 | passed | brief.md | A2：实际计算块通过明确的实现适用性、需求协调、参数绑定与微程序展开形成可验证 IR；现有 f32 Mojo native 路径在新机制中真实可调用，布局/供应与展开使用同一选择，无新旧旁路并存。 | 直接证据：ImplementationRegistry 提供 applicable/legal/parameters/bind/formTile/expand，并将 binding 写入 current IR（include/Intent/Dialect/CPU/Transforms/Implementation.h:15-35、lib/Dialect/CPU/Transforms/Implementation.cpp:10-61）；Mojo 与 Weft 均注册真实消费者和展开路径（lib/Target/Mojo/Transforms/Implementations.cpp:11-139、lib/Target/Weft/Transforms/Implementations.cpp:50-86）。Mojo legalize 消费绑定后 materialize/vectorize（lib/Target/Mojo/Transforms/Legalize.cpp:69-139），Weft 对专业 operation 执行 expand（lib/Target/Weft/Transforms/Legalize.cpp:623-638）。对照 Triton matmul.py:39-89 与 intra_kernel/example_dsl.py:204-282，候选参数和程序实例化边界均有对应证据。 |
| A3 | passed | brief.md | A3：真实 Intent 作者 kernel 从 f32 activation 产生 Q8_K 中间值并让多个 Q4_K×Q8_K 点积消费；沿唯一 KIR→CPU→Weft 链形成量化/点积专业程序的真实展开。§5 的格式、i32 统计、f32 校正和 reduction 合同完整保留；producer/captures/轴/域、消费者、effects、共享 storage 与 lifetime 显式连接。量化不在每个点积里重复，不残留 emitter/runtime recipe，不调用完整 source kernel。 | 直接证据：作者 kernel 先一次量化再由多个输出消费者调用 quantized_dot（examples/kernels/quantization/quantized_projection.py:5-19）；frontend/canonical verifier 保留 closed formats、shape relation 与 f32/u8 schema（python/intent/frontend/lowering/intrinsics/quantization.py:10-50、lib/Dialect/Intent/IR/IntentOps.cpp:1192-1212）。CPU construction 形成 invocation-local producer storage，AxisRelations/PartitionTasks 保留共享轴、任务依赖和独立准备 task（lib/Conversion/KIRToCPU/KIRToCPU.cpp:426-442、lib/Dialect/CPU/Analysis/AxisRelations.cpp:57-60、lib/Dialect/CPU/Transforms/PartitionTasks.cpp:93-132）。Weft expansion 实际生成 Q8_K records、i32 partial/reduction、f32 correction 与 ordered record loop（lib/Target/Weft/Transforms/Quantization.cpp:146-196、199-289）。外部授权 ea4026b21 的 FieldView/field-store lowering 物化 projected full-byte fields并由 final verifier 拒绝残留 projection（TianchenRV/lib/Target/PlanRISCVFieldStores.cpp:13-103、106-229、TianchenRV/lib/Target/VerifyFinalRISCV.cpp:506-527）。 |
| A4 | passed | brief.md | A4：Weft native artifact 可通过 typed buffer ABI 加载并执行完整单次 CPU invocation，任务、scalar、内部资源和 join 均闭合；SG2044 的固定量化投影 case 获得真实 generated/source ms 和 G/S，包含 activation quantization，并在同次 benchmark 通过 `1e-4 + 2e-3*abs(expected)` 容差，结果进入 `report/baselinev2/weft-rvv.csv`。不是仅生成 C、手动运行 reference 或把 SSH/编译耗时当算子时间。 | 直接证据：Weft 通过 --emit=artifact、ABI 对照、AOT system compile 和 ctypes NativeProgram load/run（python/intent/runtime/weft/compilation.py:30-73、119-135；program.py:27-57、71-99、105-198）；provider benchmark 运行 generated/source 完整 native invocation，计时包含量化与 workspace（examples/repro/v2/providers/weft/contraction.py:59-115；source runtime:17-64）。Runtime 提供的正式 benchmark 已报告 generated=6.074266 ms、source=5.798605 ms、ratio=1.047539、同次原容差通过，CSV 已记录（report/baselinev2/weft-rvv.csv:1-2）。 |
| A5 | passed | brief.md | A5：shared/implementation/provider 参数都有实际消费者，有限合法候选以相应目标上的完整 invocation 时间选优并复用 artifact/winner。既有 Mojo f32 affine、RMSNorm、dense GEMM、Softmax、LayerNorm 保持可达；受影响项使用原生产 benchmark、原算法/source/容差，实际 generated/source 时间和 G/S 更新到既有 CSV。 | 直接证据：Mojo/Weft tuning profiles 分离 shared/local 参数并列出有限合法候选（lib/Target/Mojo/Transforms/TuningProfiles.json:1-19、lib/Target/Weft/Transforms/TuningProfiles.json:1-11）；CPU pass 对无 contraction 的 M/N/K 强制保持中性值1（lib/Dialect/CPU/Transforms/Passes.cpp:95-109），NativeCall.choose 在完整 invocation 上选 winner 并复用缓存（python/intent/runtime/weft/program.py:71-99）。既有 Mojo 五项 registry 能力保持可达（examples/repro/v2/registry.py:155-160），既有 CSV 五项均为 pass（report/baselinev2/mojo-x86.csv:1-6）。 |

## 检查

_没有记录 Runtime 检查。_

## 阻塞项

- **user**: The generic Skill bridge cannot prove an independent Verifier execution; user confirmation is required before Archive. — next: `await-user`

## 风险与跳过的工作

- Weft 本轮仅覆盖标准 RVV/VLEN128 SG2044 AOT/native case；IME、双设备和自动 JIT 属于已确认非目标。
- 外部 field_view 修复仅支持完整自然布局整字节字段；encoded record owner subview offset 明确诊断。
- 未新增统一 G/S 硬门槛，Mojo 未受影响能力复用已完成 production benchmark。

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | pass | — | 独立只读验收覆盖 A1-A5；候选 17d0182→031ab52、授权外部修复 ea4026b21 与正式 Weft/Mojo CSV 证据均满足规格，建议 Runtime 接受并进入 Archive。 | 2026-09-10T13:53:46.474Z |



## 结论

独立只读验收覆盖 A1-A5；候选 17d0182→031ab52、授权外部修复 ea4026b21 与正式 Weft/Mojo CSV 证据均满足规格，建议 Runtime 接受并进入 Archive。
