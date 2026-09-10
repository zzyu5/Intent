---
generated_from_state_version: 7
---

# 验证

## 当前结果

- 结果: **已归档**
- 验证情况: **已完成检查，验证结果已确认**
- 目标周期: 1
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-09T21:32:07.785Z
- 摘要: A1–A4在本change固定的f32 Softmax/weighted LayerNorm范围内均有当前代码、生产生成物和既有benchmark证据支持。未覆盖Weft设备执行、f16/bf16、GPU及其它CPU算子；既有RMSNorm的G/S=1.112575不属于本轮A3范围。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：现有 f32 stable_softmax 与 weighted_layer_norm 不改作者算法即可通过共同 CPU construction/transformations；前者的 stride/noalias 约束被保留并兑现，exp、maximum_num、additive/maximumNumber reduce 及其 producer/consumer 形成可验证的当前程序，而不是 emitter 中新增整算子逻辑。 | 直接证据：examples/kernels/normalization/softmax.py:6-9保留strides=(None,1)、noalias，:35-36显式倒数乘法；KIRToCPU.cpp:39-56校验并写入ABI。生成的softmax.cpu.mlir:4、:26-47含noalias接口、maximum_num、exp、sum、倒数与乘法；layer_norm.cpu.mlir:22-49含两条矩统计add reduction及rsqrt/逐点消费。运行时program.py:81-100、:133-140兑现连续布局和noalias条件。 |
| A2 | passed | brief.md | A2：两条程序均得到正式 Mojo native callable artifact，以及从同一结构化 CPU 程序生成的合法 Canonical Weft IR/任务接口；后者明确不可调用、不声称设备运行，未支持的语义不静默近似或 fallback。 | 直接证据：/mnt/hdd/tmp/intentdsl-pointwise-generation.LQwCkh下两项均有.mojo、.mojo.mlir、.mojo.json；softmax.json和layer_norm.json均声明native:false并列出6个Weft task。pipeline.py:27-30与mojo/compilation.py:57-82形成shared-lib/ctypes native artifact；Weft Legalize.cpp:566-588生成Kernel IR及native:false元数据，:295-330对未支持表达式失败闭合，无静默fallback或设备执行声明。 |
| A3 | passed | brief.md | A3：两条固定 CPU benchmark 都取得 native 算子执行时间、真实同算法 MAX 库级 source 时间与 G/S，并在同次运行通过预定容差；两项分别 G/S≤1.05，结果进入既有 CSV。LayerNorm 基线明确是同矩统计的 MAX rowwise 路径，不用 Welford 数字冒充。 | 直接证据：report/baselinev2/mojo-x86.csv:5-6记录Softmax 7.967712/7.923892ms、G/S=1.005530，LayerNorm 3.893471/3.728957ms、G/S=1.044118，均pass且≤1.05。measurement.py:263-327在同一benchmark流程执行生成/源码、反向计时并于:321-323复核输出；normalization.py:17-30给出既定容差。MAX LayerNorm source:24-73使用rowwise ReduceSum及sum(x)、sum(x²)的E[x²]-E[x]²路径，明确不是Welford。 |
| A4 | passed | brief.md | A4：两条程序实际消费共同 CPU 的任务/向量、归约实现、融合和中间值生命周期机制；有限 tuning 只实例化合法物理参数并实测选优，产物/winner 继续复用。完整 lowering 不依赖 kernel 名、固定 case shape 或未实现的参数，相关新旧 executable path 不并存。 | 直接证据：生成的softmax.mojo.mlir:16-140和layer_norm.mojo.mlir:14-139含任务并行、显式归约树、vector load/shuffle、尾部循环及生命周期；Passes.cpp:135-157实例化候选并删除原函数，:160-178完成materialize、fusion、reduction traversal和vectorize；TuningProfiles.json:3及Passes.cpp:99-133仅接受有限合法正整数候选，program.py:33-49实测并缓存winner。Serializer.cpp:247-359按当前operation类型拼写，无kernel-name分支；生成IR使用动态memref.dim和tail。对照ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:228-316、:319-351，Triton按register layout/warp做归约并更新layout，而CPU VectorizeLoops.cpp:123-207沿连续logical traversal形成vector树，后果是未引入GPU warp/layout ownership。 |

## 检查

_没有记录 Runtime 检查。_

## 阻塞项

_无。_

## 风险与跳过的工作

_未报告风险。_

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | pass | — | A1–A4在本change固定的f32 Softmax/weighted LayerNorm范围内均有当前代码、生产生成物和既有benchmark证据支持。未覆盖Weft设备执行、f16/bf16、GPU及其它CPU算子；既有RMSNorm的G/S=1.112575不属于本轮A3范围。 | 2026-09-09T21:32:07.785Z |



## 结论

A1–A4在本change固定的f32 Softmax/weighted LayerNorm范围内均有当前代码、生产生成物和既有benchmark证据支持。未覆盖Weft设备执行、f16/bf16、GPU及其它CPU算子；既有RMSNorm的G/S=1.112575不属于本轮A3范围。
