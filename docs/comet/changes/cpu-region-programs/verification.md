---
generated_from_state_version: 6
---

# 验证

## 当前结果

- 结果: **验收通过，可归档**
- 验证情况: **已完成检查，验证结果已确认**
- 目标周期: 1
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-11T00:57:28.759Z
- 摘要: 独立只读核对通过 A1-A4；本轮未重复编译、benchmark 或测试，采用已存在的生产构建、native 性能与容差证据。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：真实作者区域程序沿 canonical KIR→CPU→Mojo/Weft 形成可执行程序，支持 region fold 的 typed summary/identity/combine 以及 region scan 的 transition、initial state、apply/emit、final state；区域坐标、captures、相连 MatMul/统计、访问、尾部和任务完成语义不丢失，不由 serializer/runtime 补算法。 | 直接证据：lib/Conversion/KIRToCPU/KIRToCPU.cpp:279-367 将 canonical region fold/scan 转为带 typed helper schema 的 CPU destination-passing IR；lib/Dialect/CPU/Transforms/RealizeRegions.cpp:243-324 实际生成 segment loops、source slices、summary/combine/apply/emit、state carry 与尾部处理；lib/Dialect/CPU/IR/RegionProgram.cpp:11-109 验证 partition、类型、输出关系和 helper effects。Mojo/Weft 现有构建及 native benchmark 均已通过。与 ref/triton/python/tutorials/06-fused-attention.py:54-110 的显式 provider stage loop 不同，本实现把算法保留在当前 CPU structured IR 中，serializer 不重建算法。 |
| A2 | passed | brief.md | A2：GPU 与 CPU 在各自当前执行范围上实际调用同一组区域/identity/状态分析规则，证明成立时产生有效遍历、谓词或 carry 的实际简化；不成立时保留原语义。浮点零值传播不会在缺乏证明时抹去 NaN/Inf 或 signed-zero 行为，不能以复制 GPU pass 或关闭整个优化冒充复用。 | 直接证据：lib/Analysis/RegionSemantics.cpp:7-44 提供共享区间/布尔状态规则，CPU lib/Dialect/CPU/Analysis/RegionPredicates.cpp:72-104 调用共享规则，GPU lib/Dialect/GPU/Analysis/UniformValues.cpp:12-81 适配同一 UniformValueAnalysis；CPU lib/Dialect/CPU/Transforms/RealizeRegions.cpp:205-322 和 GPU lib/Dialect/GPU/Transforms/RealizeRegionFold.cpp:1691-1832 将证明结果写回当前 loops/predicates/carries。lib/Analysis/UniformValues.cpp:55-163,265-280 仅对整数零或逐值 FMA 结果成立时传播，不以浮点零无条件消除 NaN/Inf/signed-zero。 |
| A3 | passed | brief.md | A3：相连计算的分块、访问、状态与所选 implementation 需求使用同一候选 binding；外围供应、共享准备、资源 owner 与 lifetime 在 current program 中形成，并被实际 Mojo/Weft 消费。实现内部的微块与 packing 保持可编程，既有局部实现继续复用，无 kernel-name/source-template 选择器或两套并行执行路径。 | 直接证据：lib/Dialect/CPU/Transforms/ReusePreparedInputs.cpp:18-61,64-115 按相同 implementation、maps、read-before-use 和 lexical lifetime 合并准备输入；lib/Dialect/CPU/Transforms/RealizeRegions.cpp:164-197 使用当前轴关系投影共享 slices；Mojo/Weft implementation 分别在 lib/Target/Mojo/Transforms/Implementations.cpp:36-114 与 lib/Target/Weft/Transforms/Implementations.cpp:12-45 展开，Weft lib/Target/Weft/Transforms/Legalize.cpp:804-884 消费当前 linalg IR。与 ref/tilelang/tilelang/tileop/gemm/__init__.py:121-139 及 ref/tilelang/src/transform/lower_tile_op.cc:1134-1151 一样由已选实现消费布局/资源需求，但本实现未引入整算子模板或 serializer 旁路。 |
| A4 | passed | brief.md | A4：通过现有生产 benchmark 入口获得代表性 CPU region-fold 与 region-scan 程序的真实 generated/source ms、G/S 和同次既定容差结果；Mojo/Weft 的 native 调用均有实际区域程序运行证据。相同案例使用同算法、shape、dtype、资源预算及完整调用范围，数据进入项目 CSV；受影响 GPU 和既有 CPU 项只复用必要性能运行，未受影响结果保留，不以生成成功代替执行或性能结果。 | 直接证据：report/baselinev2/mojo-x86.csv:2-3 记录 fold 0.278042/0.295968 ms、scan 0.200169/0.291698 ms；report/baselinev2/weft-rvv.csv:2-3 记录 fold 7.223379/109.298485 ms、scan 3.423664/1.904748 ms；均为 pass。examples/repro/v2/measurement.py:263-333 在 native timing 后执行同次容差比较，Mojo/Weft adapters 由 examples/repro/v2/providers/mojo/attention.py:8-21 与 weft/attention.py:101-121 连接独立 source 和完整调用范围。受影响 Triton 记录 report/baselinev2/triton-5090.csv:5，已通过 CUDA graph 与同次容差检查。 |

## 检查

_没有记录 Runtime 检查。_

## 阻塞项

_无。_

## 风险与跳过的工作

- Weft region scan generated/source ratio 为 1.797437，未设统一性能门槛。
- Triton attention ratio 为 2.385197；modern FlashAttention source 仍记录 shared-memory resource gap。
- 新增区域能力覆盖 f32 与 bool/index/integer；既有 Q4_K/Q8_K 保持，不代表全 dtype/设备覆盖。

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | pass | — | 独立只读核对通过 A1-A4；本轮未重复编译、benchmark 或测试，采用已存在的生产构建、native 性能与容差证据。 | 2026-09-11T00:57:28.759Z |



## 结论

独立只读核对通过 A1-A4；本轮未重复编译、benchmark 或测试，采用已存在的生产构建、native 性能与容差证据。
