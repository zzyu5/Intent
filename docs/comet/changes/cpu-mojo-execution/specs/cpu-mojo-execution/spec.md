# 共同 CPU execution family 与 Mojo / Weft providers

## 1. 目标、语义与范围

同一 Intent source/canonical KIR 经共同 CPU 程序，产生 Mojo native artifact 或 Weft 生成产物。CPU 是 execution family；Mojo、Weft 是 provider；x86、RISC-V 及其 ISA features 是 hardware facts。RVV 不是与 CPU 平级、另从 KIR 建立一套程序的 Intent family。

作者继续表达逻辑值、dtype、structured operations、控制与 effects，不写 AVX/LMUL、AMX/IME、hart 或 Ascend C/V 分工。默认数值语义不改变。共同 CPU 模型可以表达 scalar、shaped/vector computation 及 matrix-eligible contraction，但不保证所有 provider/hardware 都有这些操作的可执行实现。

本轮 native 性能范围仅为第 8 节的三条 x86 f32 程序。Weft 仅完成这些程序所需的生成端，不接设备执行。Intel/Apple AMX、IME 的 native 实现和新增 dtype 性能项目不在本轮范围。

## 2. 唯一编译分层

```text
Intent source → specialization / canonical KIR / verification
    ├─ GPU construction → 既有 GPU providers
    └─ CPU construction → shared CPU Program / analyses / passes
                              ├─ Mojo legalization → Mojo source
                              │                      → Mojo/LLVM → native artifact
                              └─ Weft legalization → Canonical Weft IR
                                                     → 外部 Weft compiler（本轮不接设备 runtime）
```

CPU construction 消费 immutable KIR，并创建独立完整的当前程序。后续 analysis、passes 和 provider lowering 消费这份 CPU IR；origin 只用于诊断/语义对照，不能让 KIR、role 名称或旁表共同解释执行。

CPU 程序在进入 provider 前已决定本层负责的任务、block traversal、访问、数值 flow、复用与资源生命周期。下层输入仍可有完整的 structured reduce/contract；“结构化”不等于“未定义”。不得在公共层先将它们全部降为固定 AVX scalar/SIMD loops，再由 Weft 猜回 shaped axes 或 primitive。

## 3. 正式 CPU dialect 与可验证程序

CPU dialect 的 IR、analysis、transforms 按与 GPU execution family 一致的职责层次组织。标准 `func/arith/math/scf/memref/vector/linalg` 等仅在能保持所需语义时复用；同一 physical program 可混用标准和 CPU operations，这不增加另一层 planning IR。

CPU 专属的 program/ABI、逻辑轴与访问关系、任务和资源归属、numeric obligations、capabilities、physical parameter bindings 必须有 typed owner、合法 schema 与 verifier。不能只检查 `cpu.*` 属性存在，或由各 serializer/runtime 分别解释任意字符串字典。

当前程序至少明确以下事实：

- external views 的 dtype、shape/dynamic dimension identity、offset/stride、读写方向与 alias；access 的坐标、validity/fill 和唯一写入关系；
- 有限任务分区及每个任务拥有的计算/访问域，task captures、预算约束、依赖和完成边界；
- shaped value 的 logical axes、broadcast/result relation，physical traversal、bounds、tails 和 loop carries；
- reduce 的 axes、identity、combine/accumulator dtype、允许的结合与顺序，以及 result 的 consumers；
- contract 的 free/reduction axes、operand/result/accumulator flow，所选 cache/register blocking 与复用域；
- SSA、重算或显式 materialization；资源的大小、对齐、owner、初始化、生命周期及访问；
- 本层已经决定的表示转换和 provider 所需的合法输入，不携带可在 emission 失败后改选的备用程序。

CPU current-program analyses 从上述真实 IR 重算 coverage、dependence、access footprint、reuse、lifetime、资源估计和 provider eligibility。mutation 后相关 analysis 失效；canonical analysis 不能代替已经改变的 current-program facts。

## 4. Scalar、vector 与 matrix 的边界

共同模型不是“所有值都必须先变成一维 SIMD”。一个 shaped computation 可由 scalar issue、vector values、多份 accumulator 或矩阵表示实现，但不改变作者 logical elements、dtype、数值边界和 effects。

矩阵实现仍消费显式 contraction，而不是从某个 kernel 名称或整张图匹配另一算法。若某个 provider input 确实需要 matrix fragment、packing/conversion、配置状态或独立资源，这些事实由对应 legalization 物化为 typed 当前 IR。没有实际 consumer 的空 fragment、假 capability 或只有标签的 matrix 路径不算能力。

Mojo/LLVM 与 Weft 拥有其下层机器表示、指令选择及资源实现。Intent 不复制 Weft 的 RVV/IME Physical IR，也不在共同 CPU 层放 Intel tile 配置、LMUL、IME 指令或 Ascend 的 L0/UB/跨核 flags。共同 CPU IR 不要求所有 backend 拥有相同的矩阵寄存器结构。

当前调查到的 AMX/IME 同线程局部计算，不要求作者采用 Ascend 的双执行上下文。若将来真实后端需要异步 engine、wait/barrier 或新的可观察 effects，必须单独建立相应合法性与 IR 表达；本轮不预设该能力。

矩阵能力的模型、生成和设备执行分别陈述。全 f32 benchmark 不证明 Intel AMX BF16/INT8 或当前 Weft IME1 i8 路径已运行；不能为进入矩阵路径隐式改变输入精度或算法。

## 5. 真实 CPU transformations 与优化归属

按 `doc/compiler/passes-and-analyses.md` 的纪律组织可组合的完整 transformation groups：输入事实、legality、IR rewrite、语义保持、失效 analysis 与 group 后 verifier 均明确。函数式 helper 或 registered pass 只是工程形式，不以包装名称替代这些能力。

三类程序共同约束下列可复用变换：

1. 从 worksets、dependence 与 accesses 形成 task partition、连续 grain、coverage 与唯一写入。
2. 保留 shaped value/axes，融合逐点 producer/consumer，消除不必要的中间遍历和 buffer；按照访问证明形成向量/块计算和合法尾部。
3. 从 reduce schema 与消费者关系形成局部累加、合法合并和后续遍历，处理输入复用与数值保持，而不是固定选择一种 shuffle tree。
4. 根据 contract axes、reuse、capabilities 和资源形成 cache/register blocking、多行/多列 accumulator 与数据供应；必要 packing 的循环、存储、owner 和 lifetime 显式进入 IR。
5. 对 selected provider 做表示与接口 legalization，并验证每个 operation 有合法 spelling。

Blocking、ownership、traversal、reuse 和 materialization 的决定先在当前 CPU 程序中形成。Provider 可以继续执行其真正拥有的表示变换；单纯 API 拼写直接 serialization。Serialization 不创造 loop、scratch、packing、mask、归约算法或候选。

## 6. 两个 provider 的交付边界

### Mojo native

Mojo 首个可运行目标是 Linux x86-64 f32。公共 Python compile/runtime 保留 native shared library、明确 C ABI、CPU views 和同步完成的调用。

硬件 feature 是独立 typed capability；AVX2/AVX512 可以限制本轮 Mojo legalization，但不是整个 CPU family 的定义。向量宽度从 dtype、实际 feature 和所选 binding 得到。是否有矩阵 API/指令必须有实际 provider 证据，不能从 Mojo 或 AMX 名称推断。

Mojo serializer 消费 legalized IR。普通 floating arithmetic 保持 Intent 默认语义，禁用隐式全局 contraction/fast-math/FTZ；允许的 FMA 或近似只来自对应 operation。线程运行时改变 FP environment 时，合法化与 native runtime 保持调用所需语义和调用方状态。

Native ABI 保留 pointer、extents、strides、offset、scalars、access/alias 与资源要求。初期支持 contiguous f32 In/Out 及三例所需 scalar/broadcast；未支持的 stride、InOut、dtype、control/effect 在已获得足够事实时明确拒绝，不隐式复制、换算法或进入 CUDA。

一个 executable specialization 对应一个 host-visible entry，内部任务在返回前 join。输出由 host 分配；scratch 的大小、布局、alignment、owner 和 lifetime 来自当前程序。无跨 invocation 的隐藏 packed-input 状态。初始化、编译和加载与正常调用分开。

### Weft generation only

生成端消费同一 CPU 程序，将三类程序的任务内计算、typed axes、numerical operations、views、控制和局部生命周期映射到 Canonical Weft IR；不是从 KIR 另起一个 Weft construction，也不是按 kernel 名调用整算子 std 模板。

任务内计算与 host 调度的接口显式保存其输入、输出及工作域。Weft 本身不提供隐式 hart/task identity，生成端不向其语言强加 GPU grid 或 CPU worker 根对象。本轮尚未 materialize 的任务调度与设备调用不能被隐藏为“已经可运行”。

只生成入口返回 CPU/Weft 编译产物与明确接口，不返回伪造的可调用 native artifact。生成过程中的 verifier 属于生产编译；不调用 RISC-V native discovery、JIT、SSH、设备执行或性能 runner。Weft canonical numeric/effect schema 无法保持的组合明确诊断，不通过更换算法或绕过 Weft 发 intrinsic C 解决。

Weft compiler 是外部依赖，接收其正式 Canonical Kernel IR；`TianchenRV` 仓库保持只读。它的 layout、RVV/IME selection、physical packing、schedule/resource 和 intrinsic emission 不迁入 Intent。生成成功不等于下层编译、运行、数值或性能已通过。

## 7. Specialization、缓存与有效 tuning

`intent.compile(..., tuning_config=...)` 保留有限合法候选的配置能力。共同 CPU 参数和 provider-specific 参数分属真实 consumer；配置按职责与 transforms/provider 相邻，不使用 GPU 默认表或 kernel 名称。

本轮 CPU 参数覆盖已实现的任务粒度、cache/register blocking 和 provider 向量参数；改变 binding 必须改变当前 IR/types/loops/accesses 或真实下层编译选项。不能只在 8/16 宽度间切换，却把未实现的复用或多列 accumulator 称作 tuning 能力。

至少一条正式 benchmark 在两个以上真实合法候选中实测选优，并复用 winner。候选预算小而有限，不做参数笛卡尔积；合法性与资源约束优先于成本/性能。Tuner 不发明 source tree，不暗换算法，不在编译失败后使用未声明备用程序。

产物编译身份包含 canonical specialization、ABI、会影响代码生成或合法性的 view facts、provider/hardware/options 和完整 physical binding。动态维度可通过 ABI 使用，不强制每个运行 shape 重新编译。Winner identity 包含影响选优的 shape/dtype/stride/alias facts；产物与 winner 是不同缓存，均不写回 KIR。

准备/编译允许资源预算内并发；调优计时与正式计时互不干扰。试跑不污染调用者可观察的读写状态。编译、候选选择与缓存管理耗时不作为算子性能。

## 8. 三条性能 benchmark 与成熟基线

本轮不新增 shape、dtype 或线程数扫描，继续使用：

| Case | 作者算法 | 外部 dtype / shape | 同次检查 |
| --- | --- | --- | --- |
| Batched row affine | `examples/kernels/pointwise/batched_affine.py:batched_row_affine` | f32，B=17，M=257，N=4093 | 最大绝对误差 2e-6 |
| Weighted RMSNorm | `examples/kernels/normalization/rms_norm.py:weighted_rms_norm` | f32，M=8192，N=4096，epsilon=1e-6 | 最大绝对误差 5e-5 |
| Dense GEMM | `examples/kernels/contraction/gemm.py:gemm_f32` | f32 输入/累加/输出，M=N=K=1024 | atol=2e-4，rtol=1e-5 |

GEMM、weighted RMSNorm 使用成熟 Modular/MAX CPU 库实现作为主要 source；affine 使用库级 elementwise/fused expression 表达同一 x*scale+bias，不要求库恰有同名算子。接入真实库计算入口，注明实际 compile/runtime 边界；不能通过重写一个较弱的手工 SIMD 程序满足“成熟库”要求。若库接口无法形成所需同算法、同计时范围的基线，报告具体接口问题，不默默换回手写 baseline。

手写 SIMD baseline 可用于定位上层结构或 provider 差异，但其数字不代替成熟库结果。Generated 不调用整个 baseline 算子来冒充编译优化。source/runtime 按实际语言、Modular 来源和算子职责相邻组织，保留必要来源/许可；registry 只连接完整 callable closure。

双方使用相同算法、输入 shape、外部 dtype 和同一 NUMA 的 8 个物理核预算。库支持更多 dtype/layout、采用不同中间舍入或辅助表示，不自动禁止本 case 比较；相应差异如实注明。不能将另一 dtype 的 matrix 路径当作当前 f32 基线。

算子时间使用已经编译、加载和预热的 native 执行，报告每次调用的 p50/ms。不得将 Python 逐次调度时间与 native kernel 时间混比。每次算子需要的 packing/repacking、内部 materialization、任务派发与同步都计入；编译/调优、库初始化、输入和外部输出分配排除。若使用可复用 scratch，仅排除原始空间分配，不排除依赖输入内容的预处理。

每条 benchmark 在同次运行做一次原容差检查；失败如实修复，不扩大容差、追加数值测试或扩大矩阵。只复用受影响的性能运行确认。

**性能完成条件：三项分别满足 generated/MAX 执行时间比 G/S ≤ 1.05，不使用平均数、相对自身改善或手写基线结果替代。该门槛已由用户明确确认。**

性能结果继续写入 `report/baselinev2/mojo-x86.csv` 的真实 generated/source 时间、ratio、状态和必要说明。旧手写数字不改名为 MAX 测量；取得成熟库实际结果后更新相应条目。CSV 不承担历史审计或候选调优耗时报告。

## 9. 对照、迁移与完成边界

对照先定位当前 CPU 程序的 partition、blocking、ownership、访问/遍历、reuse/materialization，再判断 provider form、serializer、外部 compiler 和 measurement。对照必须给出 Intent 与 Triton/TileLang 的实际 file:line、差异及后果，并结合 Modular CPU 实现与 Weft 的 typed fragment/转换机制。不得仅凭 IR/dialect/pass 名称断言成熟或越界。

迁移后同一 CPU 编译请求只有一条 executable path；旧的任意属性协议、过早 scalarization 或 serializer 重建路径在对应新能力替换时删除，不留双主干和兼容开关。不改无关 GPU 算法、用户文件或 Weft 实现。

本 change 使用 `main/current`。此前合并只保留实现起点，不自动通过新验收。正式执行标准为 brief 的 A1–A5；模型设计、provider 生成、native 执行、容差通过和成熟库性能达标分别报告，不互相代替。

只修改必要的编译器、runtime、source 接口及现有性能入口；不创建计划/进度副本、独立测试或临时测试脚手架。SDK、缓存与生成临时文件留在项目之外。
