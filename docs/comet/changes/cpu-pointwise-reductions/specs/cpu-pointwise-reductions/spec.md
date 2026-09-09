# CPU 普通逐点数学与归约组合

## 1. 完整能力与分层

本能力在现有共同 CPU execution family 中实现 f32 逐点数学与单逻辑轴归约的组合。作者继续使用已有 Intent source/canonical KIR；CPU construction 创建独立可执行程序，CPU analyses/passes 决定该层的任务、遍历、访问、复用与中间值，然后由 Mojo 或 Weft legalization 消费。

本能力不引入另一套 planning IR、算子模板或 KIR→Weft 旁路，也不修改 doc/ 定义的语言语义。既有 CPU native ABI、Tasks、资源与 artifact/winner 缓存能力保留；未被本能力覆盖的语言构造仍明确 unsupported。

## 2. 支持的作者程序与接口

本轮闭合两个既有作者入口：

| 入口 | 作者表达 | 固定性能输入 |
| --- | --- | --- |
| examples/kernels/normalization/softmax.py:stable_softmax | maximumNumber reduce → exp(x−max) → sum → divide | f32，M=N=8192 |
| examples/kernels/normalization/layer_norm.py:weighted_layer_norm | mean、second moment → variance → rsqrt → scale+bias | f32，M=8192、N=4096，inverse_features=1/4096、epsilon=1e-6 |

这些入口用来约束通用 lowering 能力，不是 compiler matcher 的输入。Dynamic shape 仍由既有 CPU ABI 传递；不能只接受表内数值或以 case shape 替代 legality。

本轮 runtime 继续接受非空 contiguous f32 In/Out views 与现有 scalar 参数。作者 stride constraint 在与当前 contiguous ABI 相容时被兑现，至少包括 strides=(None,1)；noalias 仍由实际 allocation/view facts 验证。不能删 annotation、默认为满足冲突约束、偷偷复制输入，或把此项扩大为通用 strided-view 支持。

## 3. 数值与 structured operations

新增的普通 exp、maximum_num 保留 scalar/shaped element type、广播/逻辑轴关系、IEEE 特殊值与普通浮点边界，不隐式启用 approximate 或 FTZ。

maximum_num 忽略单侧 NaN，双方 NaN 仍产生 NaN；它与传播 NaN 的 maximum、builtin reduce.max 不是同一 operation。对应 reduction 的实际 identity/combine 必须保留；不能只因都叫 max 就改映射。Empty-domain 与 source-order 规则仍服从 doc/；provider 无法表达且不能合法物化的组合明确拒绝。

本轮支持现有单 f32 accumulator、单逻辑轴、无 canonical capture 的 additive 与 maximumNumber reduction，及其可合法融合的逐点贡献表达。共享层保持 typed combine、identity、axes、result consumers；本轮不承诺任意 tuple/multi-axis/Welford reduce。

Softmax 保持现有三阶段依赖；LayerNorm 保持 E[x²]−E[x]²，不把 ordinary arithmetic 改为 Welford 或另一方差方法。合法归约可重结合但不可任意 permutation；普通乘加不获得全局 FMA/fast-math 许可。

## 4. 当前程序与优化职责

任务、grain、vector realization、归约树、consumer 遍历、共享输入/中间值、局部资源初始化与 lifetime 必须在当前 CPU IR 中形成。Legality 来自 typed operation、def-use、coordinate/axis、effect 与 resource facts。

对 exp 结果同时被 sum 和最终输出消费、原输入同时被多个统计归约和输出消费等关系，passes 保持真正的多 consumer，不因局部单 consumer 假设丢值或重排 effects。Fusion、SSA reuse、合法 rematerialization/materialization 依照当前依赖形成，不按 Softmax/LayerNorm 名称挑预写结构。

每个完整 transformation group 重算受影响的 analysis 并验证当前程序。只有表示或 API 拼写差异进入 provider-local legalization；serializer 不创造 loop、scratch、归约或候选。既有 bindings 继续约束实际 task/vector 等实体；新参数只有存在真实 consumer 与有限合法域时才增加，不设整算法选择。

## 5. Provider 行为

### Mojo native

Mojo legalization 对形成后的普通 math、归约 realization、访问与任务做表示转换，并保留 native FP environment、同步 join、view/scalar C ABI 与资源生命周期。Serializer 只拼写合法 IR，Mojo/LLVM 负责其机器 lowering。

公共 intent.compile 产出真实 native artifact；运行不调用 Torch/MAX 算法替代生成计算。产物与 winner 缓存保持独立且身份完整，合法候选实测选优并复用，不能恒取第一份配置。

### Weft generation only

公共 intent.generate 消费同一份 shared CPU Program，产生 Canonical Weft IR、CPU 程序与显式任务接口；不产生可调用 native artifact。

现有 Weft maximum/minimum 与 max/min reduction 已采用 maximumNumber/minimumNumber 语义，exp 也已有 canonical operation。Intent legalization 按完整 dtype、axes、identity/combine 与访问事实匹配这些正式 operations，不能混同 Intent 的 propagating maximum；无法保持的组合 fail closed。

本轮不改外部 TianchenRV，不接 RISC-V discovery、JIT、SSH、调度或设备执行。遇到必须扩展外部 schema 才能完成的真实缺口时报告具体边界并回到范围确认，不绕过 Weft 直发 intrinsic 或改作者算法。

## 6. 成熟库基线与性能

Softmax source 调用 Modular/MAX CPU softmax 入口，采用同一 stable Softmax 算法。LayerNorm source 用 Modular/MAX 的 rowwise、ReduceSum 与 elementwise 库机制表达 sum(x)、sum(x²) 和归一化，与当前作者的矩统计方法一致。

LayerNorm wrapper 可以提供输入、贡献和输出的高层 closure，但任务分区、向量遍历、归约实现和缓存由成熟 rowwise 库承担；不得另写低效 scalar/SIMD loops、手动线程调度，或直接调用 generated 计算充当 source。表格明确写“MAX rowwise 同算法基线”，不是 MAX 的现成 Welford layer_norm。若安装的库接口不可用，报告实际缺口，不退为较弱手写 baseline。

双方同用单 NUMA 8 个物理核、相同 shape/外部 dtype、相同算法及 host-visible 调用范围。计时为编译、加载、预热后的 native 每次调用 p50/ms，包含当前调用需要的 materialization、任务派发与同步；排除输入/外部输出分配、编译、tuning 与库初始化，不混入 Python 调度或 CUDA launch 时间。

本轮仅新增表中两条固定性能项到现有 Mojo registry，使用 examples/repro/v2 的生产 runner，不扩大输入矩阵。Softmax 同次容差为 atol=1e-6、rtol=1e-5；LayerNorm 为 atol=5e-5、rtol=1e-5。失败据实修复，不放大容差。

拟定完成门槛为两项分别 generated/source≤1.05，不用平均数或自比改善代替。该门槛需本次 Shape 明确确认。实际时间、ratio 与必要 source/失败说明写入 report/baselinev2/mojo-x86.csv，不记录候选耗时和历史审计；旧三项未受影响不重跑，受影响只复用对应性能入口。

## 7. 非目标与完成边界

不扩展 dtype、通用 strided/InOut views、transpose/batched contraction、scan、有序 stateful loop、Welford/tuple reduction、AMX/IME 或 DSA/Ascend，不修改无关 GPU code 或语言 doc。

只使用性能 benchmark 加同次原容差；禁止额外测试与临时脚手架。Weft compiler 自身的生成 legality/verifier 属于生产编译，不等于设备验证。

正式验收使用 brief 的 A1–A4：通用 CPU lowering、两种 provider 的明确产物、两项同算法成熟库性能，以及真实可复用的 passes/tuning。当前 change 使用 main/current；Shape 确认前不修改项目实现。
