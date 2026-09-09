# 目标

补齐共同 CPU 程序中的普通 f32 逐点数学与归约组合，使现有 stable Softmax 和 weighted LayerNorm 作者算法经真实 CPU passes 形成 Mojo native artifact，并从同一 CPU 程序生成 Canonical Weft IR。推进的是可复用的 arithmetic/reduce、访问与 producer/consumer lowering，不是两个算子模板。

# 范围

- 沿用已归档 cpu-mojo-execution 的 CPU family、typed ABI、Tasks、AxisRelations、Reduce 与 provider 边界；不另起 KIR→Weft 或按 kernel 名选择实现。
- 保留 stable Softmax 与矩统计算法，接通普通 f32 exp、maximum_num、单逻辑轴上的 additive/maximumNumber combine，以及这些 reduce 与逐点 producer/consumer 的组合。允许共享 f32 stable_softmax 作者显式采用 numerator * (1.0 / denominator) 归一化；其它作者表达不变。
- 接受可与当前 contiguous CPU ABI 共同兑现的作者 stride constraints，至少覆盖 stable_softmax 的 strides=(None,1)、noalias；保留并兑现约束，不删除 annotation，也不暗中复制不支持的 view。
- 让任务内多次归约、共享输入/中间值、尾部、局部存储及后续归一化由当前 CPU IR、analysis 和 passes 承载。Mojo serializer 只拼写；Weft 只消费已经形成的结构。
- 本轮两条 native 性能项：stable_softmax，f32，8192×8192；weighted_layer_norm，f32，8192×4096，inverse_features=1/4096、epsilon=1e-6。均用单 NUMA 8 个物理核。
- 基线使用真实 Modular/MAX 库能力：Softmax 调用现成 CPU softmax；LayerNorm 用 MAX rowwise/ReduceSum/elementwise 组织与作者相同的矩统计表达。后者明确称为“MAX rowwise 同算法基线”，不冒称现成 MAX layer_norm。
- 只在既有 examples/repro/v2 Mojo registry/runner 中接入这两项，结果写入 report/baselinev2/mojo-x86.csv；原三项结果保留，仅在实际受影响时复用其性能运行。

参考均为实现依据，不是要求移植全部上游能力的需求来源：doc/、当前 CPU/GPU provider 源码、ref/triton、ref/modular、外部 TianchenRV，以及官方 TileLang Ascend passes。

# 非目标

- 不扩展 f16/bf16、非 contiguous/负 stride/InOut、transpose/batched contraction、scan/ordinary stateful loop、tuple/Welford reduction 或任意多轴 reduction。
- 不改变 stable_softmax 或 weighted_layer_norm 的作者算法，不将矩统计改写为 Welford、online softmax 或另一整算子程序。
- 不修改外部 Weft，不接其 native discovery/JIT/设备调用；不开始 DSA/Ascend、AMX/IME runtime。
- 不扩大 shape/dtype/线程数矩阵，不新增 benchmark 之外的数值、边界、回归、兼容或组合测试，不用临时命令绕过。
- 不新建 worktree，不修改无关 GPU 实现、设计 doc 或环境配置。

# 验收示例

- A1：现有 f32 stable_softmax 与 weighted_layer_norm 不改作者算法即可通过共同 CPU construction/transformations；前者的 stride/noalias 约束被保留并兑现，exp、maximum_num、additive/maximumNumber reduce 及其 producer/consumer 形成可验证的当前程序，而不是 emitter 中新增整算子逻辑。
- A2：两条程序均得到正式 Mojo native callable artifact，以及从同一结构化 CPU 程序生成的合法 Canonical Weft IR/任务接口；后者明确不可调用、不声称设备运行，未支持的语义不静默近似或 fallback。
- A3：两条固定 CPU benchmark 都取得 native 算子执行时间、真实同算法 MAX 库级 source 时间与 G/S，并在同次运行通过预定容差；两项分别 G/S≤1.05，结果进入既有 CSV。LayerNorm 基线明确是同矩统计的 MAX rowwise 路径，不用 Welford 数字冒充。
- A4：两条程序实际消费共同 CPU 的任务/向量、归约实现、融合和中间值生命周期机制；有限 tuning 只实例化合法物理参数并实测选优，产物/winner 继续复用。完整 lowering 不依赖 kernel 名、固定 case shape 或未实现的参数，相关新旧 executable path 不并存。

# 约束与不变量

- AGENTS.md 与 doc/ 为权威；本轮是实现既有语义，不修改语言规格来适应 provider。
- maximum_num 与传播 NaN 的 maximum/reduce.max 不混同；归约只作 logical-order-preserving reassociation，不把普通算术全局 fast-math。
- CPU 当前程序拥有任务、遍历、访问、复用与生命周期；外部 Mojo/Weft compiler 仍负责自己的机器表示、指令与资源实现。
- LayerNorm 保持 E[x²]−E[x]² 与现有输出表达。MAX rowwise wrapper 只提供高层计算 closure，不能手写较弱的 SIMD/线程/归约循环当作库基线。
- 算子时间包含当前 invocation 的计算、materialization、任务派发和 join；排除编译、tuning、加载及输入/外部输出分配。新增两项不测 CUDA launch 或 Python 调度时间。
- 新性能门槛与容差通过本 Shape 明确确认，不把归档 change 的历史验收自动继承成新门禁。

# 决策

- 已接受并归档 cpu-mojo-execution；本轮选择继续丰满 CPU 普通数学与归约能力，使用单个 Native change、main/current。
- 用户已确认本 Shape 的目标、两项固定性能输入与门槛、同算法 MAX 基线、Mojo native/Weft generation-only 边界及非目标，授权进入 Build。
- 用户已确认共享 f32 stable_softmax 的归一化表达调整：先以普通 f32 除法求一次倒数，再逐元素相乘，参考 MAX softmax.mojo:168–179、628–636。这是共享作者表达的显式改变，各 target 消费同一份新 KIR；不创建 CPU 特供版本，不隐式放宽 compiler 浮点语义，原算法、容差和性能门槛不变。f16/bf16 入口与 GPU compiler 不在此次表达调整范围内。
- 不拆 Supervisor：共同 reduction/current-program analysis、Mojo/Weft lowering 与两项性能反复涉及同一核心；按算子或 provider 拆分没有独立交付价值。
- 先做 CPU 的依据：KIRToCPU.cpp:185–215 尚无 exp/maximum_num；:43 对 stride annotation 一概拒绝。stable_softmax.py:25–26 正好携带兼容的行主序约束。Weft Legalize.cpp:387–410 只接受 add，但外部 Weft functions-and-operations.md:81、109 已定义匹配的 maximumNumber/max reduce；缺口在 Intent lowering，不需扩展 Weft。
- 参考差异：Triton ReduceOpToLLVM.cpp:230–317 消费 combine 与当前 layout 形成局部树，并更新 layout；Intent 应同样从 typed combine/axes 建立实际 IR，但保持自己的 logical-order 语义，不复制 GPU register/warp ownership。
- MAX softmax.mojo:736–798 以 CPU row tasks 调用三阶段 Softmax；algorithm/rowwise.mojo:1149–1303 提供库级 reduction 与 elementwise。MAX normalization.mojo:3233–3251 的现成 LayerNorm 使用 Welford，故不直接作为当前矩统计作者算法的同算法 source。
- Ascend 暂不作为本轮主线：当前 TileLang Legalize.cpp:21–31 使用 CUDA 32-thread warp 约束，官方 AscendLowerParallelToVector 与 CombineCV 则实际形成 vector issue、跨 Cube/Vector 的 copy/flag 依赖。本机仅有 npu-smi 路径不足以确认 NPU/CANN；后续独立 Shape 再决定 provider/compiler input 与设备边界，不改当前作者 surface。

# 待解决问题

无。共享 f32 Softmax 作者表达调整已获用户确认。

# 验证预期

只复用现有生产编译与 Mojo 性能 runner；Softmax 比较容差 atol=1e-6、rtol=1e-5，LayerNorm atol=5e-5、rtol=1e-5，在同次 benchmark 中检查一次，确认后不得为通过而放大。Weft 只做生成所需的生产编译器 legality/verifier，不运行设备或独立测试。架构核对只读、给出当前/ref 的具体差异与后果，不逐条扩为新的验收门禁。
