# CPU execution family 与 Mojo provider

## 1. 完整目标与权威

Intent 能将本规格范围内的同一份作者算法，经 canonical KIR 编译为 CPU native artifact。作者不写 x86/provider 分支、线程数、SIMD 宽度或 cache tile。`doc/programming-model/` 与 `doc/dsl/` 的逻辑值、控制、数值、effects 和接口语义不因 CPU target 改变。

本 change 为现有 `doc/compiler/README.md` 中的 CPU execution family 建立真实路径，并在 Build 中将稳定 CPU program、pass 和 provider 边界补入 `doc/compiler/`。change 规格不能借 GPU-only 实现反向缩窄语言。第一轮支持下述明确子集；范围外能力必须在已掌握足够信息的层显式拒绝，不假装全 DSL 已可执行。

## 2. 编译分层

```text
Intent source → specialization / canonical KIR / verification
                             ├─ GPU construction → 既有 GPU providers
                             └─ CPU construction → CPU physical passes
                                                  → Mojo legalization / serialization
                                                  → Mojo compiler → native x86 artifact
```

CPU 与 GPU 共享 frontend、immutable KIR、可适用的 canonical analysis 和外部编译调用协议，不共享 GPU program ids、block topology、warp、MMA fragment 或 GPU storage 假设。execution family 的选择发生在 physical construction 之前。provider 与 hardware capability 是编译上下文的不同维度。

CPU construction 产生独立、完整的 executable program。其后 passes 只改当前 CPU program；origin 可用于诊断及语义对照，不能由 serializer 或 runtime 回读 KIR 来补 loop、access、ownership、workspace 或执行次序。

## 3. CPU physical program

优先使用适合的 MLIR `func`、`arith`、`scf`、`vector`、`memref` 等已有表达。缺失的 CPU task/ownership/capability facts 采用最小、typed、可验证的 CPU 表达，不为与 GPU 文件数对称而造一整套 dialect，也不让标准 dialect 的选择抹掉所需执行语义。

在进入 Mojo serialization 之前，当前 program 必须显式包含：

- 外部 views 的 dtype、logical extent、runtime element strides、offset、读写方向和必要 alias 事实；只有满足前置条件的 accesses 才能使用 contiguous/aligned/noalias 形式。
- 任务拥有的 logical worksets、唯一写入关系、线程预算约束、任务间依赖，以及同步完成边界。unordered 工作可并行；ordered control 与依赖不能随意改为 parallel。
- physical loop bounds、steps、tail guards/masks、loop carries、vector types 及真实访问坐标。inactive lanes 不读写越界地址，不能依靠 full-width load 后再 select 修补。
- reduce 的初始化、局部累加器、合并图及 source-order/数值保持条件。采用 SIMD reduction 不自动获得任意 permutation 的许可；需要的重结合必须来自相应 Intent operation 的契约。
- contraction 的 M/N/K blocking、SIMD accumulator 与 operands、数据复用和 materialization。需要 packing 时，packing loops、storage、ownership 与 lifetime 都属于当前 program。
- 中间值的 SSA、重算或显式 buffer 及生命周期；禁止只在旁表写“reuse/packing/vectorize”标签而把执行留给 emitter 决定。

初始合法程序可以用于 construction，但交付的三类程序必须经过真实结构优化，不以通用逐元素串行 interpreter/标量 loop 作为完成形态。

## 4. CPU transformations

每个 transformation 按 `doc/compiler/passes-and-analyses.md` 的纪律声明输入事实、legality、实际改写、保持语义和 analysis 失效范围。pass 边界是保持完整 program 的 transformation group，不把只修字符串或旁表的步骤宣称为优化 pass。

本轮须形成以下可复用能力，并由三类正式 benchmark 共同约束：

1. 从当前 workset、dependence 和 access facts 形成 task partition 与连续工作粒度；保留 ordered 关系和唯一写入。
2. 将已证明适合的 access/value graph 向量化，形成显式 SIMD loads、arithmetic、stores 和合法 tails；逐点 producer/consumer 的融合是 def-use rewrite，不是算子模板。
3. 将行归约变成局部累加、合法合并和 consumer 所需的遍历；保留 weight broadcast、reduction result 和输入复用关系，不物化不必要的整张中间 tensor。
4. 为普通 contraction 形成 cache blocking、register/SIMD accumulator 和可证明必要的 packing/materialization；该决定消费 contraction axes、layout、reuse 和 target facts，不匹配 GEMM kernel 名称。

Mojo/LLVM 可继续执行其拥有的 inlining、机器向量 legalization、指令选择、寄存器分配和机器调度。Intent 不复制这些下层优化，也不以“LLVM 会优化”为理由省略 source provider 必须得到的程序结构。

## 5. Mojo provider 与数值边界

Mojo legalization 消费 CPU current program 和 x86 capability，形成该 provider 真正需要的合法 API/type 形式。普通拼写差异直接机械映射；只有多个 consumers 或独立 legality 确实需要时才增加 provider-local expression。

Serializer 只发出 imports、签名、已存在的 loops/values/accesses、SIMD operations、已声明的任务调用和 ABI entry。它不得根据 kernel/role 名称选模板，现场选择分块，创造 packing、scratch、循环、tail、reduction tree 或候选参数。

普通 floating operations 保持 Intent 的默认语义。已安装 Mojo 的 `--fp-mode` 默认允许跨语句 contraction；生产调用必须显式关闭这种全局隐式融合（`--fp-mode=contract=off`），再把 Intent 允许的 contraction/FMA 实现限定在对应 IR operations 上。不能全局开启 fast-math、FTZ 或把普通 rsqrt/除法自动改为不满足契约的近似原语。

本轮用 x86 SIMD 实现 f32 运算和 f32 contraction accumulator，不要求 AMX。target triple、CPU features 和相关编译选项必须进入 artifact 的编译身份；不能把安装主机的隐式默认值当作所有设备均可运行的保证。

## 6. Native artifact、ABI 与调用

公共 Python 编译入口可选择 Mojo CPU target，保持既有 kernel definition 与 host orchestration 方式。编译器生成 Mojo source 并通过官方 `mojo build --emit shared-lib` 建立 native shared library；不把 `mojo.importer` 的隐式自动编译当作本项目 artifact 协议。

外部调用使用明确的 C ABI。CPU tensors 的 storage 指针、logical extents、element strides、offset 与 scalar arguments 按声明绑定，runtime 保持其 lifetime；支持的 CPU views 不通过 Python object arithmetic 搬运元素，也不暗中复制到 GPU。不能因 raw pointers 的存在而默认为 noalias。

初期优化覆盖 contiguous f32 views 与上述三类算法所需的 scalar/broadcast 访问。更广 stride、dtype、effect 或控制形态若尚未实现，必须在 native 调用之前给出明确的 unsupported 诊断；不能通过隐式 contiguous copy 或另一算法蒙混支持。

一个 specialization 对应一个 host-visible artifact entry。调用完成前，其内部 CPU tasks 必须 join；输出由既有 host/runtime 协议分配。若需要 scratch，layout、大小、alignment 和 lifetime 来自 physical program，runtime 仅实现已声明的资源绑定。不能缓存输入内容或暗中建立跨 invocation 的 packed-weight 状态。

调用 Mojo 并行等依赖 runtime 的能力前，launcher 按官方要求初始化 Mojo runtime；SDK 初始化、library loading 和正常调用分离。未安装工具链、编译失败或能力不支持时直接报告，不切换到 Python/Torch 算法。

## 7. Specialization、缓存与有效 tuning

native 编译不取消按需 specialization。编译产物身份包含 canonical specialization、ABI、shape/dtype、影响 legality 的 stride/alias facts、CPU/provider 选项及 concrete physical bindings；artifact 复用和 autotune winner 是不同层次，均不写回 KIR。

CPU physical parameters 直接约束当前 program 的 task grain、SIMD/loop 粒度、cache tiles 或明确的 provider options。候选是有限完整 bindings，先删除可证明非法的组合，再编译与实测；没有合法候选则报告失败，不取另一默认算法。

候选数据放在 CPU transformations 或 Mojo provider 的对应职责内，不复用第一份 GPU shared 默认表。编译调用的 `tuning_config` 可覆盖本轮 CPU 已实现的候选域；具体 schema 与实际 consumer 同步定义到稳定 CPU 规格，不接受 kernel 名、source template 或算法开关。

默认候选预算保持小而有限，不形成参数全笛卡尔积或结构/算法搜索。至少一条正式 benchmark 包含两个以上真实不同的合法参数候选，通过同一次生产调优选择 winner 并复用。改变参数必须改变相关 IR/types/loops/accesses 或实际下层编译选项，不能只改变 generated variable 名称。

候选编译在总 CPU/内存预算内并行，注意 Mojo 编译器自身的线程数，不能让多个默认全核编译器过度订阅。算子计时互不干扰；调优试跑不得污染调用者的读写状态。编译与调优耗时不充当算子性能。

## 8. 首批横向 corpus 与性能口径

本轮仅建立下列三条正式 CPU benchmark，不派生额外 shape、边界或线程数扫描矩阵：

| Case | 作者算法 | 外部 dtype 与 shape | 数值检查 |
| --- | --- | --- | --- |
| Batched row affine | 复用 `examples/kernels/pointwise/batched_affine.py` 的 `batched_row_affine` | f32，`B=17, M=257, N=4093` | 沿用现有 case 的最大绝对误差 `2e-6` |
| Weighted RMSNorm | 复用 `examples/kernels/normalization/rms_norm.py` 的 `weighted_rms_norm` | f32，`M=8192, N=4096`，epsilon=`1e-6` | 沿用现有 case 的最大绝对误差 `5e-5` |
| Dense GEMM | 在现有 contraction 模块补不含硬件选择的 ordinary f32 `I.matmul` 变体 | f32 输入/累加/输出，`M=N=K=1024` | 新增 f32 case 预先固定 `atol=2e-4, rtol=1e-5`；不继承 f16/bf16 容差 |

baseline 是同算法的手写 Mojo SIMD 程序，使用合理的任务分区、向量化与相应归约/GEMM 分块；不能使用刻意低效的标量 baseline，也不能让 generated 直接调用 baseline。复用或改编 Modular 实现时保留实际来源和必要许可；自主编写的 baseline 不伪称 upstream 原文。source 与 runtime 相邻，registry 只绑定 callable closure，不参与 compiler policy。

以现有 `examples/repro/v2/` 为唯一生产入口，增加必要 CPU provider/runtime adapter 与 CPU timing support，不另外建立 benchmark 系统。generated/source 的输入 shape、外部 dtype、算法、线程预算和计时范围相同。首批固定在同一 NUMA 节点的 8 个物理核预算，双方一致；这是一组性能配置，不是 NUMA 或线程数优化项目。

CPU 使用 native monotonic timing 的批量重复调用测量已编译、已预热入口的算子执行时间，报告每次调用的 p50/ms；不称为 CUDA Graph 或 kernel launch 耗时。重复调用不能由 Python 逐次派发来冒充纯 native 算子时间。计时必须包含该次算子必需的 packing/repacking、scratch 使用、线程任务派发和完成同步。工具初始化、编译/调优、动态库加载、输入/输出及可复用 scratch 的预分配不计入；是否复用 scratch 与输入内容预处理的区别必须明确。

在同一次正式 benchmark 中，用既定容差检查 generated/source；容差内即可，不额外追逐 bitwise 或逐操作一致。检查失败如实记录并修复，只重跑受影响的性能 case，不新增测试或放大容差。

CSV 保存真实 `generated_p50_ms`、`source_p50_ms`、`ratio = generated/source`、状态和必要说明，落在现有项目性能结果层级下。没有实际测量就不填性能值；不让表格承担历史审计或候选耗时报告。第一轮不设统一 ratio 验收门槛，但必须同时满足真实结构优化、非劣化 baseline、三类可比较结果及数值条件，不能用仅能运行代替前述 compiler 能力。

## 9. 对照与交付边界

对照 `ref/triton` 或 `ref/tilelang` 的同类 pipeline/pass 边界，最终审查必须给出当前 Intent 与 ref 双方 file:line、具体差异和实际后果；再以 `ref/modular` 的 vectorize、CPU elementwise/reduction、matmul/packing 实现确认 CPU 程序所需结构。

性能不足的归因顺序为 CPU current program 的任务分区、blocking、ownership、访问/遍历、reuse/materialization，再到 Mojo local form、serializer、外部 compiler 和 measurement。baseline 和引用只提供证据，不定义语言语义、算子名称匹配规则或未经证明的归因。

架构检查只读代码与上述 benchmark 必需的编译产物；不创建或执行独立 ABI、数值、边界、回归、压力、兼容或组合测试，包括 `/tmp` 中的变相测试。工具安装只检查工具版本、实际 CLI 能力和插件/MCP 配置，不能据此宣称后端实现已通过验收。

SDK、编辑器配置、skills、MCP 配置和缓存安装在项目之外；保留用户无关设置，不引入账号或机器路径到正式 source。完整 change 的验收以 brief 的 A1–A5 为准；创建分支和完成 Shape 不代表 CPU backend 已经实现。
