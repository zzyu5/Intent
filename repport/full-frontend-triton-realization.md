# Intent Kernel 编译器当前能力与缺口审计

> 审计代码基线：650b710
>
> 审计日期：2026-08-09
>
> 本文只描述当前实现事实。doc/ 中的文档是目标规范；文档写到了某个能力，不等于代码已经端到端兑现。

## 结论

要求没有全部完成，而且还差很多。

当前系统已经不是 stable-softmax 的专用链路：Python frontend 会直接构造唯一的 canonical Intent Kernel MLIR；C++ realizer、Machine Plan、三个 target dialect 和共享 emission driver 已经存在；softmax、contraction、streaming、ragged 等多种前向结构能够生成真实的 Triton、cuTile、TileLang 代码并运行。实现中没有按 kernel 名字分派的 realizer，也没有每个 kernel 一套 emitter。

但它目前更准确的定位仍是：

**一个覆盖多种前向结构的 GPU 算子编译器原型，而不是一门已经完备、可用于训练和部署的算子编译器。**

最核心的未完成点不是再缺几个示例，而是：

1. realizer 仍然是有限结构模式的组合，不是可任意叠加的调度部件；
2. emitter 仍会从 traversal、role 和逻辑维度重新推导部分物理索引，不完全是对已确定物理计划的机械打印；
3. frontend 和 Kernel IR 的语言面大于当前 GPU 后端真正可 lowering 的子集；
4. 多输出、optional runtime 参数、训练反向、作者主导的多 kernel 编排、BF16/FP8/量化等基本能力尚未贯通；
5. 最近的共享实现改动以后，没有重新取得 10 个旧 kernel × 3 个后端的完整数值与性能矩阵，因此不能继续引用此前的 30 项性能结论。

## 一、逐项验收

状态含义：

- **完成**：当前代码中已经形成端到端机制，并有实际 repro。
- **部分完成**：已有真实路径，但只覆盖了需求的一部分，或最新代码没有完成全量复验。
- **未完成**：只有设计、局部语法或上游参考，尚无端到端能力。

| 要求 | 状态 | 当前事实 |
|---|---|---|
| Python 不保留独立 typed Kernel IR，直接构造 canonical Kernel MLIR | 完成 | Python 只保存解析/lowering 临时状态；唯一语义 IR 是 Intent Kernel MLIR |
| C++ Kernel IR 有正式 dialect、解析和 verifier | 完成 | Intent ops/types 已注册；进入 realization 和 emission 前会执行 Kernel IR 与 MLIR 验证 |
| Kernel IR、realization、target emission 职责分层 | 部分完成 | 大边界已经成立；但 emitter 仍重建部分物理索引，realizer 仍带有限模式机 |
| 三个 GPU 表面语言共享机器决策 | 部分完成 | 共享 Machine Plan 和 driver 已成立；target 叶子没有按 kernel 裂变，但投影层仍承担少量结构推导 |
| 不按 kernel 名字分支 | 完成 | 分析以 op、region、role、domain 为依据，没有 softmax/gemm/attention/moe 名字分派 |
| realization 是可自由组合的部件 | 部分完成 | ragged ownership 与 ordered stream 已成功组合一次；多 ragged、多 stream、staged 与 ordered 的一般组合仍不成立 |
| 10 个前向 kernel 覆盖三个后端 | 部分完成 | 旧阶段曾全部跑通；最新共享改动后只完整复验了 10 个 Triton 路径和 varlen attention 的三个后端 |
| 变长 attention 同时组合 irregular ownership 与 stateful stream | 部分完成 | 一个 ragged relation + 一个 ordered stream 已端到端跑通；不能据此声称机制可任意叠加 |
| 作者能表达“只读到当前位置” | 完成 | DSL 的逻辑 I.end 进入 intent.region_end，再进入 stream stop_node，三个后端按同一语义发射 |
| 非整除形状由编译器处理 | 部分完成 | 已处理 varlen attention 的 sequence 尾块；任意 GEMM M/N/K 尾块、staged contract 尾块和更多 producer 仍未覆盖 |
| 多输出 | 未完成 | kernel 只能通过 Out/InOut view 写出；没有多结果 return ABI，也没有多输出 repro |
| optional runtime 参数 | 未完成 | runtime/view 默认参数会被 frontend 拒绝；只允许 constexpr 默认值 |
| 低秩输入沿另一轴广播 | 部分完成 | frontend/IR/当前 handler 有 broadcast；尚无专门压实该接口形态的端到端 repro |
| 混合 dtype | 部分完成 | 支持显式 cast 和 reduce/contract 累加 dtype；没有通用类型提升，也没有 f32 bias + f16 QKV 这类接口 repro |
| 反向 kernel | 未完成 | 当前示例全部是前向 |
| 作者主导的多 kernel 编排 | 未完成 | 现有多阶段来自编译器内部，不是 DSL 作者显式组织多个 kernel |
| 自动微分接入 | 未完成 | 没有 autograd 注册或 backward artifact 接口 |
| BF16、FP8、INT8/INT4 | 未完成 | 当前实际示例只覆盖 f16、f32、i32 |
| GEMM + bias + activation + residual + output quantization | 部分完成 | 已有 GEMM+ReLU、dual-GEMM gating；完整部署 epilogue 未走通 |
| 普通 Python 库调用 | 部分完成 | 有 intent.compile、CompiledArtifact 和可调用 runner；仍依赖源码树/PYTHONPATH 和显式外部编译器路径 |
| 编译产物缓存与 shape specialization key | 部分完成 | 下层各自有局部缓存；没有统一的 Intent 编译/专门化缓存 |
| 对作者友好的诊断 | 部分完成 | frontend 有源码位置；后端编译失败仍主要暴露原始 subprocess stderr/stdout |
| 全程数值与性能不退化 | 未完成验收 | 最新代码没有完成全部旧 kernel × 三后端的全量数值和性能复验 |

## 二、当前架构真正成立的部分

### 2.1 唯一 canonical Kernel MLIR

当前 frontend 的主链路是：

    Python DSL
      -> AST/constexpr/symbol/shape/region 临时状态
      -> canonical Intent Kernel MLIR
      -> C++ Kernel IR verifier
      -> GPU realization
      -> Machine Plan
      -> target realization dialect
      -> target emitter
      -> Triton / cuTile / TileLang artifact

Python 中没有另一套长期存在、与 MLIR 平行的 typed Kernel IR。frontend 在构造 MLIR 时完成语义检查和带源码位置的报错。

Intent dialect 也不是字符串协议。include/Intent/Dialect/Intent/IR/IntentOps.td 注册了控制流、shape、broadcast、reduce、scan、contract、ragged、buffer、atomic、fence、RNG、call/return 等 op；lib/Transforms/VerifyKernelIR.cpp 对函数 metadata、节点 ID、schema、属性和 region 结构做 C++ 侧验证。realization 和 emission driver 都在进入后续阶段前调用 verifier。

### 2.2 没有按 kernel 名字裂变

当前 realizer 读取的是：

- ABI 和 view；
- domain、parallel、ordered、ragged 等 region 结构；
- reduce、contract、load/store、pointwise 等 op role；
- shape、dtype、axis 和逻辑依赖；
- target capability。

它不读取 “这是 softmax” 或 “这是 MoE” 来选择整条路径。三个 emitter 也使用 per-op handler，不存在一个 softmax emitter、一个 GEMM emitter、一个 attention emitter。

这证明当前系统已经越过“根据 kernel 名称套模板”的阶段。

### 2.3 三个 target 是一个框架中的叶子

Triton、cuTile、TileLang 都经过相同的 Kernel IR 遍历、共享 realization 框架和 emission driver。各 target dialect 只保存自身确实需要显式表达的字段，没有强行把三门语言做成同一个字段表。

因此，当前代码已经具备“第四个 GPU 表面语言应新增 capability、target projection、target dialect 和 leaf handler，而不改 kernel 分析入口”的基本形状。

## 三、当前架构尚未成立的部分

### 3.1 SchedulePolicy 仍是有限模式机

lib/Target/Common/Realization/SchedulePolicy.cpp 当前不是把 ownership、traversal、streaming、staging 等部件任意组合，而是枚举并约束了几种已知结构：

- ragged 路径要求恰好一个 ragged relation；
- ragged ownership 固定为一个 outer member 加一个 tiled member；
- ordered 路径要求恰好一个 ordered stream domain；
- ordered-row、ordered-tiled、ordered-ragged 依赖固定的 domain/cardinality 组合；
- 没有 ordered stream 时，再分别进入 ragged、row、tiled 分支；
- staged 与 ordered 仍位于互斥分支，而不是两种可叠加机制。

varlen attention 的价值在于，它证明了“一个 ragged ownership + 一个 ordered stream”可以从现有部件组合出来；但它没有证明：

- 两个 ragged relation 可以共存；
- 两个独立 stream 可以共存；
- staged contraction 与 ordered stream 可以组合；
- ragged、staged、ordered 三者可以同时作用；
- 新结构只增加 op handler 而完全不改 policy。

所以“十个 kernel 恰好被已有 mapping 覆盖”不能等价为“调度机制已经可组合”。

### 3.2 emitter 还不够机械

三个 emitter 已经不做 kernel 级匹配，但它们仍根据 traversal、role、logical dimension 等信息构造 target 侧索引表达式。换言之，Machine Plan 已经提供了决策的大部分事实，却还没有把所有需要打印的物理事实显式化。

真正达到目标边界时，target emitter 应只做：

1. capability 检查；
2. 已确定概念到目标语法的映射；
3. 明确委托给下层的部分；
4. 对不支持的单个 op/概念就地报错。

当前实现接近这个方向，但尚未完全退化到纯机械投影。

### 3.3 frontend/Kernel IR 的语言面大于后端子集

语言表面和 canonical Kernel IR 已经定义或构造了很多能力，但 GPU realization/emission 只真正支持其中一部分。

| 能力 | frontend / Kernel IR | 当前 GPU 端到端 |
|---|---|---|
| runtime if / for / while | 已有 | 未形成一般 realization/emission |
| @intent.fn、call、return | 已有 | helper lowering 有基础；一般 call 不是当前 GPU 主路径 |
| reduce max/add | 已有 | 支持有限轴结构 |
| scan | 已有 | 未进入当前 plan/emitter 主子集 |
| reshape / transpose / broadcast | 已有 | broadcast 有 handler；reshape/transpose 未形成一般后端能力 |
| record / extract / select | 已有 | 未形成一般后端能力 |
| buffer / alloc / atomic / fence | 已有 | scatter-add 有特定支持；一般 buffer/atomic/fence 未贯通 |
| RNG | 已有 | 未贯通 |
| pointwise | 已有 | 只支持当前登记的一组 unary/binary/compare |
| contract | 已有 | 当前矩阵收缩路径要求特定 axis 和 f32 累加语义 |
| ordered state stream | 已有 | 支持当前单 stream 形态 |
| ragged ownership | 已有 | 支持当前单 ragged relation 形态 |

因此，不能用“op 已在 ODS 或 frontend 中存在”来宣称“语言构造已经可以 lowering 到三个后端”。

## 四、当前 kernel 与 baseline 覆盖

repro 入口当前接受 11 个 kernel：

1. softmax
2. layer_norm
3. rms_norm
4. logsumexp
5. gemm
6. dual_gemm
7. attention
8. varlen_attention
9. online_softmax
10. moe
11. grouped_gemm

其中前十个是上一阶段的固定前向集合，varlen_attention 是本轮新增的组合性样本。

### 4.1 最新代码上的复验范围

| 范围 | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| 原十个 kernel 数值复验 | 最新代码已通过 | 最近共享改动后未全量重跑 | 最近共享改动后未全量重跑 |
| varlen attention，causal=False | 通过 | 通过 | 通过 |
| varlen attention，causal=True | 通过 | 通过 | 通过 |
| 原十个 kernel 的最新完整性能矩阵 | 未重新取得 | 未重新取得 | 未重新取得 |

此前报告中的 30 项时延、赢家和比值来自更早代码状态。它们可以证明当时三条路径能运行，但不能作为 650b710 的当前性能验收结果。

### 4.2 上游 baseline 的真实性

“生成代码可运行”和“存在公平的上游 baseline”是两件事。当前 baseline 状态如下：

| kernel | 上游 baseline | 比较限制 |
|---|---|---|
| softmax | Triton / cuTile / TileLang | 三者都有直接参考 |
| layer_norm | Triton / cuTile | TileLang 缺失 |
| rms_norm | Triton / TileLang | cuTile 缺失 |
| logsumexp | 无 | 只能做 reference 数值校验 |
| gemm | Triton / cuTile / TileLang | 三者都有直接参考 |
| dual_gemm | 三者都有 | 上游是组合 baseline，不完全等同单个融合 kernel |
| attention | 三者都有 | 算法、外层包装和计时边界需要逐项核对 |
| online_softmax | 三者都有 | 存在算法结构不完全一致的情况 |
| moe | 三者都有 | 上游常含排序、分组、多 kernel 或 wrapper，不能直接把比值当单 kernel 结论 |
| grouped_gemm | 三者都有 | adapter 和输入组织可能不同 |
| varlen_attention | 仅 TileLang causal=True | Triton/cuTile 没有当前可比上游；causal=False 也无上游数值 |

没有上游 baseline 不表示生成 kernel 不可运行，只表示不能得出性能对齐结论。

## 五、本轮 varlen attention 到底证明了什么

### 5.1 两种已有机制确实组合了一次

varlen attention 同时需要：

- ragged ownership：不同 sequence 的 token 区间不规则；
- ordered state stream：沿 key block 推进 online-softmax 状态；
- logical read stop：causal 情况下只读到当前 query block 可见的位置；
- 非整除尾块：每条 sequence 的长度不是 BLOCK_SIZE_K 的整数倍。

这不是新增一个 “varlen_attention schedule”。它复用了现有 ragged relation、ordered stream、state、contract、reduce 和 mask/padding 概念，因而是当前架构通用性的一次真实正证。

### 5.2 causal 上界来自作者语义，不是 kernel 猜测

DSL 用 I.end 表达逻辑读取终点。它被 lowering 为 intent.region_end，realizer 将其绑定到 stream stop_node，Triton 投影后的循环上界为：

    range(
        0,
        tl.cdiv(
            tl.minimum((query_block + 1) * BLOCK_SIZE_Q, sequence_length),
            BLOCK_SIZE_K,
        ),
    )

它与上游 flash_attn_triton.py 中“把 causal key 循环上界收紧到当前 query 可见范围”的结构相同。无效的 key block 不会启动，而不是启动后只靠 mask 丢弃。

这项能力属于作者的算法陈述：作者知道读取范围，编译器负责把它保持到物理循环。

### 5.3 非整除尾块的当前覆盖

本轮输入 sequence lengths 为：

    [4093, 3961, 3833, 3701, 3571, 3449, 3319, 3187]

总 token 数 29114，所有 sequence length 都不能被 64 整除。三个后端实际走过 packed Q/K 的 ragged 尾块以及 softmax/PV 的有效性约束。

当前 Machine Plan 中的 intent_plan.padding 不是 target emitter 临时猜 mask，而是：

1. 从逻辑 iteration domain 得到有效区间；
2. 把有效性绑定到产生该值的 producer；
3. 在 producer 处融合 predicate；
4. 按消费语义选择填充值；
5. 三个 target 只投影同一 padding 决策。

reduce max 使用负无穷填充，add/contract 的无效贡献使用零填充。

但当前 padding 机制仍然很窄：

- 只会物化到 binary 或 mask producer；
- 只支持 zero 和 negative-infinity 两种 fill；
- staged contract 暂时跳过；
- block argument、load、unary、broadcast、contract、reduce、gather 等 producer 不能普遍承接 padding；
- 本次 head dimension 为 128，收缩维仍是整数倍；
- 没有证明任意 GEMM M/N/K 尾块。

所以它证明的是“真实的 ragged sequence 尾块路径”，不是“一般非整除形状已经解决”。

### 5.4 当前可引用的实测数据

输入是 8 条变长 sequence、总 token 29114、head dimension 128。以下是 650b710 上当前 varlen attention 的真实结果：

| provider | causal | 数值 | generated p50 / p95 | upstream p50 / p95 |
|---|---:|---|---:|---:|
| Triton | False | PASS，max error 3.0518e-05 | 0.4260 / 0.4312 ms | 无 |
| Triton | True | PASS，max error 1.2207e-04 | 0.3211 / 0.3262 ms | 无 |
| cuTile | False | PASS | 0.3820 / 0.3879 ms | 无 |
| cuTile | True | PASS | 0.2860 / 0.2978 ms | 无 |
| TileLang | False | PASS | 0.4883 / 0.4992 ms | 无 |
| TileLang | True | PASS，max error 1.2207e-04 | 0.3476 / 0.3550 ms | 0.2919 / 0.3037 ms |

唯一可做直接上游比较的 TileLang causal=True 中，generated/upstream p50 为 1.1908，即当前生成实现慢约 19.08%。它已经处于相近量级，但尚未达到“同水平或更好”。

## 六、为什么离“能用的算子编译器”还远

### 6.1 训练链路为空

当前没有：

- 任意一个 backward kernel；
- 由作者陈述的多 kernel backward orchestration；
- 跨并行边界的归约编排；
- recompute 与 saved intermediate 的接口；
- autograd 注册；
- forward artifact 与 backward artifact 的连接。

source 中已有可对照的 FlashAttention backward、LayerNorm backward、SiLU-and-mul backward、split-K reduce、fused add RMSNorm backward 等上游实现。它们目前只是参考源，尚未变成 DSL 和编译器能力。

### 6.2 ABI 还不够表达真实算子

当前 kernel 不允许返回 SSA/Python 值，写出依赖 Out/InOut view。这个模型可以继续承载多个输出 buffer，但代码尚未把“一个 kernel 同时写 output 与 log-sum-exp”压成正式 repro。

runtime tensor 参数不能带默认值，因此无法自然表达：

- bias=None；
- residual 可选；
- 某种配置下才存在的 scale/metadata；
- 向量 bias 与矩阵 bias 两种 ABI。

constexpr 默认值已经支持，但它不能替代 runtime optional ABI。

### 6.3 dtype 基本盘缺失

当前示例的实际 dtype 是 f16、f32、i32。训练常用 BF16，推理常用 FP8、INT8、INT4 和混合 scale/zero-point metadata，这些均未端到端进入 DSL、Kernel IR、realizer、target capability、emitter 和 runner。

已有 explicit cast 和 accumulation dtype 只是必要基础，不等于具备混合精度算子能力。

### 6.4 部署 epilogue 只覆盖了片段

GEMM+ReLU 和 dual-GEMM gating 已证明同一调度中可以容纳收缩与融合点运算，但还没有覆盖真正部署常见的：

    GEMM -> bias -> activation -> residual -> output quantization

这里会同时压到不同 rank 输入、广播、混合 dtype、量化参数和输出存储语义，当前尚无完整路径。

### 6.5 还不是正常可安装、可缓存、可接框架的库

现在已经有可导入的 Python API、intent.compile、CompiledArtifact、launcher 和 callable runner；这比只有 shell repro 更进一步。

但仍缺：

- 标准 Python packaging/install 入口；
- 统一的 artifact cache；
- 明确的 shape/dtype/constexpr/target specialization key；
- framework/custom-op/autograd 接入；
- cuTile、TileLang 与 Triton 对等的后端 IR/编译信息采集；
- 将 backend 原始错误转换为 DSL 作者可理解诊断的边界。

各下层自己的 JIT/tuner cache 不能替代 Intent 编译器层的 artifact cache。

## 七、当前最准确的完成边界

可以确认已经完成的是：

1. Python frontend 直接构造 canonical Intent Kernel MLIR；
2. 正式 Intent dialect、C++ verifier、Machine Plan 和三 target dialect；
3. 不依赖 kernel 名字的 per-op realization/emission 主框架；
4. 10 种前向结构曾经在三个 GPU 表面语言上端到端运行；
5. varlen attention 把单 ragged ownership、单 ordered stream、logical stop 和 sequence tail 组合起来；
6. causal loop 上界确实收紧，三个后端当前 varlen 路径数值通过；
7. target 叶子已经主要表现为 capability、projection 和 per-op emission，而不是整 kernel 模板。

不能声称完成的是：

1. realization 的一般可组合性；
2. emitter 已经完全不做物理推导；
3. Kernel IR 中全部语言构造都能进入三个 GPU 后端；
4. 一般非整除 shape；
5. 多输出和 optional runtime ABI；
6. backward、多 kernel orchestration 和 autograd；
7. BF16/FP8/INT8/INT4 与量化 epilogue；
8. 当前代码下 10/11 kernel × 3 backend 的完整性能不退化；
9. 可安装、统一缓存、可直接接训练框架的产品级 Python 库。

## 八、后续工作的正确顺序

继续堆第十二、第十三个前向 kernel 不能解决当前主要问题。合理顺序是：

### 第一优先级：把有限模式变成可组合 realization

- 将 ownership、traversal、ordered stream、staging、ragged relation、boundary/padding 从互斥 pattern 分支中拆成有明确输入输出的独立决策；
- 允许同一 kernel 出现多个 relation/stream；
- 让 plan 显式携带 emitter 当前仍在重建的物理索引事实；
- 先用 staged + ordered、multi-ragged 等组合样本验证，而不是新增 kernel 名字分支；
- 补全 staged contract 和一般 producer 的尾块语义。

### 第二优先级：用一个真实 backward 压实接口和编排

- 从 source 中选择有三后端或至少一个高质量上游 baseline 的 backward；
- 同时压到多输出、recompute、跨并行归约和作者主导的多 kernel orchestration；
- 保持逻辑语义在 DSL/Kernel IR，物理分解在 realization；
- 不把 backward 做成一个 kernel family matcher。

### 第三优先级：补基本 dtype 与部署接口

- 先贯通 BF16；
- 再选择一个真实 FP8 或 W4A8 上游 kernel；
- 用 GEMM+bias+activation+residual+quantization 压实低秩广播、mixed dtype 和输出量化；
- target 不支持时由 capability 明确拒绝，不做静默降级。

### 第四优先级：把编译器变成可复用库

- 定义统一 specialization/cache key；
- 让 compile 不必每次重复 frontend 和外部编译；
- 建立可安装入口；
- 接一个 framework custom op 与 autograd；
- 将 target 编译诊断映射回 DSL 源码位置。

每完成一个共享机制，都应使用现有的单条 repro 入口实际 emit、运行并做数值比较；性能变更则重新取得受影响 kernel 的三后端数据。不能再用早期代码的性能表替代当前代码验收。

## 九、可直接复核的入口

当前唯一应使用的活体入口是：

    ./examples/run/repro.sh triton varlen_attention
    ./examples/run/repro.sh cutile varlen_attention
    ./examples/run/repro.sh tilelang varlen_attention

原十个 kernel 也通过相同入口，把最后一个参数替换为：

    softmax
    layer_norm
    rms_norm
    logsumexp
    gemm
    dual_gemm
    attention
    online_softmax
    moe
    grouped_gemm

关键实现证据位于：

- doc/compiler/architecture.md：规范中的层次边界；
- doc/compiler/kernel-ir.md：规范中的 canonical Kernel IR；
- doc/compiler/physical-plan.md：规范中的 realization 与 search space；
- python/intent/frontend/source/signature.py：当前 ABI/default/return 限制；
- include/Intent/Dialect/Intent/IR/IntentOps.td：正式 Intent op schema；
- lib/Transforms/VerifyKernelIR.cpp：Kernel IR verifier；
- lib/Target/Common/Realization/SchedulePolicy.cpp：当前有限 schedule policy；
- lib/Target/GPU/Realization/Analysis/Operations.cpp：当前可分析 op 子集；
- lib/Target/GPU/Realization/Plan/Build.cpp：当前 plan handler 与 padding 物化；
- lib/Target/Common/Emission/Driver.cpp：共享 emission driver；
- lib/Target/Triton/Emission/Handlers/Operations.cpp：Triton per-op 投影；
- 对应的 cuTile、TileLang Emission/Handlers/Operations.cpp：另外两个 target 叶子。

## 最终判断

这轮工作证明了架构已经具有真实的通用骨架，也证明了一次 ragged + streaming + causal stop + tail padding 的组合，不是软最大值模板的改名。

但“能覆盖一组前向 kernel”和“是一门完备可用的算子编译器”之间仍有明显距离。当前最应该解决的是组合模型、后端兑现边界和训练/ABI 基本盘；在这些完成前，不能把项目描述为已完成。
