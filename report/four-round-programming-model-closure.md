# Intent 编程模型收敛后的四轮推进报告

## 报告范围

这份报告只覆盖连续四轮工作：

1. 按编程模型审计报告的登记表，修正 correctness、唯一语义来源、Plan binding 和名义存在但尚不能兑现的语言构造；
2. 先收紧 TileLang FP8 MQA 的错误能力声明，再用一次性探针主动撞语言拼写、语义来源与物理决定的边界；
3. 补完 generic reduce/scan combine 与 stage execution contract，同步规范并冻结编程模型；
4. 在 RTX 5090 D 与 H100 上完整重跑，同时加入五个冻结后才选择的真实小算子，处理它们暴露的共享问题。

起点不是“文档已经正确，只需照表实现”。第一轮开始时，[编程模型审计](programming-model-and-compiler-architecture.md)已经明确：文档、历史设计和当前代码必须重新对照，算法语义、派生事实、已选物理决定与 target spelling 要分别只有一个权威来源。四轮工作的共同目标，是让真实实现满足这条边界，而不是为了把矩阵涂绿给算子加特例。

四轮结束后的主链仍然是：

```text
Python DSL
    ↓
canonical Intent Kernel MLIR       唯一算法语义
    ↓
shared semantic facts              可重算的 provenance / use-def / legality
    ↓
GPU Physical Plan                  已选的 ownership / range / validity / stage
    ↓
Triton / cuTile / TileLang leaf    capability + 机械投影 + runtime 接线
    ↓
下层 compiler                       layout / register / instruction / tuning
```

本轮没有把 GPU Plan 冒充成所有机器共用的具体 Plan。未来 CPU/RVV 是新的 target family：复用 canonical Kernel IR 与共同 Plan vocabulary，但拥有自己的 realizer、物理决定和 target extension。

---

## 一、四轮结束时拿到了什么

### 1. 编程模型已经冻结，但冻结的是边界，不是“所有 target 都支持所有 op”

冻结后的核心是结构化 logical-region kernel：作者写 logical domain、region、dataflow、数值路径、状态、effects 与阶段依赖；编译器选择 physical owner、range、validity、storage 和 execution stage；surface 只检查能力并投影。

冻结不表示：

- TileLang 0.1.13 必须机械表达 Triton/cuTile 的每一种 closure；
- 下层首次编译超时也要由共享层绕开；
- GPU 规则可以直接搬给 RVV；
- 一个 logical callable 必须只有一次机器 launch。

冻结表示：遇到新算子时，不能再日常性修改上述归属；只有真实算法无法用现有 Core 表达、也不能交给下层已有能力时，才重新讨论语言规格。

### 2. 双机全量的最终规模

两份固定表现在各有 118 个 `kernel × case` 记录；每条记录有 Triton、cuTile、TileLang 三个 generated 单元格，因此两台机器合计 708 个 provider-case 单元格。

| 机器 | Triton：pass / unsupported / timeout / failed | cuTile：pass / unsupported / timeout / failed | TileLang：pass / unsupported / timeout / failed |
|---|---:|---:|---:|
| RTX 5090 D | 116 / 1 / 1 / 0 | 115 / 2 / 1 / 0 | 104 / 13 / 0 / 1 |
| H100 | 116 / 1 / 1 / 0 | 114 / 3 / 1 / 0 | 106 / 12 / 0 / 0 |

遗留数值问题收口后，合计是 671 个真实数值通过、32 个提前明确不支持、4 个下层首次编译超时、1 个真实失败。没有把 unsupported、compile timeout 或数值失败写成 pass。

这里没有违反本轮“不做全量”的测试范围：118 行主体仍来自第四轮双机全量，只把直接受修复影响、已在 H100 定向复验的 `max_pool2d × TileLang` 单元格从 failed 更新为本轮实测值；其余 707 个单元格没有重跑或改写。

完整数字在：

- [RTX 5090 D 全量表](baseline/kernel-performance.csv)
- [H100 全量表](baseline/kernel-performance-h100.csv)

### 3. 三个 provider 的赢家确实随机器和算子变化

按表中 p50 的精确值计算，单独赢家与并列项如下：

| 机器 | Triton 单独 | cuTile 单独 | TileLang 单独 | 并列 | 无可用结果 |
|---|---:|---:|---:|---:|---:|
| RTX 5090 D | 45 | 22 | 42 | 8 | 1 |
| H100 | 58 | 28 | 29 | 2 | 1 |

5090 的 8 个并列项包括 Triton/TileLang 2 个、cuTile/TileLang 2 个、cuTile/Triton 2 个、三者并列 2 个；H100 的两个并列项都是 cuTile/Triton。赢家分布在两台机器上明显改变，说明三个 surface 并非只是在重复同一份结果；同时它也说明机器相关的最终质量不能从单卡结论外推。

---

## 二、第一轮：把登记表里的 correctness 与唯一语义合同落实到代码

这一轮对应的核心提交是：

- `96d4cf1`：闭合 canonical/physical binding contract；
- `44ffdc7`：在 frontend 拒绝尚未兑现的 source 构造，并删除未定义的 `I.fence`；
- `c1c5962`：代码完成后同步相关 `doc/` 与审计报告。

### 2.1 Welford/state-stream：真正的问题是 `I.full` 丢了已选 padding

最小 repro 使用 `M=64, N=257`，以 `state_stream` 携带 `(count, mean, m2)`，chunk 内用内建 reduction，chunk 间用 Chan/Welford 标量公式更新。修复前它可以编译、运行，输出也有限，但：

- mean 最大误差约 `0.079155`；
- variance 最大误差约 `0.586592`。

这证明问题不是“Welford 写不出来”，而是非整除物理尾块的 count 被错误参与归约。

定位后的完整链条是：

1. 既有 Plan 构造已经知道 region-shaped `I.full((column_region,), 1)` 在 reduction tail 上需要 identity padding；
2. `intent.full` 的三个 target handler 却只发原始 fill，没有读取该 value 的 Plan padding binding；
3. 于是逻辑上不存在的物理 lane 也被填成 `1`，污染 count、mean 与 M2。

修复没有识别 Welford 名字，也没有给 state-stream 开分支。Triton/cuTile 的 `emitFull` 统一经过 `padExpression`，TileLang 的 `emitFull` 在有 binding 时逐元素消费同一份 planned padding。修复后同一个 repro 的 mean 误差降到约 `1.49e-8`，variance 误差降到 `2.38e-7`。

语料自查也解释了为什么此前全量一直没发现：现有 kernel 中 region-shaped `I.full` 主要作为 state-stream carry init/accumulator，或者作为显式 select 的填充值；此前没有真正制造“沿该 region 的物理尾块对 `full` 结果再归约”这个组合。一次性最小 kernel 命中了这个空白，验证后删除，没有进入 corpus。

### 2.2 SSA type、ABI metadata 与 result metadata 收敛成一份可验证合同

修复前，真实 MLIR SSA type 与 `intent.result_types`、`intent.result_shapes`、ABI/view metadata 可以分叉。分析有时读 metadata，leaf 又读真实 `ViewType`；一份错误的外部 IR 可能一路运行到后面才产生不一致行为。

现在 canonical verifier 集中检查：

- function parameter 的真实 type 与 ABI metadata；
- operation result 的真实 SSA type、rank、shape 与 result metadata；
- view access mode 与 element dtype；
- result node/name/shape 的数量和对应关系。

这不是再造一套 typed Python IR，而是验证 canonical MLIR 自身携带的冗余字段没有分叉。

### 2.3 region block argument 进入公共 KernelModel

Frontend 原本已给 region block argument 分配稳定 ID，verifier 也会检查；但公共 `KernelModel.values/valueIDs` 只收 ABI argument 和 op result，三个 emitter 因而各自解析 `region_argument_nodes` metadata。

本轮把 nested region 的 block argument 与 ABI/result value 一起纳入公共 value-ID index。后续 Plan 和 leaf 通过同一个 `getValueID()` 读取，不再各自解释 metadata。这项修改同时为 GPU 之外的 target consumer 留下了可复用的 canonical identity。

### 2.4 删除“同 extent 猜轴”的 provenance fallback

旧 `axisFromLabel` 在找不到精确来源时，会按 symbol/extent 选择第一个同大小 domain，甚至构造 implicit axis。它能让简单程序继续跑，却可能静默把一个逻辑轴替换成另一个同尺寸轴。

现在 logical-axis provenance 只允许来自：

- SSA use-def；
- ABI shape symbol；
- region block argument；
- index relation。

无法唯一定位时直接给出诊断。相同 extent 不再被当成相同 identity。

### 2.5 Plan 显式保存 leaf 真正需要的 binding

这一轮新增并闭合了两类关键绑定：

- region argument stable ID → selected axis/range purpose/level；
- state-stream node → selected stream axis/range/relation。

同时，range 同时携带 canonical logical extent 与 selected physical tile。Row-vector、普通 region 与 stream consumer 从 Plan 取“选了哪一份范围”；leaf 只把 ABI symbol、`next_power_of_2` 或 target dtype 拼成目标语法。

Ragged/stream 的算法 use-def 没有整份复制进 Plan。它在公共 KernelModel 中索引一次，Plan 只保存多个合法物理映射中选定的 binding。这保持了“Kernel IR 是算法真理，Plan 是已选决定”。

### 2.6 语言表面去掉假能力

这一轮没有因为深层尚未支持就静默改写 source：

- `I.fence` 没有 scope、ordering、participants 或 barrier 语义，public API 被删除；
- 非 unit-step `I.domain` 在 frontend 明确拒绝；
- runtime `state_stream extent` 在 frontend 明确拒绝，仍保留 fixed/`I.auto` extent 与 runtime logical stop；
- `I.partition(count=...)` 语义仍被认为合理，但当前 realizer 不支持，因此在构造 Kernel IR 前明确诊断。

`state_stream` 本身没有删除，`partition(count)` 也没有伪装成 `extent`。

### 2.7 文档同步发生在代码之后

`c1c5962` 修改 compiler architecture、backend lowering、Kernel IR、Physical Plan、domain/control 与 tensor-flow 等文档，使其描述上述实际合同。顺序是代码先成为真相，文档再跟随；没有为了符合旧文档反向凑实现。

---

## 三、第二轮：收紧假能力，并用一次性探针找结构依赖

这一轮对应：

- `d506a93`：TileLang 提前拒绝 lane-owned FP8 MMA；
- `2d091e8`：补齐 floating power 的三 target 投影；
- `adec5ad`：让 staged slice 的 def-use 穿过 state-stream structured boundary；
- `0433707`：不再把静态 contraction tile 钉死为完整 extent。

探针只用于主动撞角落，放在 `/tmp`，看完生成源码或完成一次数值运行后删除；没有建立 `test/`、pytest、fixture 或新的验证框架。

### 3.1 TileLang FP8 MQA：从“能力检查通过、下层 assertion”改成准确 unsupported

此前 `fp8_mqa_logits` 可以通过 Intent/TileLang capability check，最后在 CUTLASS FP8 MMA assertion 失败。根因不是 FP8 dtype 本身，而是该 contraction 的 matrix-M 轴是 runtime lane extent，TileLang 0.1.13 当前原语不能机械投影这一形态。

现在 TileLang leaf 读取共享 Plan 中的 `ContractionAxes`，在 source emission 前检查：

- operand 是否为 FP8；
- matrix-M 是否落在 runtime lane-owned result axis。

命中时给出明确 unsupported，不再把下层 assertion 当成“偶发失败”。这只是 capability 子集收紧，没有改 Kernel IR，也没有为 MQA 按名字分支。

### 3.2 稀有 canonical op 的拼写：真正发现的是 floating `power`

对现有 corpus 中出现 0–1 次的 canonical op 做最小源码探针时，`intent.binary power` 暴露为名义存在、三条 leaf 未完整兑现的路径。

修复后：

- common GPU legality 明确支持 f8e4m3fn、f8e5m2、f16、bf16、f32；
- Triton 投影为 `libdevice.pow`，非 f32 先扩到 f32 再按结果 dtype 转回；
- cuTile 投影为 `ct.pow`，FP8 情况同样显式扩宽；
- TileLang 投影为 `T.pow`。

这是一处逐 op 的 target spelling 闭环，没有新增调度机制。

### 3.3 staged contraction 与 ordered/state-stream 组合暴露了真正的 use-def 断点

普通 MLIR 的直接 users 无法表达 state-stream result 的完整语义来源：一个 stream result 同时来自 init operand 与 region terminator 的 yield operand。旧 stage slicing 只沿普通 defining op/users 回溯，在 staged contraction 与 stream 同时出现时会漏掉中间值和 terminal 可达性。

修复在公共 `KernelModel` 中建立：

- `structuredResultSources`：structured result 对应哪些 init/yield source；
- `structuredValueUsers`：这些 source 在 structured 语义上流向哪个 result。

stage dependency、slice collection 与 terminal reachability 都沿这份公共索引走。Leaf 没有增加“遇到 attention/MoE 就补一段”的判断。

同一探针还暴露 stage member tile 曾被字符串 `ragged_member` 代替。现在 `StageAxisOp(member)` 读取该 member axis 已选的 ownership range；feature/reduction/member 三个 tile 不必碰巧相同。

### 3.4 静态 contraction extent 不再覆盖可调 tile

极端 K/静态小域探针发现：只要 contraction domain 是静态的，旧逻辑就会把 reduction tile 直接设成完整静态 extent。它把“逻辑范围已知”错误等同于“物理 tile 已选”，使 tile 永远只有一个值。

本轮删除这条覆盖。只有 source/Plan 明确需要的 inner stream contraction extent 保持 fixed；普通静态 contraction 的 tile 继续是可选参数，交给下层 tuner。

### 3.5 探针轮的实际边界

这轮没有把“两条 ragged、两条 ordered、staged+ordered、三个不同 stage tile”变成新的 kernel category。现有逐轴角色、多 range、公共 structured use-def 与 per-stage axis binding 足以承载组合；真正需要修的是信息穿过 structured boundary 时被丢失，以及一个物理字段被钉死。

由于探针按要求没有保留 raw fixture，这一轮没有可长期引用的逐探针 timing 表。永久证据是上述共享代码改动，以及第三、四轮对相同消费者的双机数值复验；不能把“没有保存临时日志”包装成一套永久测试设施。

---

## 四、第三轮：补完 generic combine 与 stage execution，冻结模型

这一轮的核心提交是：

- `3f2cf14`：typed generic combiner、target 投影、stage contract 主体；
- `0464b84`：把 record reduction 的 padding binding 规范化到真正发射的 field；
- `17699ed`：补齐 stage grouping policy；
- `aac30c9`：同步正式规范并冻结编程模型。

### 4.1 generic reduce/scan combine 进入 canonical Kernel IR

Generic combine 没有做成 opaque Python callable，也没有只把允许字符串从五个扩成更多。Frontend 把 `@intent.fn` combiner lower 成带 `intent.role = "combiner"` 的 typed helper body。

Canonical operand ABI 是：

```text
source components
+ identity components
+ explicit scalar captures
```

其正式合同是：

- accumulator 可以是 scalar/tensor component，也可以是同 shape 的 record components；
- helper 参数是两组 accumulator component，再加显式 capture；
- helper result schema、component 数和 dtype 必须与 accumulator 完全一致；
- identity 按 component 显式给出；
- runtime capture 必须通过 `combine_operands` 成为 scalar SSA operand；
- tensor/record 不能作为隐式 capture；
- helper 只能包含 pure scalar canonical op，不允许 load/store/atomic/RNG/effects；
- 作者选择 reduce/scan 就表示接受合法 reassociation，compiler 不再反证数学结合律。

需要严格顺序时仍应使用 ordered/state-stream；generic combine 没有把 ordered 算法偷偷换成 reduction tree。

### 4.2 emitter 只把 closure 交给下层，不自建归约树

Triton：

- generic reduce → `tl.reduce(..., combine_fn=...)`；
- generic scan → `tl.associative_scan(..., combine_fn=...)`；
- helper 发射为 `@triton.jit` function。

cuTile：

- generic reduce → `ct.reduce(..., func=..., identity=...)`；
- generic scan → `ct.scan(..., func=..., identity=...)`；
- helper 发射为 cuTile function。

cuTile identity 必须为常量，所以 runtime capture 通过附加 `(capture, valid)` carrier 适配 tuple ABI：真实 lane 的 `valid=true`，identity lane 的 `valid=false`。它不改变作者 accumulator schema，也不参与数学决策。

TileLang 0.1.13 的当前 PrimFunc surface 只有 fixed `ReduceKind`。定向验证确认 `T.comm_reducer` 虽能构造 `tirx.Reduce`，当前 CUDA lowering 会明确报 `Do not have a default for tirx.Reduce`；generic reduce/scan 因而在 source emission 前明确 unsupported，没有发串行慢路径冒充支持，也不再保留“更低层或许能闭合”的未验证状态。

### 4.3 contract 没有因为 reduce 放开而制造假 semiring

`tl.dot`、`ct.mma` 与 TileLang GEMM 并不接受任意 multiply/combine closure。`I.contract` 因此继续只接受当前矩阵原语可承接的 multiply/add 与 dtype/accumulator 组合。其他 semiring 必须由作者显式写 pointwise + reduce。

这保持了两条边界：generic reduce 是下层已有 callable primitive 的薄投影；generic contract 若强行对称放开，就会失去高性能矩阵原语并制造假能力。

### 4.4 `I.arg_reduce.max` 收敛为 generic reduce 的语法糖

Frontend 现在生成 `(value, index)` 两 component reduction；typed helper 明确写出 maximum 与 lowest-index tie-break。Kernel IR 中 helper 是语义权威，`combine_builtin=argmax_lowest` 只允许 target 选择已验证数值等价的 native spelling：

- Triton 使用带 lowest-index tie 的 native max/index；
- cuTile 使用 `ct.max + ct.argmax`；
- TileLang 用 max、相等候选与 min index 组合。

替换独立路径前，cross entropy forward/backward 的 value、index、tie 与 gradient 已在三个 target 上对照通过。

### 4.5 generic record Welford 又发现一处 padding alias 丢失

Tuple/record Welford 对 `M=64,N=257` 的验证发现：reduction identity padding 被记在 `intent.extract` 别名上，但 leaf 真正发射的是底层 record field。Triton 还缺 domain-result value ID → selected physical tile 的索引。

`0464b84` 在共享 Plan 构造中把 binding 规范化到真正 materialized field，Triton只增加读取 Plan 后的 shape spelling。修复后，5090/H100 上 Triton/cuTile 的 mean 误差约 `3e-8`，variance 误差 `2.38e-7`。这仍是 value provenance/binding 修复，不是 Welford 特判。

### 4.6 Stage execution 从隐含 stream 顺序变成可验证 Plan 合同

`StageOp` 现在显式保存：

- stage node；
- dependencies；
- inputs / outputs；
- operation slice / terminals；
- synchronization；
- fusion policy；
- grouping policy。

`StageBufferOp` 保存：

- canonical value；
- 唯一 producer stage；
- consumer stages；
- owner roles；
- access；
- lifetime；
- visibility。

GPU realizer 沿 canonical def-use 建立这些字段。Plan verifier 检查拓扑顺序、dependency 与 input producer 的精确一致、single writer、consumer 后继关系、owner role 存在性、final/intermediate stage 形态。公共 emission preflight 再检查当前三个 surface 能兑现的合同。

当前 GPU 具体选择是：

```text
synchronization = same_stream
fusion           = forbidden
grouping         = fixed_operation_slice
buffer access    = single_writer_read_only_consumers
lifetime         = producer_to_last_consumer
visibility       = same_stream
```

三个 surface 可以生成多个 compiler-private kernels，但调用方仍看见一个 logical callable。多 stage 不再只依赖“碰巧在同一 CUDA stream 顺序 launch”；Plan、verifier、preflight 和 runtime 投影共同声明并兑现该顺序。

Plan schema 可以容纳更宽的 fusion policy，但当前 GPU realizer固定选 `forbidden`，三个 surface 也只接受现行合同。这是 schema 能力与当前 target capability 的差异，不是现行路径内的矛盾。

### 4.7 冻结前的定向验证

这一轮没有提前跑全量，只运行直接消费新合同的 repro/probe：

| 能力 | 5090 | H100 |
|---|---|---|
| generic record reduce/scan + runtime scalar capture，Triton/cuTile，`N=4093` | 通过；sum 最大误差约 `2.29e-5`，max 误差 0 | 通过，误差一致 |
| generic Welford record reduce，`M=64,N=257` | 通过；mean 约 `3e-8`，variance `2.38e-7` | 通过，误差一致 |
| fixed-add 多 component 长轴 scan，三个 surface | 通过；两 component 约 `7.63e-6`、`1.53e-5` | 通过，误差一致 |
| TileLang generic combine | emission 前 unsupported | emission 前 unsupported |
| `I.arg_reduce.max` cross entropy forward/backward | 三 target 通过 | 三 target 通过 |
| 两 stage ragged MoE execution contract | 三 target 通过 | 三 target 通过 |

H100 TileLang 初次验证曾误用系统 CUDA 11.5 `nvcc`，它不识别 `sm_90a`；切换到机器已有 CUDA 12.2 后同一生成源码通过。这个问题留在运行环境，不进入 compiler/Plan 分支。

### 4.8 文档冻结

`aac30c9` 在能力完成后同步：

- compiler architecture；
- Kernel IR；
- Physical Plan；
- backend lowering；
- compiled artifact；
- DSL model、control、tensor-flow；
- reduction/GEMM kernel spec；
- 编程模型审计报告。

冻结后的不变量是“一个 source kernel 对应一个 logical callable”，而不是“只能有一个机器 launch”；generic combine、stage execution 与 target-family 分叉都成为正式语义，不再以“当前不支持”留在文档里。

---

## 五、第四轮：冻结后的五个真实小算子与双机全量

这一轮没有重新设计语言，而是先选真实算子，再看冻结后的模型是否自然承接。新增：

| 算子 | 为什么选 | 主要压到的结构 |
|---|---|---|
| `max_pool2d` | 常见视觉算子，不是 contraction/attention | 两个输出 partition、负 affine 边界、两个小域 reduction、`-inf` identity |
| `softmax_backward` | 基本反向算子 | `N=4097` 非整除 reduction tail、f32 accumulator、广播回写 |
| `csr_spmm` | 常见稀疏算子 | runtime-bounded ordered loop、数据读取索引、间接 gather、向量 accumulator |
| `batch_norm_training` | 训练基本盘 | 三输出、mixed f16/f32、两轴分层 reduction、不同秩广播、runtime scalar epsilon |
| `triangular_solve` | 小型因子分解/递推 | in-place `InOut`、动态内层上界、先前写值再读、顺序标量状态 |

它们分别位于：

- [max pool](../examples/kernels/vision/max_pool.py)
- [softmax backward](../examples/kernels/backward/softmax.py)
- [CSR SpMM](../examples/kernels/sparse/csr_spmm.py)
- [batch norm training](../examples/kernels/normalization/batch_norm.py)
- [triangular solve](../examples/kernels/factorization/triangular_solve.py)

统一 runtime/reference 接线在 [small_operators.py](../examples/repro/common/small_operators.py)。这不是单元测试目录；仍通过已有 `examples/run/repro.sh` 走完整 DSL → MLIR → Plan → target source → JIT → GPU → reference 数值对照。

### 5.1 五个算子的最终结果

数字为 generated p50，单位 ms；`FAIL` 不填伪数字。

| kernel | 5090 Triton | 5090 cuTile | 5090 TileLang | H100 Triton | H100 cuTile | H100 TileLang |
|---|---:|---:|---:|---:|---:|---:|
| max_pool2d | 0.0180 | 0.1060 | 0.0218 | 0.0225 | 0.0497 | 0.0279 |
| softmax_backward | 0.1334 | 0.1457 | 0.1326 | 0.1041 | 0.0797 | 0.1430 |
| csr_spmm | 0.0460 | 0.0500 | 0.0220 | 0.0423 | 0.0902 | 0.0211 |
| batch_norm_training | 0.1102 | 0.0747 | 0.0774 | 0.3036 | 0.1645 | 0.1431 |
| triangular_solve | 0.0312 | 0.0255 | 0.0225 | 0.0411 | 0.0270 | 0.0348 |

新增 30 个单元格现已全部数值通过。H100 TileLang `max_pool2d` 在第四轮全量时曾如实保留为 failed；后续定向定位与修复见 5.6，没有把旧失败直接改名为 unsupported。

### 5.2 新算子暴露并修掉的共享问题

#### a. 浮点常量必须保持浮点 spelling

某些下层会把没有小数点的 `0` 当整数，影响 mixed-type arithmetic 与 combiner helper。现在三 leaf 与公共 combiner 统一使用 typed finite-float spelling；例如浮点零发成 `0.0`，而不是依赖目标语言重新猜类型。

#### b. ordered/普通 `for` block argument 不应被强行绑定成 physical region owner

Region binding 的 Plan 构造原本只认识 parallel/state-stream。Triangular solve 的普通顺序 loop block argument属于算法控制，不需要 ownership/traversal range。现在 Plan 只给真正需要 physical range 的 parallel/state-stream argument 建绑定，不为 sequential loop 伪造决定。

#### c. logical-axis provenance 必须穿过 reshape 与 scan/state carry

Batch norm、pooling 与递推组合暴露：逐元素/reshape 结果的 axis 来源和 scan/state carried value 曾在某些路径被覆盖或遗漏。`93df35b` 将 provenance 合并规则放回 shared facts，三个 leaf 不再从 result shape 各自反猜。

#### d. validity proof 不能用循环论证把 tail 消掉

Max pool 的 3×3 filter domain 被物理取整时，直接 load 的 padding 与后续 reduction identity 都参与 correctness。旧中和证明会把“load 自己已经带 padding”当成 downstream consumer neutralization，从而消掉本来必须存在的 validity。

修复后：

- neutralization proof 可以穿过 `intent.reduce` 和 `intent.yield`；
- 但 load 自己的 padding 不再被当成证明“这个 load 不需要 padding”的依据；
- validity binding 落在真正发射的 SSA value/field；
- cuTile 只在 tensor-indexed relation 需要时额外物化 validity，纯 shape tail 交给 `ct.scatter(check_bounds=True)`；
- 不重复发射下层已经能处理的 shape bounds。

#### e. TileLang store 分成能力驱动的 bulk path 与 checked path

TileLang 原路径要么过度逐元素化，明显伤害 transpose 等既有 kernel；要么对 tail/indirect store 过宽，可能写错。

现在 leaf 只按 target 能力选择 spelling：

- compile-time 能证明 planned tile 不超过 view extent、且不存在 expanded/tensor-indexed relation 时，用 native bulk `T.copy`；
- 其余形态使用逐元素 checked store；
- 没有按 `max_pool2d` 或 transpose 名字分支；
- 没有在 program 内再加动态“猜是否完整块”的分支。

这恢复了现有 transpose：5090 TileLang p50 为 0.0915 ms，H100 为 0.0810 ms，同时保留 tail validity。后续逐候选定位证明 H100 max pool 的错误不在这条 store 分流，而在下层对嵌套 reduction fragment 的一个非对称候选布局；见 5.6。

### 5.3 既有关键 repro 的恢复值

共享 provenance/validity/store 修改影响的不只五个新算子，因此定向复验还确认：

- cuTile continuous GQA decode：5090 1.2708 ms，H100 1.8275 ms；
- cuTile causal Conv1D backward：5090 0.4729 ms，H100 0.6305 ms；
- TileLang matrix transpose：5090 0.0915 ms，H100 0.0810 ms。

这些值来自最终全量表，不是把旧数字复制到新行。

### 5.4 两个表怎样更新

本轮把原有 113 行按原顺序保留，在末尾追加五个新算子，得到 118 行。Generated status/p50/p95 用本轮双机实跑刷新。

上游 source baseline 没有重测：对旧 113 行逐列比较，六组 `*_source_p50_ms/*_source_p95_ms` 完全保持；五个新算子没有可比上游实现，因此 source 列为空。没有为了让新行好看拼 PyTorch reference 或 adapter 数字。

### 5.5 性能回退核查

全量中最显眼的两个旧表差异是：

- 5090 cuTile `block_scaled_matmul`：旧表约 0.0703 ms，当前约 0.1750 ms；
- 5090 Triton `mamba_chunk_scan`：旧表约 0.022 ms，当前约 0.0302 ms。

在对应旧 commit 上用当前机器同日重跑，分别也得到约 0.175 ms 与 0.0302 ms；因此无法把这两项归因于四轮代码改动。对 block-scaled matmul 又重复测得 `0.1761/0.1792 ms` 与 `0.1756/0.1786 ms`，旧 commit 在当前环境也稳定落在同一区间；kernel DSL、runtime 计时范围、shape、候选集以及生成的 `ct.mma` 结构在两个提交间没有改变。历史 `0.0703 ms` 因而是当前环境无法复现的旧测量，而不是当前代码回退。旧测量没有保存当时的 driver/cuTile JIT、时钟/功耗与编译缓存状态，所以不能诚实地把 2.49 倍差异进一步指定给其中某一个因素；“环境”在这里表示已经用旧代码同机 A/B 排除了 compiler diff，但历史环境快照不足以继续分解，而不是泛泛猜测。

这不等于宣称所有微小波动为零，只说明本轮没有找到可由当前代码 diff 解释的既有大幅性能退化。

5090 上 cuTile 的单独赢家从旧表 27 变成 22，也不是“新增五行把比例稀释”——这里计的是绝对个数。逐行比较得到：7 个旧 cuTile 单独赢家不再单独获胜；`grouped_query_head_add` 从并列变成 cuTile 单独赢家，新增 `batch_norm_training` 也由 cuTile 获胜，净变化正好是 `-7 + 2 = -5`。七个转出项里，Conv2D、selective scan 与 conv2d variant 的 cuTile 自身没有退化，分别是 Triton/TileLang 变快；record、FP8 e4m3 与 reshape-cache variant 是数微秒级换位；只有 varlen attention noncausal 的 cuTile 从 `0.3798` 到 `0.4145 ms`，同时 Triton略快，构成一个约 9% 的真实表内失位。结论是赢家下降主要来自其他 provider 改善和短核近似并列，不是一处让 cuTile 普遍退化的共享改动。

### 5.6 四轮后的遗留收口

H100 TileLang max pool 的逐候选 A/B 给出了明确根因。六个候选中，`8×8`、`16×16`、`32×32`、`128×64` 与 `128×128` 都数值正确，只有 `64×128` 在 H100 上产生 `inf` 和大量有限错误；同一个 `64×128` 候选在 5090 上正确。生成源码中的 load validity、两次 `T.reduce_max` 与 checked store 均相同，错误只随 TileLang 对非对称 fragment layout 的 H100 lower 改变；而其 autotuner 使用 `skip_check=True`，只按延迟把这个错误候选选成 winner。

暴露面不是 max pool 名字，而是“rank 至少为 4 的 tensor 连续经过两次降 rank reduction，且两个 program axes 独立可调”。TileLang target indexing 现在为这类结构登记 program tile 的相等约束，runtime tuner 从六个候选缩到四个对称候选；Triton/cuTile 与其他 TileLang kernel 的候选不受影响，也没有按 H100/5090 型号分支。修复后 5090 为 `0.0217/0.0229 ms`，H100 为 `0.0279/0.0283 ms`，两边数值误差均为 0。

另外三处冻结边界同时收口：

- `I.end` 正式定义为 rank-one 半开 domain/region 的 exclusive endpoint；stream stop 取与 streamed axis 的逻辑交集，空交集不执行 step、carry 保持 initial state，stop 不能被解释成物理 tile end 或数据依赖 early-exit；
- `I.assume_in_bounds` 是支配后续同一 SSA index/view/axis 访问的 unsafe 调用前置条件，违反即未定义行为，不执行 clamp/runtime check；
- 现有 2:4 sparse contraction 的 format、compressed/metadata/RHS axis 与 dtype schema 由公共 verifier 核对。它仍明确登记为 format-specific convenience/过渡入口；未来收敛到 `sparse_contract + format descriptor`，但在只有一种格式时不制造下游仍硬编码 2:4 的假通用 API。

---

## 六、四轮之后，各层到底持有什么

### 6.1 DSL / 编程模型

持有作者才知道的算法：

- logical callable ABI、alias、effects；
- domain、region、partition 是否存在；
- logical index relation 与 precondition；
- parallel、ordered、state carry；
- reduce/scan/contract/gather/scatter/buffer/atomic；
- dtype、identity、数值路径；
- 作者显式的阶段和调用编排。

不持有 tile、program id、warp、register/shared placement、GPU layout 或 RVV `vl/LMUL`。

### 6.2 Canonical Kernel IR

是唯一算法真理，额外保存稳定 op/value/block-argument identity，以及 typed pure combiner body。它不复制 Physical Plan，不因 target 不同改写作者算法。

### 6.3 Shared facts

保存可从 Kernel IR 重算但昂贵的事实：axis provenance、structured use-def、ragged/state-stream relation、contraction/scan、validity 与 access footprint。它不保存“多个合法答案中选了哪个”。

### 6.4 Target-family Physical Plan

保存无法从算法唯一推出的选择和 leaf 必须直接消费的 binding：

- axis roles；
- ownership/traversal/reduction/lane/access ranges；
- region/stream argument 到 selected range；
- program mapping/persistence；
- padding/validity/physical fill；
- storage/workspace；
- contraction/stage axis tile；
- stage dependency/buffer/lifetime/visibility/grouping；
- 交给下层 tuner 的参数名。

GPU Plan 中的 program grid/worker 不是未来 RVV Plan 的强制字段。

### 6.5 Surface leaf

只做：

- capability check；
- canonical op + Plan binding 到目标原语/语法；
- ABI/JIT/workspace/runtime 接线；
- source-located unsupported。

本四轮修掉的共同模式，正是 leaf 从 shape、role 名或邻近 op 重新拼事实；这类逻辑要么回到 shared semantic index，要么成为显式 Plan binding。

### 6.6 下层 compiler

继续负责 layout、thread/register mapping、instruction selection、pipeline 与 candidate winner。Generic combine 的 reduction tree、普通 GEMM 的 MMA 选择、能由 `check_bounds` 处理的纯 shape tail，都没有被抢回 Intent。

---

## 七、当前明确边界

以下不是“已经完成”的假象：

1. **CPU/RISC-V/RVV 尚未接入。** 当前冻结的是它们应消费的 Kernel IR 边界，不是后端实现完成。
2. **TileLang generic reduce/scan closure 明确不支持。** 0.1.13 的 `comm_reducer` 会生成当前 CUDA codegen 不处理的 `tirx.Reduce`；固定 combiner仍可使用。
3. **TileLang lane-owned FP8 MQA 明确不支持。** 已在 emission 前拒绝，不再进入 CUTLASS assertion。
4. **5090 TileLang `grouped_query_head_add` 仍为下层编译失败。** 同一 Plan 在另两个 surface 通过，当前失败保留在 target/toolchain 边界。
5. **token-sparse MLA 首次编译成本仍过高。** Triton/cuTile 为 compile timeout，TileLang明确 unsupported；没有为编译时间在共享层发明结构。
6. **部分 target capability 本来就是子集。** 例如 TileLang CAS、二维联合 access footprint、某些 split-K/staged/private-storage 组合；完整状态以两份 CSV 为准。

这些边界没有推动新增 kernel-name 分支，也没有让一个 surface 变成第二套 realizer。

---

## 八、提交链

| 轮次 | 提交 | 作用 |
|---|---|---|
| 登记表闭合 | `96d4cf1` | canonical type/metadata、block argument ID、exact provenance、region/stream binding、`full` padding consumption |
| 登记表闭合 | `44ffdc7` | 删除 `I.fence`，提前拒绝 unrealized source constructs |
| 登记表闭合 | `c1c5962` | 按实际代码同步文档 |
| 探针自查 | `d506a93` | 收紧 TileLang FP8 MQA capability |
| 探针自查 | `2d091e8` | 补齐 floating power 的逐 target 投影 |
| 探针自查 | `adec5ad` | structured state-stream use-def 与 staged binding |
| 探针自查 | `0433707` | 保持静态 contraction tile 可选择 |
| 能力闭合 | `3f2cf14` | typed generic combine 与 stage contract 主体 |
| 能力闭合 | `0464b84` | record field padding binding |
| 能力闭合 | `17699ed` | execution stage grouping policy |
| 规范冻结 | `aac30c9` | 同步正式文档并冻结编程模型 |
| 小算子闭合 | `8fd36cf` | 常量、load validity、顺序 region 等首批投影缺口 |
| 小算子语料 | `021beb1` | 五个真实小算子与统一 repro 接线 |
| 全量修复 | `93df35b`–`4208b48` | axis/scan provenance、validity proof、TileLang store capability 投影 |
| 双机矩阵 | `b82bdb1` | 刷新两份 118 行全量表，固定 source baseline |
| 遗留收口 | `c3546a1`、`9e21f00` | TileLang nested-reduction candidate legality，不按算子或架构分支 |
| 语义收口 | `b729a63` | sparse fixed schema、`end`/precondition 正式语义与 TileLang combine 确定边界 |

---

## 最终判断

这四轮没有通过继续堆 kernel matcher 来“完成编译器”。真正完成的是三件承重工作：

1. 已经存在的算法事实不再在 frontend、metadata、facts、Plan 与 leaf 之间静默丢失或按 shape 猜回；
2. generic combine 与 multi-stage execution 这两项真实表示能力进入 canonical IR/Plan，而不是留成 target 私有技巧；
3. 冻结后的模型经五个事先未为编译器设计的小算子和两台差异明显的 GPU 做了 708 个 provider-case 单元格的叠加核验。

因此当前可以诚实地说：GPU 主线已经形成完整的算子级 DSL → canonical Kernel MLIR → target-family Physical Plan → 多 surface emission 闭环。它仍有明确 target 能力边界，也还没有 CPU/RVV 实现；但这些边界已经能被定位到具体层，而不再要求修改编程模型、给算子开特例，或让 leaf 重做一遍编译器。
