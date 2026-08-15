# 两轮架构收敛与冻结准备报告

## 1. 报告范围

这份报告覆盖最近两轮实际推进，不把更早的四轮能力建设重新记一遍：

1. 收掉四轮全量之后尚未闭合的数值、能力与语义问题；
2. 对新增的 generic combine 与多阶段执行合同做反向自查，删除第二份真理、算法改写入口和冗余路径。

对应提交为：

| 轮次 | 提交 | 作用 |
|---|---|---|
| 第一轮 | `c3546a1` | 首次约束 TileLang 嵌套降秩归约的非法候选 |
| 第一轮 | `9e21f00` | 把约束从 whole-kernel 扫描收敛到具体 reduction binding |
| 第一轮 | `b729a63` | 补齐 sparse 2:4 canonical schema 校验并钉死 TileLang generic combine 能力边界 |
| 第一轮 | `6ed16e7` | 更新定向结果、双机状态与归因 |
| 第二轮 | `dbeaac6` | 压缩 Physical Plan、删除派生字段和失效发射路径，完成冻结前边界自查 |

本报告是状态与证据记录，不是规格。稳定设计只在 `doc/`；运行结果、当前 target 能力、未实现项与历史取舍只在 `report/`。

## 2. 第一轮：收掉四轮推进后的剩余问题

### 2.1 H100 TileLang `max_pool2d` 的静默数值错误

#### 现象

同一份 Intent Kernel IR 和 Physical Plan 在 5090 的三个 surface、H100 的 Triton/cuTile 上都正确，只有 H100 TileLang 能编译、能运行，但数值错误。它不是语言表达失败，也不是普通 compile failure，而是最危险的“运行成功但结果错误”。

逐候选 A/B 后，原始六个 TileLang profiles 中只有 `program_m=64, program_n=128` 在 H100 产生 `inf` 和大量有限错误；`8×8`、`16×16`、`32×32`、`128×64`、`128×128` 均正确。同一个 `64×128` 候选在 5090 正确。

生成源码的 load validity、两次 reduction 与 checked store 没有变化，错误只跟随 H100 上 TileLang 对非对称 nested-reduction fragment layout 的 lowering。TileLang tuner 使用 `skip_check=True`，只按延迟选中了这个错误候选，因此不能把责任推给运行时输入。

#### 第一次修法与随后的自我纠正

`c3546a1` 做了两件事：

- runtime tuner 的 `autotune_configurations(...)` 增加 `equal_role_groups`，让 target projection 可以声明两个参数角色必须相等；
- TileLang emitter 在嵌套降秩 reduction 存在时，要求 `program_m == program_n`。

这个方向对，但最初触发条件仍是 whole-kernel 的 `hasNestedTensorReduction(...)`。同一轮的 `9e21f00` 随即把它收敛为具体 reduction binding 上的 `isRankReducingReductionChain(...)`，并在 TileLang `RealizationIndex` 中登记 `requiresSymmetricProgramTiles`。因此最终形态不是：

- 按 `max_pool2d` 名字分支；
- 按 H100/5090 架构型号分支；
- 在共享 realizer 中替 TileLang 发明一套调度。

它是 TileLang surface 对一个可复现 lowering 限制的参数合法性约束：只读取 canonical reduction 结构和已经存在的 `program_m/program_n` 参数角色，然后过滤该 surface 无法正确兑现的候选。Triton、cuTile 和无该 reduction chain 的 TileLang kernel 不受影响。

#### 修复结果

| 机器 | 数值 | p50 / p95 |
|---|---|---:|
| RTX 5090 D | 最大误差 0 | `0.0217 / 0.0229 ms` |
| H100 | 最大误差 0 | `0.0279 / 0.0283 ms` |

H100 TileLang `max_pool2d` 从真实数值 failed 收回为 pass。两机矩阵随之成为 671 个数值通过、32 个提前 unsupported、4 个下层 compile timeout、1 个 CSV failed 状态。这里只更新了受修复影响的单元格，没有把其余 707 格重跑或抄写成新结果。逐候选临时 probe 在验证后按项目纪律删除，仓库保留的是最终 CSV、对应代码 diff 与 [编程模型架构审计](programming-model-and-compiler-architecture.md) 中的结果记录，而不是完整 probe transcript。

### 2.2 Sparse contraction 的 attrs 不再只是装饰

Canonical op 已经叫 `intent.sparse_contract`，但真正闭合的只有 2:4：此前 `intent.format`、压缩轴、metadata 轴和 RHS reduction 轴虽然进入 IR，公共事实分析没有把它们全部作为语义合同验证，错误组合可能继续流入 leaf。

`b729a63` 在共享 `KernelFacts` handler 中集中验证：

- `format == "two_of_four"`；
- compressed axis、metadata axis、RHS reduction axis；
- compressed、metadata、RHS 与 result 的 rank；
- compressed/RHS element dtype 一致；
- metadata 是 `i16`；
- compressed/metadata 的 axis provenance 一致。

这没有增加一个按 kernel 名称的入口，也没有伪造尚未存在的任意稀疏格式 API。当前实现事实仍是：2:4 是 format-specific convenience；稳定设计要求 canonical sparse contraction 显式保存 format identity、压缩轴和 metadata schema。只有真正出现第二种格式后，才有足够证据决定通用 descriptor 的公共形状。

### 2.3 `I.end` 与 `I.assume_in_bounds` 的合同被钉死

这两项没有再发明新的 canonical op；实现路径已经存在，缺的是不会随解释变化的精确定义。

`I.end(domain_or_region)` 被确定为：

- 只接受 rank-one 半开区间；
- 返回自身逻辑坐标中的 exclusive endpoint；
- 空 region 的 endpoint 等于 begin；
- 作为 `state_stream.stop` 时，与 streamed axis 取逻辑交集；
- 空交集不执行 step，carry 保持 initial state；
- 超过 axis end 不扩展原 workset；
- 不表示 physical tile end，也不表示依赖 carry 收敛的 early exit。

`I.assume_in_bounds(index, view, axis)` 被确定为 unsafe 调用前置条件：

- 只支配后续同一 SSA index、同一 view/logical buffer 和同一规范化 axis 的访问；
- 不 clamp，不生成 runtime check，也不是性能 hint；
- 违反条件属于调用方错误，语义未定义；
- compiler 可以把它用于 legality proof 或 mask 消除，但不能在作者未声明时猜测数据范围。

它们属于 Core 算法语义，因此保留在设计文档；“哪条实现当前是否通过”只记录在本报告。

### 2.4 TileLang generic reduce/scan 不再停留在“也许能做”

当轮定向 probe 记录表明：TileLang 的 `T.comm_reducer` 能构造 `tirx.Reduce`，但 CUDA lowering 明确报 `Do not have a default for tirx.Reduce`；generic scan 也没有等价的机械入口。

因此 `b729a63` 将能力边界改成 emission 前的明确诊断：

- fixed reduction/scan 继续按 target primitive 投影；
- generic typed closure 在 TileLang 上明确 unsupported；
- 不生成串行慢路径冒充支持；
- 不把 TileLang 的限制抬进共享 Kernel IR 或 Physical Plan。

Triton/cuTile 仍机械委托给下层原生 generic combiner。这个结论说明 generic combine 的共享设计成立，TileLang 只是能力子集。

### 2.5 两个性能疑点的归因

#### cuTile 单独赢家从 27 降到 22

逐行比较不是“新增行稀释比例”，而是绝对数量变化：7 个旧 cuTile 单独赢家转出，`grouped_query_head_add` 和新增 `batch_norm_training` 增加 2 个，净变化 `-5`。

七个转出项中：

- Conv2D、selective scan、conv2d variant 是其他 provider 变快，cuTile 自身没有退化；
- record、FP8 e4m3、reshape-cache variant 是数微秒级换位；
- varlen noncausal attention 的 cuTile 从 `0.3798` 到 `0.4145 ms`，同时 Triton 略快，是约 9% 的真实表内失位。

因此没有证据指向一处让 cuTile 全面退化的共享改动。

#### `block_scaled_matmul` 历史值无法复现

旧表为约 `0.0703 ms`，当前约 `0.175–0.179 ms`。历史 A/B 记录显示：在对应旧 commit 上用同一台机器同日重跑，同样得到约 `0.175–0.179 ms`；kernel DSL、shape、计时范围、候选集和生成的 `ct.mma` 结构在两个提交间没有变化。临时运行日志没有保留在仓库；本报告只保留当轮 A/B 结论，不能把它当成一份可再次执行的固定 artifact。

能确定的是 compiler diff 不是原因；不能确定的是历史 driver、cuTile JIT、时钟/功耗或 cache 中哪一个造成了旧值，因为当时没有保存这些环境快照。本轮没有把“环境”进一步伪装成某个未经证明的具体原因。

## 3. 第二轮：冻结前反向自查

第二轮不再扩算子或语言能力，而是逐字段问：这个信息是算法真理、可重算事实、已选物理决定，还是 target spelling？凡是可从上一层唯一推出的，都不能在下一层保存第二份。

### 3.1 Stage Plan 从“缓存全部事实”压成“只保存选择”

`IntentPlan_StageOp` 最终只保存：

- stage node identity；
- operation slice；
- synchronization。

被删除的内容包括独立 `StageBufferOp`，以及 stage dependency、inputs/outputs、terminal、intermediate lifetime/visibility 等可派生字段。公共 `indexStageOperations(...)` 现在从 `Kernel IR + operation slice + synchronization` 重算并验证：

- cross-stage value 必须解析到唯一、拓扑更早的 producer；
- intermediate 必须是 tensor value；
- effectful terminal 不能被多个 stages 复制；
- stage 必须是 intermediate-producing 或 effectful-final，二者不能混成模糊状态；
- dependency 必须指向 predecessor。

Operation slice 被保留，因为 pure recomputation 与 intermediate materialization 是多个合法物理方案中的真实选择；依赖和 lifetime 被删除，因为它们由这个选择与 Kernel IR 唯一推出。

### 3.2 Stage-axis 不再复制 logical extent

`IntentPlan_StageAxisOp` 只保存两种稳定绑定之一：

- canonical domain axis；或
- `source_value + tensor_axis`。

再加上已选 tile 和可选 worker axis。Logical extent 不再作为第二份字符串字段保存。公共 `getLogicalShape(...)` 从 ABI metadata 或 canonical result metadata 统一取得 source shape；SurfacePlan 再解析目标 tensor dimension。Leaf 不从 tensor 名字、shape label 或附近 operation 反猜 stage axis。

### 3.3 其他派生或失效字段被删除

| 删除项 | 为什么不该留 |
|---|---|
| Range 的 lower/upper offset 缓存 | 只有 producer 写入，没有 consumer；地址关系已有 canonical index/validity 来源 |
| Scan Plan 中重复的 combine/inclusive semantics | 算法语义已在 canonical scan op；Plan 只需要 physical axis、owner、residency 与 materialization |
| StreamAxis 固定 `inner_reduction` role | 值恒定且不形成选择，只是重复标签 |
| Pointwise `reuse_operand=-1` | 所有 producer 都写同一个无效值，TileLang 对应 in-place reuse 分支不可达 |
| 三个 leaf 的重复 pointwise/reuse 检查 | 删除字段后没有合法调用路径，保留只会暗示仍有第二套决定 |

这轮净变化为 `+376/-514`。代码减少不是目的；关键是删掉的信息都有更高层的唯一权威来源，剩下的 Plan 字段确实代表选择。

### 3.4 Generic closure 没有给 compiler 算法改写权

审计结果：

- typed helper body、component identity、purity 和显式 capture 只在 Kernel IR 中保存；
- analysis/realizer 不分析 combiner 的数学意义，不重排、不替换、不特化 closure body；
- Triton/cuTile emitter 只是把同一 typed body 机械翻译成下层 helper/lambda；
- `ordered`、`state_stream` 与 generic `reduce/scan` 保持不同 canonical structures；
- `contract` 没有因为 reduce 放开而获得 arbitrary semiring 通道。

因此 generic combine 是作者算法的一部分，不是 compiler 用来发明另一种归约算法的入口。

### 3.5 多阶段合同没有变成图编译器入口

曾经存在的 fusion permission/grouping 冗余字段被删除。最终边界是：

- compiler-private stages 只能位于同一个 logical callable 内；
- operation slice 是 target-family realizer 选出的物理 grouping；
- leaf 不能重新分组或合并 stages；
- dependency、lifetime、visibility 从 Kernel IR 与 slice 派生；
- wrapper-visible 多 kernel orchestration、ABI、调用次数和跨 callable fusion 不由 Intent compiler 决定。

所以多 stage 只是在一个算子 callable 内兑现作者已有的数据依赖，不是图级 fusion/fission 的预留入口。

### 3.6 没有新增按算子分类的入口

共享 realizer 仍然按 axis/range/operation binding 组合，不按 softmax、MoE、attention、pooling 名字分类。第二轮没有新增 kernel-name matcher。

第一轮的 TileLang nested-reduction 约束也经过了同样审计：它位于 TileLang capability projection，绑定具体 reduction chain 和 target parameter roles；它没有改变 Kernel IR、共享 Physical Plan 或其他 surface 的决定。这里是必要的 target legality，不是第二套算子编译器。

## 4. 两轮使用的验证证据

### 4.1 第一轮定向验证

- H100/5090 TileLang `max_pool2d`：逐候选定位后双机数值为 0 误差，延迟见 2.1；
- TileLang `comm_reducer`：确认生成 `tirx.Reduce`，CUDA lowering 无对应实现，随后改为 emission 前 unsupported；
- sparse 2:4：公共 schema/provenance verifier 覆盖 format、axis、rank 与 dtype；
- 性能归因：旧 commit 与当前 commit 在同机同日 A/B，排除 `block_scaled_matmul` 的 compiler diff。

### 4.2 第二轮受影响范围验证

第二轮没有跑全量矩阵，只运行直接消费新合同的 repro：

```text
cmake --build /tmp/intentdsl-build --target intent-compile -j 8
examples/run/repro.sh triton moe
examples/run/repro.sh cutile moe
examples/run/repro.sh tilelang moe
examples/run/repro.sh triton unique_consecutive
examples/run/repro.sh cutile unique_consecutive
examples/run/repro.sh tilelang unique_consecutive
```

构建通过；MoE 三 surface 数值通过，作为多阶段 operation slice、stage-axis 与 workspace 的直接消费者 repro；`unique_consecutive` 三 surface 数值通过，作为 scan semantics 从 Kernel IR 读取后的直接消费者 repro。这些结果证明相应真实路径可运行，不把每个内部字段误写成独立测试覆盖。

另外，当轮记录显示 5090/H100 上的一次性 `/tmp` probe 覆盖了 generic record reduce/scan、Welford `M=64,N=257`、多组件长轴 scan、arg-reduce sugar、两 stage ragged MoE 和 TileLang nested-reduction candidate legality。Probe 验证后删除，没有进入语料、fixture 或测试目录；因此仓库只保留汇总结果，不保留可独立复核的完整命令 transcript。

这份证据足以说明直接消费者已闭合，但它不是第二轮之后的双机全量回归；两份 baseline CSV 没有因为定向验证而被重写。

## 5. 本次文档与报告边界纠正

上一轮把“稳定设计”和“当前实现状态”混进了 `doc/`。本次按用户要求统一纠正：

- `doc/` 只保留编程模型、Core 语义、唯一权威来源、Physical Plan/leaf 边界和设计变更门槛；
- 移除 target 版本号、具体 unsupported/timeout/failed 列表、benchmark 数字、当前 package/realizer 状态、历史性能取舍和路线图；
- 撤掉“已经正式冻结”的状态宣告，改为设计变更必须满足的门槛；
- `partition(count)`、generic combine、sparse contraction、fence、workspace placement 等只在 `doc/` 说明稳定语义，不说明当前实现是否通过；
- 当前能力、失败、未实现项、验证范围和历史取舍只留在 `report/`；
- baseline CSV 没有改动。

报告目录收敛为：

```text
report/
├── baseline/   固定性能表与 baseline 说明
└── freeze/     编程模型审计、冻结准备与状态证据
```

原先平铺、内容已经被这两份冻结报告覆盖的旧报告随本次提交删除，不再让同一状态分散在多个顶层文件中。

## 6. 真正冻结前的准确状态

### 已经收敛的架构边界

- 一份 target-independent Kernel IR 保存作者算法；
- 可重算事实只存在于公共 analysis/emission index，不形成第四层 IR；
- 每个 target family 产生自己的 Physical Plan；
- Plan 只保存多个合法物理方案中选了哪个；
- surface leaf 只做 capability、机械投影和 runtime 接线；
- generic closure 不被 compiler 改写；
- compiler-private stages 不改变 logical callable 的 ABI/effects，也不开放跨 callable fusion；
- 不存在按 kernel 名称选择 realization 的入口。

### 仍需作为实现状态而非设计歧义看待的边界

- CPU/RISC-V/RVV target family 尚未接入；这不改变共享 Kernel IR 边界，但跨机器族复用还没有端到端实现证据；
- TileLang generic reduce/scan closure 是明确 target capability subset；
- `partition(count=...)` 有稳定 source 语义，但 realizer/frontend 路径尚未闭合；
- sparse contraction 的实现证据仍只有固定 2:4 schema，通用 format descriptor 尚不是 public API；
- `private_workspace` 的驻留选择仍未证明已经选优；
- 第二轮结构删减只做了受影响 repro，没有双机全量矩阵证据；
- 第一轮全量快照仍有 1 个 CSV failed（5090 TileLang `grouped_query_head_add`，其余 surface/设备通过）和 4 个下层 compile timeout；它们没有被伪装成 pass，也没有据此改写 Core。

因此这两轮完成的是：把模型中已经确认的真理来源、物理选择与 target 能力边界收敛干净，并把状态从设计文档剥离出来。它已经是可供最终冻结判断的代码与证据基线，但本报告不自行宣布“真正冻结”。
