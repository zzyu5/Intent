# 编译器边界复审与遗留关闭

## 结论

这一轮没有重复做一次“扫代码找信息丢失”的清单审计，而是用固定四问重新检查最近新增的 generic combine、stage execution、E8M0 cast 和 GQA validity 路径：

1. 这件事作者是否已经写进 Kernel IR；
2. 下层 target 是否已经原生承担；
3. 它是算法语义、可重算事实、已选物理决定，还是 target spelling；
4. 换成另一种 GPU surface 或未来 RVV/CPU 后，这个归属是否仍成立。

审计得到两个真问题并已修复：

- E8M0 cast 之前只有“当前 block-scaled f32 样本能过”，没有完整 dtype 合同：Triton 把 `0xff` 解成 `inf` 而不是 NaN，Triton/TileLang 的 E8M0→f16 也没有正确闭合。现在 Triton 和 TileLang 都按完整位编码机械解码，cuTile 继续委托其原生 dtype；在下层支持该 dtype 的 provider/device 组合上，相同 raw bytes 的 f32/f16 结果一致。
- staged contraction 的 feature axis 之前在多个合法候选存在时默选最后一个结果轴。这是与历史 staged-tile 同名问题同类的“当前语料恰好唯一，所以错误猜测没有暴露”。现在 GPU realizer 要求唯一候选，否则在 Plan 构造时诊断，不再按轴顺序猜。

generic combine 与 GQA validity 路径没有发现需要拆层或改写的隐藏特化。Stage execution 的当前 GPU 能力被重新定性得更准确：公共合同可以表达 stage slice、axis binding 和 same-stream execution，但当前 GPU realizer/三个 surface 真正闭合的是 **ragged staged contraction**，不是任意作者程序的通用 stage scheduler。

上一份报告的三个挂账也已关闭为明确状态，而不再以“以后再看”的模糊形式存在：

- `private_workspace` 系数是可回退的 GPU placement policy，不是 correctness 合同；当前真实边界没有证据要求修改。以后只有实际 placement/编译/性能失败才能重开。
- `partition(count=P)` 是语义已定义但尚未实现的 Core 子集；当前没有真实算法需要它，frontend 明确拒绝。它不是当前编译器静默缺口，也不因挂得久而自动获得实现优先级。
- timeout 与 unsupported 已逐格重新分类；没有把下层编译时间成本伪装成 target capability。5090 TileLang 的 `absorbed_mla_prefill` 原来被粗略记成 unsupported，本轮改为 `compile_failed`，因为它确实生成了 target source，只是当前候选在布局或资源阶段全部失败。

## 1. 本轮检查范围与验证纪律

本轮没有跑全量矩阵。检查范围只包含：

- 上一份报告中仍列为挂账的 residency、partition 和 compile-time 项；
- 两份 baseline 中所有非 pass 格子；
- generic combine、stage execution、E8M0 和 GQA validity 的纵向实现路径；
- 新发现问题的直接消费者。

一次性 probe 全部位于 `/tmp`，用于观察生成 Plan/target source 或实际跑一次数值；它们不进入语料、baseline runner 或测试设施。

## 2. 上一轮遗留的最终状态

### 2.1 Compile timeout 没有被改造成能力边界

当前 timeout 只有 `token_sparse_mla_prefill`：

| 设备 | provider | 当前状态 | 事实 |
|---|---|---|---|
| 5090 | Triton | `compile_timeout` | target source 可生成，首次下层编译超过整项时限 |
| H100 | Triton | `compile_timeout` | target source 可生成，超过整项 900 秒时限 |
| H100 | cuTile | `compile_timeout` | 能进入候选编译，但每个候选超过当前单候选时限，最终没有有效配置 |
| 5090 | cuTile | pass | 同一算法约 66 秒首次编译后成功，证明这不是 canonical capability 缺失 |

共享层没有因此增加 token-sparse 分支，三个 leaf 也没有把 timeout 改写成 unsupported。编译成本仍归下层编译器和 provider tuner，不进入 Kernel IR 或 Physical Plan。

### 2.2 Unsupported 逐格复核

CSV 的 `unsupported` 表示“当前设备上的这个 provider 路径明确不能兑现”，但不能再笼统解释为“整门 target 语言永远没有这种算法能力”。逐格根因如下。

#### Triton

| case | 设备 | 根因 | 定性 |
|---|---|---|---|
| `sparse_2to4_gemm` | 5090、H100 | 当前 Triton surface 没有 2:4 sparse MMA 投影 | target primitive 子集；emission 前明确拒绝 |

#### cuTile

| case | 设备 | 根因 | 定性 |
|---|---|---|---|
| `fp8_mqa_logits` | 5090、H100 | matrix-M 是 runtime-sized lane，当前 cuTile matrix projection 要求可静态兑现的 M tile | 当前 surface/primitive shape 子集，明确诊断 |
| `sparse_2to4_gemm` | H100 | 当前 cuTile surface 没有对应 2:4 sparse contraction primitive | target primitive 子集 |
| `block_scaled_matmul` | H100 | cuTile 直接报告 `float8_e8m0fnu is not supported on sm_90` | 下层 dtype/device capability；不是我们的 E8M0 spelling 问题 |

`block_scaled_matmul` 在 5090 cuTile 通过，正好说明该状态是 provider × device capability，不是算法或公共 Plan 缺口。

#### TileLang

| case | 设备 | 根因 | 定性 |
|---|---|---|---|
| `atomic_compare_exchange` | 5090、H100 | TileLang 0.1.13 surface 无 CAS primitive | target primitive 缺失 |
| `conv2d`、`variant_conv2d_reduce_order` | 5090、H100 | 当前 cooperative transfer 不能把两个 checked access footprint 合成一次 bulk copy | 当前 leaf projection 子集；明确拒绝串行伪支持 |
| `paged_attention`、`continuous_gqa_decode`、`paged_mla_decode`、`paged_splitk_attention`、`varlen_gqa_decode_logits`、`splitk_attention_reduce` | 5090、H100 | 单行 contraction 没有 native matrix projection；已否决慢一个数量级的 scalar product-reduce fallback | 当前 target primitive shape 子集 |
| `token_sparse_mla_prefill` | 5090、H100 | 当前 TileLang surface 没有机械 batched-GEMM projection | 当前 leaf/primitive 子集 |
| `fp8_mqa_logits` | 5090、H100 | FP8 contraction 的 matrix-M 是 runtime lane extent，不能兑现成当前 MMA primitive | 当前 target primitive shape 子集 |
| `absorbed_mla_prefill` | 5090 | 小 query tile 在 TileLang layout inference 冲突；能编译的大 tile 需要 212992 B 以上动态 shared memory，超过该设备可用量 | **不是 unsupported**；target source 已生成，改记 `compile_failed` |

同一个 `absorbed_mla_prefill` 在 H100 TileLang 上通过，定向结果为最大误差 `1.2207e-4`、p50/p95 `0.1027/0.1038 ms`。这进一步证明 5090 的失败不能被包装成“TileLang 没有这个算法能力”。本轮尝试加入 16/32 query 候选，实际分别撞到 layout inference conflict；64/128 候选继续超过 5090 shared-memory 上限，因此候选扩展没有形成合法实现，已撤回，没有留下无效搜索空间膨胀。

### 2.3 `private_workspace`：从挂账改为 policy 判据

当前选择规则读取真实设备的 `registers_per_unit`，用两个经验预算决定：

- 小 scalar array 是否保留为 scalar/local state；
- owner-private structured vector 是否放进 local vector；
- 放不下或访问不结构化时是否落入 private workspace。

这两个除数不决定程序是否正确：workspace 是合法 fallback，选错只表现为编译资源或性能问题。因此它们的正确分类是 **GPU target policy heuristic**，不是 Kernel IR 语义、Plan schema 能力或 correctness invariant。

上一轮已经用真实 kernel 压到 64/512 两个阈值，并同时验证了必须 spill 的动态访问。两台现有设备都报告 65536 registers/SM，所以无法用它们证明任意寄存器容量上的最优系数；但“无法证明普适最优”并不等于存在未关闭的 compiler bug。

本轮关闭规则是：

> 当前系数不再作为待办长期挂账。只有新的真实设备或 kernel 出现“placement 导致编译失败、资源溢出或显著性能异常”，才以那个具体事实重开；不能仅因系数带经验性就主动重设计。

### 2.4 `partition(count=P)`：明确的未实现 Core 子集

`count=P` 与 `extent=T` 的作者语义确实不同：P 会进入 part identity、partial-buffer ABI 或多个作者 kernel 共同观察的分片协议，所以不能把它静默改写成 extent，也不能拿物理 CTA 数冒充。

当前全部语料仍是 67 处 extent partition、0 处 count。重新检查 split-K、partial reducer、LayerNorm backward 分组、causal-conv backward 和 paged 分片后，仍没有一个现有算法必须让 P 成为 source-visible 值。

因此最终状态不是“还没想清楚”，而是：

- 它是已定义但当前未实现的 Core 子集；
- frontend 在构造 Kernel IR 前给源码位置诊断；
- 当前 GPU realizer 不声称支持；
- 只有真实算法必须观察 part count/identity 时才实现纵向闭环。

## 3. Generic combine 的四步复审

### 3.1 作者写下的内容没有被转成 Plan 第二真理

Combiner helper 的 accumulator schema、result schema、component identity、runtime capture 和 pure body 都在 canonical Kernel IR。`ReductionOp`/`ScanOp` 只保存 result space、axis、materialization 和已选 lowering 等物理信息，没有复制 closure body或重新编码 combine 数学。

`arg_reduce.max` 仍只是 frontend sugar：Kernel IR 中真正权威的是 `(value,index)` tuple combine 与 lowest-index tie closure。Target 原生 argmax 只能作为已验证的等价 spelling hint。

### 3.2 下层委托边界仍成立

- Triton：typed helper 机械变成 `tl.reduce`/`tl.associative_scan` 的 jitted combiner；
- cuTile：typed helper 机械变成 `ct.reduce`/`ct.scan` 的 `func`；
- TileLang：固定 combiner 走原生 ReduceKind；generic closure 在 source emission 前明确 unsupported，因为 `comm_reducer` 产生的 `tirx.Reduce` 没有当前 CUDA lowering。

Runtime capture 的 `(capture, valid)` carrier 只适配 cuTile/Triton tuple identity ABI，不进入作者 accumulator schema，也不让 compiler 分析或改写 closure 数学。

### 3.3 针对历史“单一触发形态”的探针

最低下标 tie 是最容易因 target 原语默认行为分叉的地方。一次性 probe 使用两行重复最大值：

```text
[1, 3, 2, 3] -> 1
[5, 5, 4, 5] -> 0
```

Triton、cuTile、TileLang 均返回 `[1,0]`。因此 cuTile `ct.argmax` 的当前投影与 Kernel IR 的 lowest-index 合同一致，没有因为 leaf 省略显式 tie 参数而形成第二套语义。

此前已有的 `N=4093` record scan + runtime capture 已经覆盖 block-local scan、跨块 scalar carry、capture-valid carrier 和 tail；本轮代码复审没有发现 leaf 按 kernel 名或邻近算子选择 combiner。

### 3.4 结论

Generic combine 纵向路径干净：作者 closure 在 Kernel IR，昂贵 provenance 是派生 facts，Plan 只选物理 realization，leaf 只做原生委托或明确拒绝。没有新增 shared-layer 特判。

TileLang 的 `isRankReducingReductionChain` 只用于过滤一个已知会产生错误 fragment layout 的 target tuner 组合；它不替换 closure、不选择算法，也不进入共享 Plan。它是 target-specific candidate legality，而不是假发射。

## 4. Stage execution 的四步复审

### 4.1 准确能力边界

当前 GPU realizer 只从“到达 scatter terminal 的 ragged contraction”生成 StageOp。三个 surface 的厚代码负责：

- 读取同一 stage slice；
- 读取 member/feature/reduction 的 StageAxis binding；
- 物化 compiler-private intermediate；
- 在 current CUDA stream 上按 stage ordinal launch。

所以当前实现不能被描述成“任意 Kernel IR 程序都可以自动切 stage”。更准确的合同是：

> Stage schema 与公共 preflight 表达 compiler-private execution slice；当前 GPU producer/consumer 能力子集是 ragged staged contraction。未来 RVV/CPU 或新的 GPU realization 可以产生别的 slice，但必须各自声明并闭合 capability。

这不是把 thick leaf 误判为第二个编译器。Leaf 没有重新选择 stage grouping；它只消费 GPU realizer 已经给出的 ragged contraction binding，并按目标 API 物化。

### 4.2 Plan 没有重新保存可派生算法事实

StageOp 只保留：

- stage identity；
- 已选 operation slice；
- `same_stream` synchronization。

StageAxisOp 保留 target realization 选中的 source dimension/domain、tile role 和 worker axis。Inputs、outputs、dependencies、terminal ownership、intermediate lifetime 都由公共 emission index 从 canonical SSA def-use 和 stage slice重算，没有重新进入 Plan schema。

当前 dependency index只沿跨 stage SSA value 建边；它没有声称处理任意 alias-only effect dependency。现有 GPU stage producer生成的两个真实家族都由 SSA intermediate 串接，且 effectful terminal只在最后 stage。因而这不是当前隐藏 correctness bug；如果未来真实算法需要两个 stage 仅靠对同一 external view 的 effects 排序，必须先扩充公共 effect dependency合同，不能让三个 wrapper继续靠 ordinal 猜。

### 4.3 找到并修掉的偶然轴选择

`stageFeatureAxis` 原来从 result axes 末尾向前找第一个非 member、非 unit axis。现有 MoE/grouped GEMM 的 staged result 恰好只有一个 feature axis，所以定向验证一直通过；一旦结果同时含两个非 member 轴，realizer 会静默把最后一个当 feature。

这与历史 bug 的形态一致：不是当下数值错，而是现有语料让一个不成立的唯一性假设看似成立。

修复后：

- 零个候选：诊断没有 staged feature axis；
- 一个候选：形成 StageAxis binding；
- 多个候选：在 GPU realizer 中诊断歧义；
- leaf 不再有机会按名字或轴顺序补猜。

直接消费该判据的 5090 repro 全部通过：MoE 与 grouped GEMM × Triton/cuTile/TileLang，共 6 条；数值均与 reference 一致。

## 5. E8M0 cast 的四步复审

### 5.1 发现的问题

E8M0 是 Kernel IR dtype 语义，不是 block-scaled matmul 的私有技巧。旧路径却只保证当前样本所需的 E8M0→f32：

- Triton ABI 把 E8M0 storage 暴露为 uint8，旧 leaf 用 `exp2(bits-127)`；`0xff` 因而变成 `inf`，而 E8M0 合同要求 NaN；byte 0 也可能因 exp2 的 subnormal 行为丢精度。
- Triton 对 E8M0→f16 直接把 uint8 数字 cast 成 f16，得到 `[0,1,127,254,255]`，不是解码后的数值。
- TileLang 的 E8M0→f32 有专门 decode，但非 f32 重新落回普通 `T.cast`；E8M0→f16 在下层所有候选上失败。
- cuTile 原生 `float8_e8m0fnu` 的 `ct.astype` 在 5090 上已经正确处理这些情况。

### 5.2 修复归属

修复仍在 target spelling 层：

1. 取 8-bit E8M0 encoding；
2. byte 0 映射为 float32 bits `0x00400000`；
3. byte 1..254 左移 23 位形成对应 power-of-two float32；
4. byte 255 映射为 canonical qNaN `0x7fc00000`；
5. 目标不是 f32 时，再从该 canonical f32 值机械 cast；
6. 目标仍是 E8M0 时保持原生 identity/cast 路径。

Triton 与 TileLang 各自只负责把同一 dtype 语义拼成自己的 bitcast/shift/select 语法；cuTile 已有原生能力，继续委托。没有 GPU 型号、kernel 名、block-scaled 形状或邻近 contract 分支，也没有把 dtype decode 塞进 Physical Plan。

### 5.3 探针和真实 kernel

一次性 raw-byte probe 输入 `[0,1,127,254,255]`：

| 设备/provider | E8M0→f32 | E8M0→f16 | argmax tie |
|---|---|---|---|
| 5090 Triton | 完全一致，含 NaN | 完全一致 | `[1,0]` |
| 5090 cuTile | 完全一致，原生委托 | 完全一致 | `[1,0]` |
| 5090 TileLang | 完全一致，含 NaN | 完全一致 | `[1,0]` |
| H100 Triton | 完全一致，含 NaN | 完全一致 | `[1,0]` |
| H100 TileLang | 完全一致，含 NaN | 完全一致 | `[1,0]` |
| H100 cuTile | 下层明确拒绝 E8M0 on sm90 | 同左 | `[1,0]` |

真实 `block_scaled_matmul` 也在两个设备上复验：

| 设备/provider | 最大误差 | p50 / p95 |
|---|---:|---:|
| 5090 Triton | `7.629e-06` | 0.0180 / 0.0187 ms |
| 5090 TileLang | `7.629e-06` | 0.0369 / 0.0390 ms |
| H100 Triton | `7.629e-06` | 0.0166 / 0.0173 ms |
| H100 TileLang | `7.629e-06` | 0.0412 / 0.0421 ms |

## 6. GQA validity 收紧的四步复审

### 6.1 为什么它不是 GQA 特判

BoundaryNeutralizationProof 只根据：

- load result 对 validity domain 是否有唯一 tensor-axis provenance；
- validity domain 是否沿 canonical value flow继续存在；
- 最终 store/scatter 是否仍检查该 domain；
- 如果 domain 被 reduction/contract 消去，另一操作数和 identity padding 是否能证明无效 lane 不可观察。

它不读取 kernel 名、GQA head 数或固定 shape。`intent.cast/broadcast/reshape/transpose/unary/binary/compare/select/mask` 的 pass-through 不是假设“数值仍是同一个 padding”；它只在 logical validity domain仍被后续 store检查时省去中间 materialization。对真正消去 domain 的 reduction/contract，仍走单独的 identity proof。

### 6.2 针对历史多角色/组合错误的探针

一次性 probe 在 GQA 的 derived scalar head index 后增加一个普通 `+1`，避免只验证原来那条完全相同的表达式：

- Triton，token extent 257：`consumer_neutralized=true`，最大误差 0；
- cuTile，token extent 257：`consumer_neutralized=true`，最大误差 0；
- TileLang，token extent 4096：`consumer_neutralized=true`，最大误差 0；
- TileLang，token extent 257：在运行前明确拒绝“guarded float16 bulk transfer 需要 exact power-of-two extent”，没有静默错误或串行慢路径。

普通非 derived-index 的 `load + 1 + store` 不设置 consumer-neutralized，三家在 extent 257 上同样数值正确。说明 proof 没有把所有 pointwise tail 粗暴归成一类。

### 6.3 结论

GQA 修复仍是共享 validity provenance 判据，不是只服务于 GQA 的分支。多个 validity domain、tensor-indirect 和 reduction/contract 都是 fail-closed；当前未发现与 W4A8“同一轴多角色覆盖”同形的事实丢失。

## 7. 对三类历史 bug 的对应检查

| 历史 bug 形态 | 本轮对应检查 | 结果 |
|---|---|---|
| 同一逻辑轴同时承担 stream/reduction，内层初始化覆盖全局索引 | generic scan 的 axis/carry 与 GQA validity domain 是否靠单一角色；stage axis 是否允许多个候选 | combine/validity 没有角色覆盖；stage feature 的多候选猜测已修 |
| member/feature/reduction tile 因名字和值碰巧相同而隐藏错误 | StageAxis 是否绑定 source value/domain + tensor axis；多个非 member 轴是否仍静默选一个 | binding 已 typed；最后一个轴 fallback 已删除 |
| 一个 cuTile 坏候选污染同 context 后续候选，伪装成整个空间无效 | timeout/unsupported 是否来自共享能力判断；absorbed MLA 是否真的 target 不支持 | timeout 保持 compile cost；absorbed MLA 改记 compile_failed，并逐候选定位 layout/resource 原因 |

## 8. 改动与定向验证汇总

永久代码改动只有三处职责明确的修正：

1. Triton E8M0 cast：完整位级 decode，再机械 cast；
2. TileLang E8M0 cast：同一语义的 target spelling；
3. GPU staged feature binding：要求唯一来源轴，删除轴顺序 fallback。

固定表只更新直接受影响事实：

- 两台机器的 block-scaled Triton/TileLang 定向数值与时间；
- 5090 TileLang `absorbed_mla_prefill` 从 `unsupported` 改为 `compile_failed`；
- source baseline 数字、表结构和其它格子不动。

实际验证：

- raw E8M0 f32/f16 + argmax tie：5090 三 provider；H100 Triton/TileLang，cuTile得到明确 sm90 capability rejection；
- `block_scaled_matmul`：5090/H100 的 Triton、TileLang；
- staged MoE/grouped GEMM：5090 三 provider，共 6 条；
- validity derived-index + pointwise 变化：5090 三 provider；
- H100 cuTile block-scaled capability：定向得到下层 `float8_e8m0fnu is not supported on sm_90`；
- 5090/H100 TileLang absorbed MLA：5090逐候选定位失败，H100数值与性能通过。

## 9. 关闭后的边界

这一轮后没有遗留未定位 correctness 失败，也没有仍以“未验证”措辞悬空的上一轮事项。剩下的是已经声明性质的能力边界：

- compile timeout：下层编译成本；
- `compile_failed`：target source 已生成，但当前 provider/device 没有有效 layout/resource candidate；
- unsupported：当前 provider/device 的 primitive 或 projection 子集明确不能兑现；
- `partition(count)`：语义存在、当前未实现、frontend fail-closed；
- private workspace 系数：已证实 correctness 可回退的 target policy，不再作为无证据重构任务。

未来若出现新的真实失败，仍按四问重新定位，不能因为某个状态已经登记就继承旧结论，也不能为了消灭 unsupported 而加入串行伪支持。
