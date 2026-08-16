# 编译器决策空间形式化与算法对齐审计

## 1. 报告目的与范围

这份报告独立记录两件已经完成的工作：

1. 把 Intent Kernel 编译器必须填写的物理决定形式化，明确哪些决定只能由共享层确定、哪些必须由 GPU realizer 选择、哪些应交给 Triton、cuTile、TileLang 自己的 tuner 实测。
2. 逐格审计两份固定性能表中带 upstream 数字的比较，确认 DSL 与 upstream 是否执行同一个算法、相同调用数和相同计时范围；不满足比较条件的数字不再保留。

本轮不是全量回归。验证只覆盖被决策归属修正、算法改写和 target 投影改动直接影响的 kernel。两份固定表继续作为数字记录，不承担检查、门禁或版本管理职责。

本报告中的“下层”指拿到最终 target source 后继续编译和调优的 Triton、cuTile 或 TileLang 工具链；“共享层”指仍然看得见 canonical Kernel IR、region、use-def 和跨 op 结构的分析、证明与 GPU realization。

## 2. 决策空间的形式化

### 2.1 分类对象

U、S、P 分类的对象不是 compiler 中出现的所有事实，而是：

> 在算法表示确定之后，编译器仍必须从一个候选集合中填写一个值的物理决定。

设算法表示为 `A`，target capability 与设备事实为 `C`，某个待填决定为 `d`，其合法候选集合为 `Legal(A, C, d)`。分类时依次回答三个问题：

1. 下层能否只根据它已经看见的 target source、shape、dtype、设备信息和真实运行结果选择 `d`？
2. 如果不能，算法语义与 target correctness 合同是否只允许一个合法答案？
3. 如果仍有多个合法答案，选择是否会改变 grid、循环、ownership、驻留或 stage 等源码结构？下层是否已经失去作出该选择所需的 Intent 结构信息？

由此得到三类决定。

### 2.2 U：唯一合法解

`|Legal(A, C, d)| = 1`。错误选择会改变地址、边界、数值、依赖或可见性，因此不是性能问题，而是 correctness 问题。

U 必须在共享 lowering 或 Physical Plan 中形成唯一权威来源。三个 target leaf 只能读取和机械兑现，不能各自再判断一次。

典型例子：

- 地址算术必须使用的语义位宽；
- reduction tail 的 identity 与 padding；
- 已选 stage 之间的拓扑依赖、读写归属和内存可见性；
- 已选 physical range 与 region argument/use 的绑定；
- affine access footprint 与 transfer/source-axis range 的精确绑定。

U 与派生事实不同。provenance、use-def、result axes 等信息已经存在于 Kernel IR，可以随时重算；它们不是编译器在多个答案中作出的选择，不应为了方便消费而复制成第二份 Plan 真理。

### 2.3 S：结构性选择

`|Legal(A, C, d)| > 1`，多个答案都保持算法语义，但不同答案会改变生成源码的拓扑结构；下层拿到 target source 时，已经看不到作出这个判断所需的 Intent region、ownership、use-def 或跨 op 关系。

S 由 target-family realizer 选择，并记录在 Physical Plan。leaf 只读取已选结果。

典型例子：

- 逻辑轴到 lane、program、worker、stream 等物理用途的分配；
- persistent traversal 是否存在；
- pointwise lane promotion 是否成立；
- program order、worker folding、program-group membership 与 worker reuse；
- logical buffer、scan result、transfer result 和 contraction operand 的驻留层级；
- indirect-ragged 或 serial traversal 是否标量化。

作者显式写下的 `partition` 不属于 S。它决定 kernel body 看见一个元素还是一个 region，是算法程序结构；编译器不能擅自增加、删除或替换。编译器为这个 region 选择 physical extent，才进入物理决定空间。

### 2.4 P：参数性选择

`|Legal(A, C, d)| > 1`，不同答案不改变 Kernel IR 或 Physical Plan 的结构，只改变 tile、launch 数值，或者同一 target 上等价的 API 拼写。下层具备编译、运行并实测 winner 的完整信息与能力。

P 必须交给相应 provider 的 tuner。Intent 只提供合法候选集合，不用静态阶梯、经验阈值或设备型号分支替下层猜 winner。

典型例子：

- concrete tile、scan chunk、program-group width；
- Triton `num_warps`、`num_stages`；
- cuTile occupancy、`num_ctas`；
- TileLang threads、stages、`GemmWarpPolicy`；
- cuTile masked gather 与 gather-plus-where 两种等价拼写；
- Triton native scaled dot 与显式 scale decode 后普通 dot 的等价 target 投影。

P 不只包括整数参数。只要语义与 Plan 完全相同、区别只存在于 target surface 的等价写法，就属于 target-local P。

### 2.5 不属于 U/S/P 的内容

以下四类内容不是物理决定：

1. **派生事实**：由 Kernel IR 唯一重算的 provenance、use-def、result axes、logical roles、domain extent。
2. **capability 与 legality**：例如某设备不支持 E8M0、某 surface 只接受 32-bit descriptor、某 target 没有 generic combiner。
3. **目标语法映射**：canonical op 到 `tl.*`、`ct.*`、`T.*` 的名字、dtype spelling、参数顺序。
4. **测量与运行策略**：warmup、timeout、CUDA graph、cache flush、不可安全 replay 的 effect 检查。

这四类内容分别回答“事实是什么”“目标能否表示”“目标语法怎么写”“怎么可靠测量”，而不是“多个合法物理实现里选哪一个”。

### 2.6 最终判定顺序

以后新增或审查一个决定时，固定使用以下顺序：

1. 能从 Kernel IR 唯一重算：保留为派生查询，不进入 Plan。
2. 是 target 能力边界：放在 capability/legality，不能伪装成 tuner candidate。
3. 是纯 target 拼写：放在 leaf syntax/op handler。
4. 是运行和测量约束：放在 runtime，不进入物理 Plan。
5. 其余才进入 U/S/P：唯一正确答案是 U；改变源码结构且下层缺少 Intent 信息是 S；其余可由 provider 实测的是 P。

审计没有发现无法用这套顺序归类的现有物理决定。

## 3. 现有决定的全量归属审计

| 决定或事实 | 分类 | 唯一权威位置 | 审计结论 |
|---|---|---|---|
| logical index provenance、地址表达式、domain extent、use-def、result axes | 派生事实 | Kernel IR + shared query | 不物化进 Plan；找不到精确来源直接诊断，不按相同 extent 猜轴 |
| 地址算术宽度 | U + capability | shared correctness contract + leaf guard | 语义地址使用 i64；cuTile/TileLang 在投影到较窄 descriptor 前证明 offset 上限，不能静默窄化回绕 |
| region argument 到 logical axis/purpose | 派生事实 | Kernel IR + shared facts | 作者 region 语义的稳定绑定，不是物理候选 |
| 已选 physical range 到 region argument/use | U | Physical Plan binding | leaf 不再从 tensor shape 重建 |
| affine footprint 到 transfer/source-axis access range | U | Physical Plan access binding | 每个 footprint 精确绑定已有 ownership range |
| validity、reduction identity、padding、tail fill | U | shared proof + Plan | leaf 只兑现 mask/fill |
| full-domain row/lane 的最小合法 GPU extent | U | shared GPU legalization | body 已要求看见完整逻辑域；最小 power-of-two fragment 与 tail identity 是 correctness 兑现，不是 tuner tile |
| stage dependency、I/O ownership、visibility | U | Physical Plan stage contract | 不依赖 emitter 恰好按同一 stream 顺序 launch |
| parallel/ordered/reduction/contraction/ragged 等逻辑角色 | 派生事实 | Kernel IR + shared facts | 从 op、region 与 use-def 唯一推出，角色可以组合 |
| 逻辑角色到 lane/program/worker/stream 的用途分配 | S | shared GPU realizer | 按轴组合，不按 kernel 类别分支 |
| indirect-ragged/serial traversal 标量化 | S | shared GPU realizer | 改变循环与搬运形态，不能冒充普通 tile 参数 |
| program order、worker folding、group membership、reuse-worker | S | shared GPU realizer | provider tuner 不重新决定 ownership |
| program-group 的具体 width | P | provider tuner | Plan 中 group membership 是 S；`GROUP_SIZE_M` 等数值是 P |
| persistent traversal | S | shared GPU realizer | 改变循环和 grid，不能下放为 launch 常量 |
| pointwise lane promotion eligibility | S | shared GPU realizer | 仅对纯逐元素 body；不能自动把 contraction 改写成块算法 |
| buffer、transfer、contraction operand、scan result 驻留 | S | shared GPU Plan builder | 由 owner、lifetime、reuse 与消费关系选择，leaf 只拼写 |
| concrete ownership/traversal/reduction tile 与 scan chunk | P | provider tuner | SearchSpace 只声明合法参数轴，不固定 winner |
| `num_warps`、`num_stages`、threads、occupancy、`num_ctas` | P | provider tuner | 已清除固定 row winner；不可 replay 的 kernel 属于 runtime legality |
| TileLang `GemmWarpPolicy` | P | TileLang tuner | FullRow、Square、FullCol 作为离散候选，由 target compiler 过滤非法组合 |
| cuTile gather 两种等价拼写 | P | cuTile leaf tuner | 两台设备可选不同 winner，不进入共享 Plan |
| Triton scaled-dot 两种等价拼写 | P | Triton leaf tuner | 同一 `scaled_contract` 和同一 Plan，只改变 target surface |
| matrix unit、descriptor 位宽、scaled group 等限制 | capability | target family / leaf | 不支持时在编译或 launch 前明确拒绝，不制造假候选 |
| canonical op 到 target intrinsic | syntax | leaf handler / spelling | 逐 op 机械投影，不把 op kind 复制进 Plan |
| tuner eligibility、cache key、warmup、timeout | runtime | shared effect query + provider runtime | cache key 覆盖影响 winner 的 shape、dtype、device；不是 Plan 决定 |

审计后的固定数字只允许承担四种角色：U 的唯一兑现、target capability 上限、S policy 的结构阈值、P 搜索空间中的候选值。候选表里出现 `8` 不等于 winner 被固定为 `8`。

## 4. 发现并修正的决策错位

### 4.1 Triton row launch 参数

原实现使用 `block_size >= 32768 → 32 warps`、`>= 8192 → 16`、`>= 2048 → 8`、否则 `4` 的设备无关阶梯，并在另一条 row configuration 中固定 `num_warps=8`、按 shared-memory 大小二选一 `num_stages`。

这些值不改变算法和源码拓扑，Triton autotuner 又能直接实测，因此属于 P。固定阶梯已删除；合法 warp、stage、row occupancy 组合由 Triton tuner 联合测量。普通 row wrapper 的 cache key 从单一 lane extent 扩成完整动态 shape，避免不同 shape 误用同一个 winner。

### 4.2 cuTile 与 TileLang 的 row 参数

cuTile row occupancy、TileLang row threads/stages 同样属于 P，已进入各自 tuner。effectful 或不可安全 replay 的 kernel 不强行 autotune；这是 runtime legality，不是重新写死一个物理 winner。

### 4.3 TileLang GEMM warp policy

TileLang leaf 原先在六处固定 `GemmWarpPolicy.FullRow`。FullRow、Square、FullCol 不改变 canonical contraction 或 GPU Plan，只是 TileLang surface 的离散执行参数，因此改成 `gemm_warp_policy` target-local candidate，并与 tile、threads、stages 一起实测。

扩充 candidate profile 时曾出现一个真实错误：selector 因 profile 多带了当前 kernel 没有消费的字段而删除旧的合法候选，H100 sparse GEMM 从约 `0.1807 ms` 退到 `0.2205 ms`。修正后 selector 只比较当前 roles 的覆盖度，旧、新候选可以共存，最终恢复到 `0.1815 ms`。修复的是候选集合合法性，没有加入设备型号分支。

### 4.4 重复 domain extent 推导

Triton、cuTile、TileLang 三个 leaf 原先各自递归解析 ragged/domain extent，共有一百余行重复逻辑。domain extent 是由 Kernel IR 与 ABI shape 唯一重算的派生事实，不应复制成三个判断，也不应物化进 Plan。

现在三个 leaf 统一调用 `target::emission::logicalDomainExtent`。共享的是查询规则，不是第二份存储 schema。

### 4.5 scaled contraction 的边界

旧 block-scaled matmul 在 DSL 中写成 E8M0 decode、逐元素 scale 和普通 contraction，而 upstream cuTile 使用原生 `ct.mma_scaled`。在 leaf 中 pattern-match 并折叠普通 op 链会改写作者算法，违反 Kernel IR 不改写原则；继续把两个算法挂在同一个 baseline 下也不公平。

Triton `tl.dot_scaled` 与 cuTile `ct.mma_scaled` 都证明“带显式 scale tensor 的 contraction”是独立、真实的算法角色，因此增加 canonical `intent.scaled_contract`：

- Kernel IR 只保存 data/scale operands、FP8/E8M0 format、scale group、reduction pair 与 accumulator dtype；
- 不保存 warp、tile、layout、storage 或 provider 字段；
- 普通 contraction 与 scaled contraction 共用轴角色、padding proof 和 Plan contraction；
- Triton、cuTile、TileLang 分别逐 op 投影，不从普通 op 链识别 pattern。

Triton 的 native scaled dot 与显式 decode 后普通 dot 是 P，由 tuner 实测；cuTile 在支持的设备上直接委托 `ct.mma_scaled`；TileLang 0.1.13 没有同级 scaled MMA surface，机械 decode 后调用 `T.gemm`，没有向共享层加入 TileLang 专属决定。

## 5. 算法对齐审计方法

每个存在 upstream 数字的 provider cell 只能得到三种结论：

- **A：算法一致。** 算法、kernel 调用数、输入输出、workspace 位置与计时范围一致，保留 source 数字；差距是 compiler/provider 问题。
- **B：本应一致但接错。** DSL 或 adapter 没有照 upstream 算法实现；先改写并重新测量，旧数字不能继续使用。
- **C：不应对齐。** upstream 是另一算法或另一调用编排，而把 DSL 改成它会改变作者选择；撤掉 source 数字。

计时比较遵守两条边界：单 kernel 对单 kernel时比较 kernel-only；融合或调用数不同只能在完全相同的端到端 scope 下比较。为 adapter 反复计时而增加的 copy、concat、gather 或 workspace 初始化不能只算在一侧。

审计覆盖更新后两份 CSV 中全部非空 source cells，没有保留“算法不同但先挂着看看”的第四种状态。

## 6. 保留的 upstream 对照

| kernel / case | provider | 结论 | 对齐依据 |
|---|---|---|---|
| softmax | Triton、cuTile | A | 单次 row load/reduce/store，kernel-only |
| layer_norm | Triton、cuTile | A | 同一 forward normalization 与 affine |
| layer_norm_backward | Triton | B→A | adapter 直接调用 upstream 两个 backward kernel；forward-saved mean/rstd 在计时外 |
| rms_norm | Triton | A | 同一 weighted RMSNorm |
| fused_add_rms_norm | Triton | A | 同一 fused residual + RMSNorm |
| cross_entropy | Triton | B→A | DSL 改成单 kernel fused loss/prediction/in-place gradient，与 Liger scope 一致 |
| gemm/base | 三家 | A | 同形状、dtype、单 contraction |
| bf16_gemm | cuTile、TileLang | A | 同一 bf16 GEMM |
| batched_gemm NN/TN/NT/TT | cuTile | A | batch、四种 transpose 与调用数一致 |
| attention | 三家 | A | dense forward attention，同一 QKV/O scope |
| attention_bias | Triton | A | 同一 fused bias attention |
| varlen_attention/causal | TileLang | A | 同一 causal varlen online attention |
| varlen_gqa_prefill | TileLang | A | 同一 varlen GQA prefill |
| paged_attention | Triton | A | 同一 paged decode attention |
| online_softmax | TileLang | A | 两边都是 online/two-pass 算法 |
| grouped_gemm/base | 三家 | A | grouped problem 与端到端 list/result scope 一致 |
| swiglu_forward | Triton、cuTile | A | 同一 fused SiLU×up |
| swiglu_backward | Triton | A | 同一 backward kernel |
| rope_qk_full/partial/inverse | cuTile | A | Q/K、旋转维度与方向逐 case 一致 |
| mla_prefill | cuTile | A | 同一 MLA prefill contraction 结构 |
| w4a8_packed | TileLang | A | 同一 signed packed-W4/A8 dequant GEMM |
| embedding_forward_lookup | Triton | A | 同一 embedding gather |
| embedding_backward_atomic | Triton | A | 同一 atomic embedding gradient；只有真正测得 source 的设备保留数字 |
| block_scaled_matmul | cuTile | B→A | Core 直接表达 E8M0 scaled contraction，投影为 `ct.mma_scaled` |
| splitk_attention_reduce | cuTile | A | 同一 split-K partial reduction |
| fp8_gemm e4m3/e5m2 | TileLang | A | dtype 和单 GEMM scope 分别一致 |
| index_select_rows | Triton | A | 同一 row gather |
| scaled_index_add | Triton | A | 同一 indexed scaled add |
| sparse_2to4_gemm | TileLang | A | 同一 2:4 metadata contraction |

## 7. 改写后完成对齐的算法

### 7.1 Cross Entropy

旧 DSL/adapter 没有严格复现 Liger 的单 kernel fused 路径。现在 DSL 在一次调用中完成 loss、prediction 与 in-place gradient，与 upstream 的计算和计时 scope 对齐。

定向结果：

| 设备 / provider | generated p50 | source p50 | 结论 |
|---|---:|---:|---|
| RTX 5090 / Triton | 0.3432 ms | 0.3421 ms | 差约 0.3%，同一量级 |
| H100 / Triton | 0.1944 ms | 0.2170 ms | generated 快约 10.4% |

### 7.2 LayerNorm backward

DSL 和 adapter 现在都消费 forward 保存的 mean/rstd；upstream adapter 直接调用两段真实 backward kernel，workspace 与统计量准备在计时外。此前不可比的 adapter 开销已经移除。

| 设备 / provider | generated p50 | source p50 | 结论 |
|---|---:|---:|---|
| RTX 5090 / Triton | 0.0826 ms | 0.0625 ms | generated 慢约 1.32×，是真实 compiler gap |
| H100 / Triton | 0.0809 ms | 0.0858 ms | generated 快约 5.7% |

5090 的差距不能继续归因于算法或 adapter。generated `scatter_reduce` 需要清零两个 f32 partial buffers；upstream 使用 row-group lock 和 bf16 partial buffer。剩余问题是 collision/partial-storage realization 的共享设计，不是 LayerNorm 名字特例。

### 7.3 Block-scaled matmul

DSL 现在直接写 `scaled_contract`，不再用普通 op 链模拟另一个算法。cuTile baseline 因而从 B 收敛为 A；Triton 与 TileLang 也从同一个 canonical op 自然投影。

| 设备 | Triton generated | cuTile generated | TileLang generated | cuTile source |
|---|---:|---:|---:|---:|
| RTX 5090 | 0.0118 ms | 0.0139 ms | 0.0258 ms | 0.1390 ms |
| H100 | 0.0169 ms | capability unsupported | 0.0280 ms | 不可运行 |

H100 cuTile 的失败是当前 API/设备组合不能表达 E8M0 的 capability 边界，不是候选调优失败，也没有被提升成共享 Core 限制。

## 8. 撤掉的 upstream 对照

| kernel | provider | 撤掉原因 |
|---|---|---|
| softmax | TileLang | source 是 online/two-pass，DSL 是 stable single-pass |
| rms_norm | TileLang | source kernel 不含相同权重路径，adapter 在 kernel 外乘权重 |
| dual_gemm | 三家 | source 是两个独立 GEMM 加外部 epilogue，DSL 是单 fused kernel |
| online_softmax | Triton、cuTile | source 是普通 stable softmax，DSL 明确选择 online/two-pass |
| moe | 三家 | source 是两个 grouped GEMM 加 merge，DSL 是一个 fused ragged FFN kernel |

这些记录继续保留 generated 数字，但 source cell 为空。撤掉对照不是放弃 kernel，而是停止用不同算法的数字推导编译器性能结论。

## 9. 定向性能验证

本轮没有重跑全部 122 条记录，只跑被改动真正影响的项。所有标为 pass 的定向运行都完成真实 GPU 数值对照。

TileLang warp-policy 与候选集合修改的结果：

| 设备 / case | 修改前 generated p50 | 修改后 p50 | 结果 |
|---|---:|---:|---|
| RTX 5090 GEMM/base | 2.1181 ms | 2.0475 ms | 数值通过，快 3.3% |
| RTX 5090 GEMM/tail | 2.1325 ms | 2.0819 ms | 数值通过 |
| RTX 5090 sparse 2:4 | 0.2309 ms | 0.2103 ms | 数值通过，快 8.9% |
| RTX 5090 block-scaled | 0.0279 ms | 0.0258 ms | 数值通过，快 7.5% |
| H100 GEMM/base | 1.7485 ms | 1.7468 ms | 数值通过，未退化 |
| H100 GEMM/tail | 1.8982 ms | 1.8126 ms | 数值通过，快 4.5% |
| H100 sparse 2:4 | 0.1814 ms | 0.1815 ms | 数值通过，0.1 μs 差异；旧候选仍在空间内 |

同一候选机制在两台设备上选择的 winner 可以不同，且没有 `if sm90`、`if sm120` 等架构型号分支。这正是 P 交给 provider tuner 的预期行为。

## 10. 代码落点与边界

本轮主要代码落点如下：

- canonical `scaled_contract`：`include/Intent/Dialect/Intent/IR/IntentOps.td` 与 Python frontend structured lowering；
- shared contraction facts、proof 与 Plan binding：`lib/Target/Common/Realization/`、`lib/Target/GPU/Realization/`；
- 三 target 的逐 op 投影：`lib/Target/{Triton,CuTile,TileLang}/Emission/Handlers/Operations.cpp`；
- target-local 参数候选：`python/intent/runtime/tuning/{triton,cutile,tilelang}.py`；
- 共享 domain extent 查询：`include/Intent/Target/Common/Emission/SurfacePlan.h`；
- 对齐后的 DSL：`examples/kernels/loss/cross_entropy.py`、`examples/kernels/backward/layer_norm.py`、`examples/kernels/contraction/block_scaled.py`。

对应提交：

- `12b2e8e`：Cross Entropy 与 upstream 算法对齐；
- `767c2e5`：launch 参数委托与 canonical scaled contraction；
- `ba06622`：backward 算法与 baseline scope 对齐；
- `31a776a`：scaled contraction target spelling 委托；
- `a0f7de7`：共享 extent 查询与 TileLang GEMM warp-policy tuning；
- `d82d1f4`：扩充 TileLang contraction 候选；
- `fc17060`：保留覆盖度相同的合法候选；
- `1e3c67d`：固定表和原统一报告中的审计结果落盘。

## 11. 最终结论与仍然存在的边界

这轮形成的硬边界是：

- U 只能有一个权威来源；
- S 只能由仍看得见 Intent 算法结构的 realizer 选择；
- P 必须交给 provider tuner 实测；
- 派生事实不复制进 Plan；
- capability 不伪装成搜索空间；
- target syntax 不反向进入共享层；
- baseline 只保留算法与测量 scope 真正一致的比较。

当前仍然存在但没有被伪装成完成的事项：

1. RTX 5090 Triton LayerNorm backward 仍比对齐后的 upstream 慢约 1.32×，根因范围已经收敛到 grouped partial-reduction realization。
2. W4A8 的 compact quasi-affine coverage、ordered-ragged attention 的联合物理决定仍有真实性能差距。
3. CPU、RISC-V、RVV target family 尚未接入；当前决策模型没有把 GPU tile 写进 Kernel IR，但跨机器族 realizer 的完整性和性能尚未被证明。

这些是 realization 或新 target-family 工作，不构成修改 U/S/P 判据的理由。以后若出现无法归类的真实决定，应先修正判据，而不是把它临时塞进 leaf 常量、Plan 字段或 tuner 候选。
