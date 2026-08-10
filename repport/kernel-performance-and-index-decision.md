# 统一测量口径后的 Kernel 性能与缺口审计

## 结论

这一轮修正了判断问题的前提，而不是继续扩张机制：

1. benchmark 已明确分成 kernel-only、end-to-end 和 runtime-launch 三种口径。生成侧输出、workspace、编译与 materialization 均移出 kernel-only 区间；可捕获的短 kernel 在 generated/upstream 两边都使用 CUDA Graph。
2. cuTile dense attention 已重新连接当前 TileGym 实现，旧路径的约 68 ms 假差距消失；当前 generated/upstream 为 `5.0562 / 4.8477 ms`，只差 4.3%。
3. 当前真正干净、且超过 5% 的 compiler gap 仍是两组：cuTile BMM 的 NN/NT/TT 布局，以及 TileLang causal varlen attention。cuTile BMM 的共享候选空间已扩展，TN 已收敛到 2.8%，其余三种仍慢 10%–12%。
4. 旧报告“先建立 quasi-affine 索引 IR”的结论撤回。`+`、`//`、`%` 已经作为普通 Kernel MLIR SSA 运算存在；缺口是 realization 没有为 scalar pointwise 结果保留 logical-axis provenance，三个 emitter 随后又把动态索引替换成了物理轴基址。应先保留现有 use-def 与精确 SSA 表达式，不应再造一套索引表达式 IR。
5. LayerNorm backward 的问题确认在 DSL 源码：当前每行写一份 f32 partial，再扫完整的 `M×N` workspace；上游用固定小分组和原子部分和。这不是 realizer 应暗中改写的选择。
6. SwiGLU backward 的 adapter clone 已移出计时，真实结果变为 Triton `0.0934 / 0.0320 ms`。但 DSL 明确要求三路输入先转 f32、使用独立输出，上游只显式提升 `a` 且原地覆盖 `a/b`；同时 generated 固定采用 persistent grid-stride。它是“源码数值/ABI选择 + 物理映射”混合问题，尚不能归成第三个纯 compiler gap。

## 测量合同

表中数字均为 p50，单位为毫秒。`G/U` 表示 generated/upstream；单元格前缀表示计时范围：

测量设备为 NVIDIA GeForce RTX 5090 D；表格以本轮最终 measurement path 的最近一次结果为准。全量运行构成共同基线；随后专门复跑的 LayerNorm backward 与 SwiGLU backward 覆盖各自在全量日志中的旧行，其他 kernel 的编译与执行路径没有再变化。

- `K`（kernel-only）：输入、输出、workspace 和编译产物在区间外准备，区间内只重放 device launch。只有算法相同且 kernel 数量相同时，`G/U` 才用于判断生成 kernel 是否更快。
- `E`（end-to-end）：区间内保留用户为了得到该结果必须支付的全部 GPU 工作。融合改变 kernel 数量时，只在这一口径讨论收益。
- `R`（runtime-launch）：launch wrapper 仍需读取动态 GPU metadata，例如 varlen 路径中的 `.item()`；它不是 kernel-only，不能与 `K` 混算比值。
- `—`：当前没有可用上游测量。`†`：虽有数字，但算法或执行范围不同，不作为 kernel 性能结论。

具体实现位于 `examples/repro/common/support.py:15-71`：`prepare_kernel_call` 预绑定生成器输出并绕过 `run()` 的输出分配；`benchmark(..., cuda_graph=True)` 先在独立 stream 捕获，再对 graph replay 记录 CUDA Event。所有可捕获的短 kernel 两边对称使用 Graph；grouped GEMM、MoE、LayerNorm backward 等动态多阶段路径使用 Event。常规与 extended 项均取 100 次采样，MoE 保持 20 次大 kernel 采样。

LayerNorm backward 的 Triton upstream 只能经 autograd wrapper 进入，无法拆出内层 kernel，且在当前环境无法安全 CUDA-Graph capture。因此只保留数值核验，性能明确记为 unavailable，不再用 wrapper Event 时间冒充 kernel-only。全量日志中曾出现的 `0.3173 ms` 是修正前对整个 autograd callable 记录的 Event 时间，已由最终专用复跑的 `generated=0.3601 ms, upstream=unavailable` 覆盖，不进入表格。

## 同一代码基线的完整性能表

“最快 G/U”只是把可用数字连同其 scope 一起列出；scope 不同或带 `†` 时，不能据此计算跨 provider 加速比。

| Kernel / 实际形状 | Triton G/U | cuTile G/U | TileLang G/U | 最快 G | 最快 U |
|---|---:|---:|---:|---:|---:|
| stable softmax, f32 `8192×8192` | K `0.3560 / 0.3558` | K `0.3580 / 0.3577` | K `0.3492 / 0.3621` | TileLang K `0.3492` | Triton K `0.3558` |
| LayerNorm fwd, f32 `8192×4096` | K `0.1772 / 0.1727†` | K `0.1799 / 0.3458†` | K `0.1734 / —` | TileLang K `0.1734` | Triton K `0.1727†` |
| LayerNorm bwd pipeline, bf16 `4096×4096` | E `0.3601 / —†` | E `0.2433 / —` | E `0.2739 / —` | cuTile E `0.2433` | — |
| weighted RMSNorm, f32 `8192×4096` | K `0.1770 / 0.1748` | K `0.1795 / —` | E `0.1732 / 0.3499†` | TileLang E `0.1732` | Triton K `0.1748` |
| fused add RMSNorm, bf16 `8192×4096` | K `0.1778 / 0.1758` | K `0.1793 / —` | K `0.1737 / —` | TileLang K `0.1737` | Triton K `0.1758` |
| row logsumexp, f32 `8192×8192` | K `0.1627 / —` | K `0.1692 / —` | K `0.1609 / —` | TileLang K `0.1609` | — |
| GEMM f16 `4096×4096×14336` | K `2.1163 / 2.0773` | K `2.1326 / 2.2818` | K `2.1546 / 2.3724` | Triton K `2.1163` | Triton K `2.0773` |
| GEMM M/N/K tail `4093×4080×14320` | K `2.1419 / —` | K `2.0866 / —` | K `2.1306 / —` | cuTile K `2.0866` | — |
| BF16 GEMM `4096×4096×14336` | K `2.0788 / —` | K `2.0421 / 2.2391` | K `2.1016 / 2.3036` | cuTile K `2.0421` | cuTile K `2.2391` |
| BF16 BMM NN, `32×512×1024×512` | K `0.0852 / —` | K `0.0899 / 0.0816` | K `0.0995 / —` | Triton K `0.0852` | cuTile K `0.0816` |
| BF16 BMM TN, same logical shape | K `0.0898 / —` | K `0.0836 / 0.0813` | K `0.1000 / —` | cuTile K `0.0836` | cuTile K `0.0813` |
| BF16 BMM NT, same logical shape | K `0.0857 / —` | K `0.0916 / 0.0816` | K `0.0993 / —` | Triton K `0.0857` | cuTile K `0.0816` |
| BF16 BMM TT, same logical shape | K `0.0877 / —` | K `0.0915 / 0.0816` | K `0.1000 / —` | Triton K `0.0877` | cuTile K `0.0816` |
| fused int8-output GEMM `4096×4096×14336` | K `2.1388 / —` | K `2.2487 / —` | K `2.1935 / —` | Triton K `2.1388` | — |
| gated dual GEMM `2048×4096×4096` | E `0.6976 / 0.7757†` | E `0.6832 / 0.8076†` | E `0.7383 / 0.9243†` | cuTile E `0.6832` | Triton E `0.7757†` |
| dense attention f16 `(4,32,4096,128)`, noncausal | K `5.0009 / 4.9757` | K `5.0562 / 4.8477` | K `4.8917 / 6.6956` | TileLang K `4.8917` | cuTile K `4.8477` |
| vector-bias attention, same shape | K `5.1641 / 7.0941†` | K `5.4977 / —` | K `5.0400 / —` | TileLang K `5.0400` | Triton K `7.0941†` |
| packed varlen attention, `U=29114,D=128`, noncausal | R `0.4187 / —` | R `0.3825 / —` | K `0.4579 / —` | cuTile R `0.3825` | — |
| packed varlen attention, same input, causal | R `0.3174 / —` | R `0.2874 / —` | K `0.3212 / 0.2529` | cuTile R `0.2874` | TileLang K `0.2529` |
| streamed online softmax, f32 `8192×8192` | K `0.3820 / 0.3560†` | K `0.3513 / 0.3580†` | K `0.3841 / 0.3621†` | cuTile K `0.3513` | Triton K `0.3560†` |
| MoE `T=4096,D=4096,F=14336,E=8,top-k=2` | E `8.9172 / 10.3062†` | E `10.2524 / 9.5964†` | E `11.4495 / 9.3652†` | Triton E `8.9172` | TileLang E `9.3652†` |
| grouped GEMM `R=8192,K=N=4096,G=8` | E `1.3093 / 1.8736†` | E `1.4548 / 1.8244†` | E `1.2787 / 1.4260†` | TileLang E `1.2787` | TileLang E `1.4260†` |
| grouped GEMM member/K/N tail `8191×4080×4080` | E `1.3212 / —` | E `1.4600 / —` | E `1.2868 / —` | TileLang E `1.2868` | — |
| SwiGLU fwd bf16 `4096×14336` | K `0.2264 / 0.2229` | K `0.2326 / 0.2331` | K `0.2674 / —` | Triton K `0.2264` | Triton K `0.2229` |
| SwiGLU bwd bf16 `4096×4096` | K `0.0934 / 0.0320†` | K `0.0959 / —` | K `0.0953 / —` | Triton K `0.0934` | Triton K `0.0320†` |

## 慢项归因

### 可直接归为 compiler gap

| 项目 | 当前比值 | 事实归因 |
|---|---:|---|
| cuTile BMM NN | `1.102×` | 数学、dtype、kernel 数量和 layout 语义一致；上游使用持久化调度与成熟的 load/transposition 路径。 |
| cuTile BMM NT | `1.123×` | 同上；转置访问仍未被当前通用 contraction 投影兑现到上游水平。 |
| cuTile BMM TT | `1.121×` | 同上；两侧转置组合继续暴露 load/traversal 差距。 |
| TileLang causal varlen attention | `1.270×` | 同一 self-varlen causal 算法、一个 kernel。generated 将有界 Q/K/V transfer 展开成逐元素 `T.Parallel` predicate，并插入同步；上游用 bulk `T.copy` 后在 score 上消除无效 lane。缺的是“可安全投机 bulk load”的通用合法性证明与投影，不是 attention 特判。 |

cuTile 的 generic contraction 候选由 2 个扩为 24 个，仍由 `exhaustive_search` 选择，没有 BMM 名字分支。TN 已从明显差距收敛为 `1.028×`，因此不再列为 compiler gap；NN/NT/TT 的剩余差距不是继续堆 tile 常数即可解释。

TileLang 的 query/stream 候选已加入 `64×64, stage=2, threads=128`。它能被 tuner 选中，但 causal varlen 仍为 `0.3212 / 0.2529 ms`，说明当前瓶颈不是单一 stage 参数，而是 transfer 兑现方式。

### 应先改 DSL 源码或明确语义

- **LayerNorm backward**：`dw_partial/db_partial` 是两张 `4096×4096xf32`，共 128 MiB；上游在本形状使用 128 个分组、两张约 2 MiB 的 bf16 partial。源码应明确写出分组 ownership、原子合并和第二阶段归约，realizer 不应把作者的可观察 workspace 偷换成另一算法。
- **LayerNorm forward**：DSL 使用 `E[x²]-E[x]²`，上游使用 centered-square variance；数学目标相同，数值和访存算法不同。
- **SwiGLU backward**：clone 污染移除后暴露 `2.9146×`。两边公式和 kernel 数量相同，但 DSL 明确把 `dc/a/b` 都 cast 为 f32 并要求独立 `da/db`，上游只显式提升 `a` 并原地写回 `a/b`；此外 generated 的物理计划固定为 persistent grid-stride，上游一行一个 program。先决定并对齐作者想要的精度与 ABI，再单独判断 row mapping；当前比值不作为纯 emitter 结论。新增的 `I.sigmoid` 已直接落成 `intent.unary {operator="sigmoid"}`，Triton/TileLang 分别映射已有原语，cuTile 做等价逐元素展开，因此“缺 sigmoid”已不是原因。

### baseline 或 scope 不可比

- **weighted RMSNorm / dual GEMM**：generated 融合了 baseline 分开的权重乘法或双 GEMM epilogue；优势属于 end-to-end fusion。
- **online softmax**：DSL 是真正的跨 tile recurrence；Triton/cuTile baseline 是 full-row stable softmax，TileLang baseline 在当前配置退化成一次迭代。
- **MoE / grouped GEMM**：三家 upstream 的 route preparation、list/packed ABI、gather、merge 与 kernel 数量均不同。表保留 E2E 观察，但不据此宣称 device kernel 优劣。
- **vector-bias attention**：上游数值路径误差明显更大；即使都过当前 tolerance，也不能把 `0.728×` 直接解释为同语义 kernel 加速。

## 快项归因

- cuTile GEMM/BF16 GEMM 和 TN BMM 的改善来自把合法轴、合法 tile 与资源边界交给当前设备的下层 `exhaustive_search`，而不是加 kernel 分类。
- TileLang GEMM/BF16 GEMM 相对固定 upstream 配置更快，属于当前形状和当前设备重新选择配置。
- TileLang weighted RMSNorm、三个 dual GEMM、MoE 和 grouped GEMM 的较大优势主要是 fusion 或 packed ABI 的 E2E 优势，已经只记在 `E` 列。
- 跨 provider 取最优仍有实际价值，例如 dense attention 的最佳 generated 来自 TileLang，而最佳 upstream 来自 cuTile；但跨 provider 最优不能掩盖同 provider 的 compiler gap。

## 按提问顺序重审缺口

| 场景 | 作者是否已写下 | 下层是否已有 | 当前判断 |
|---|---|---|---|
| RoPE 的 `i + offset` | 是；frontend 已生成普通 `intent.binary add`，访问持有该 SSA operand | 三个表面都能直接打印加法 | 不建 quasi-affine IR；先停止丢弃 SSA/use-def。 |
| GQA 的 `q_head // group` | 是；`FloorDiv` 已生成 `intent.binary floor_divide` | 三个表面都能打印整数除法 | 同上；many-to-one provenance 从现有 SSA 向源轴回溯。 |
| CE 的 `label[row]` 索引 | 是；标签是一次 scalar `intent.view_load`，再成为 logits 的 `value_index` | 下层能做动态 scalar address | 不做区间推理；label 范围是调用前置条件，ignore sentinel 是作者显式谓词。 |
| EQ/NE/LT/LE/GT/GE | 是；frontend 已定义并 lower 六种 compare | 三个下层都支持 | GPU analysis 与 spelling table 只接 GE，是有限映射丢失，不是新机制。 |
| 两个 ordered stream | 是；现有 online softmax 已有两个顺序 stream | 三后端均运行 | 从缺口清单删除。 |
| LayerNorm backward 分组 partial | 否；当前作者明确写的是 per-row partial | 下层不会替作者改跨 kernel workspace | 改 DSL 算法，不由 realizer 猜。 |
| cuTile BMM persistent traversal | 作者只写 contraction 与 layout，不写物理遍历 | cuTile upstream 已展示成熟实现 | 这是已证实的 physical realization gap。 |
| TileLang varlen bulk transfer | 作者写了 ragged validity 与 causal stop | TileLang 有 bulk `T.copy`，但需要我们证明投机读取安全 | 补共享合法性事实与机械投影，不加 attention 分支。 |
| 多个 ragged relation 位于不同轴 | 当前没有一个 source 同时制造 | 容器已允许多个 relation | 语料覆盖空白，不先造机制。 |
| 多个 ragged relation 位于同一轴 | 当前没有作者场景 | emitter 仍有 unique-relation 限制 | 记录为潜在限制；等真实 source 给出语义再拆。 |
| ordered ragged + indirect member map | 当前 varlen source 是连续 offsets | emitter 明确拒绝该组合 | 潜在 capability gap，不冒充当前 blocker。 |
| ragged staging + ordered stream | 当前没有一个 source 同时制造 | 数据结构不排斥，运行未证明 | 语料覆盖空白。 |
| transpose + K tail + mixed-rank epilogue + int8 store | 作者分别写过这些部件 | 三后端分别跑通单项 | 组合覆盖空白，不是已证实机制缺口。 |

清单因此明显变短：不再需要一个独立 quasi-affine schema，不再把两个 ordered stream 当缺口，也不为尚未出现的组合先建机制。

## 索引 use-def 的精确断点

当前链路是：

1. `python/intent/frontend/lowering/ast/expressions.py` 已把 `+`、`//`、`%` 生成普通 `intent.binary`；`python/intent/frontend/lowering/ast/indexing.py:204-224` 把最终 SSA 值作为 `VALUE_INDEX` operand 保存。作者的信息没有在 frontend 丢失。
2. `lib/Target/Common/Realization/KernelFacts.cpp:309-319` 对 tensor pointwise 结果记录轴，但 scalar pointwise 结果直接返回 success；所以 `i + offset` 或 `q_head // group` 没有进入 `facts.valueAxes`。
3. `recordLoadAxes` 与 `inferIndexedAxes` 只查 `facts.valueAxes`；`resolveDomain`（同文件 `860-888`）只认直接 domain/parallel/stream 值，不沿 scalar SSA 回溯。
4. 三个 emitter 在动态 `value_index` 上调用 `resolveAxis`，随后打印该物理轴的 base index。以 Triton 为例，`lib/Target/Triton/Emission/Source/Emitter.cpp:1568-1615` 没有使用已经发射出的 binary SSA 值，因此把 `i + offset` 实际打印回 `i`。

因此对“use-def 是否够”的回答是：

- **够确定 ownership 和源 logical axis**：沿 SSA operands 回溯到唯一循环变量即可。
- **够保留精确地址**：地址应使用已经发射的 SSA expression，而不是从 axis 重新合成。
- **不自动等于相同 validity**：`i` 与 `i + offset` 共享源轴，但后者可能越界。若作者的逻辑 domain/precondition 已保证 paired access 有效，应保留该陈述；否则 mask 必须比较精确地址与 extent。不能仅凭“用了同一循环变量”宣称二者边界相同。

这解释了为什么当前不应先做数值区间系统：真正丢失的是已有 use-def 和作者写下的表达式。只有在恢复这两者后，仍有某个真实 kernel 必须证明而作者与下层都没有表达的范围事实，才轮到新增分析。

CE 也不要求这套数值推理。`label[row]` 是 tensor-derived scalar，编译器无法从 use-def 证明其值域；默认语义应是调用方满足 `0 <= label < vocab`，而 `ignore_index` 由作者显式比较。runtime clamp 会改变错误语义，不采用。

## 本轮验证

- 测量口径、baseline 重接与通用 tuner 候选修改后，现有 19 个 repro × 3 个 provider 共 57 个顶层调用全部通过真实 `DSL → Kernel MLIR → Physical Plan → provider source → GPU` 数值执行；展开 BMM layout、varlen causal/noncausal 与 tail case 后形成上表 25 行。
- extended 45 个入口统一到 100 次采样后全部通过；随后只删除了 LayerNorm backward 不合法的 upstream wrapper 计时，并以该入口专用复跑确认最终 `generated=0.3601 ms, upstream=unavailable`，没有改变其生成或数值路径。
- `I.sigmoid` 加入后，现有 `swiglu_backward` 在 Triton、cuTile、TileLang 三条 repro 单独复跑，三者均完成生成、真实执行和数值对照；Triton 生成源码已直接出现 `tl.sigmoid`，TileLang 出现 `T.sigmoid`，cuTile 保持等价展开。
- 未建立 test 目录、fixture 或额外验证设施；验证仍只有现有 `examples/run/repro.sh <provider> <kernel>` 手动入口。

## 当前实际卡点

当前会直接阻塞现有 source 或已被性能证据证明的问题只有：

1. scalar index use-def/provenance 与精确 SSA 地址在 realization/emission 中被丢弃；它阻塞 RoPE、GQA 以及后续按 `row % group` 重写 LayerNorm backward。
2. cuTile BMM NN/NT/TT 的 persistent traversal 与转置 load 兑现仍未达到当前 upstream。
3. TileLang causal varlen attention 缺少可复用的“无效 lane 在影响可观察结果前被消除”证明，导致本可 bulk copy 的 transfer 被逐元素展开。
4. LayerNorm backward 的 DSL multi-kernel 编排仍是 per-row partial 算法；在改源码前没有公平 upstream 性能数字。

其余条目要么已经在作者源码或下层存在，只是当前链路丢失；要么只是尚未被语料动态制造的组合，不应提前扩成新机制。
