# 用 Triton 逼出真实 GPU 编译能力：实现、A/B 与决策空间审计

## 1. 本轮结论

本轮没有把“超过 1.05”简单等同于“再加一个 target form”。实际结果分成三类：

1. **真实补出的能力**
   - Core 增加了 typed `join`，Mamba3 可以直接表达两半配对后的交错值，不再用四条散写模拟；
   - Physical Program 的 contraction flow 现在记录“真正送进 contraction 的 accumulator SSA 值”，Triton 可以把 `alpha * previous + dot(...)` 机械投影为原生 `tl.dot(..., acc)`；
   - Triton 的 `program_m/program_n` 数值搜索空间补进了此前完全缺失的窄行/整行以及 128×128 候选，两个 embedding entry 在两台机器上都回到 1.05 内。
2. **实测否决的猜测**
   - 放宽 persistent 并不能修 ragged grouped contraction；强制 persistent 反而更慢；
   - H100 Flash Attention 的差距不是缺 source winner 的 tile/warp/stage、不是缺 TMA、不是缺 `tl.dot(..., acc)`、不是 shape specialization，也不是 descriptor rank；这些替代形态都实际跑过；
   - BatchNorm 只改变同一个 `reduce` 内部的合法树形不能追平 source。要得到 source 的 lane-array 算法，必须把归约移过作者写下的 ordered state-stream 边界，那已经不是 physical reduction-tree 选择。
3. **自查发现的一处静默错误**
   - shared contraction flow 扩展后，TileLang 仍把“缩放后的 previous”错当成“原始 previous”原地累加，生成代码能跑但数值错误。现在只有 `accumulator_input_value == accumulator_value` 时才允许原地复用；缩放 carry 走原有显式路径。

两张 Triton 固定表已经完整重跑并更新：

- [triton-5090.csv](baseline-new/triton-5090.csv)：54 行，51 pass，51 个可比项中 46 个在 1.05 内；
- [triton-h100.csv](baseline-new/triton-h100.csv)：54 行，52 pass，52 个可比项中 45 个在 1.05 内。

没有新增 status 回归。剩余超过 1.05 的格子都在第 6 节逐项登记；其中一部分是本轮新 A/B 关闭的算法/数值合同差异，一部分是此前已用跨设备反转或微秒级绝对差确认不应建立共享规则的项。H100 Flash Attention 仍是一个**没有追平、但已经把常见解释逐项排除掉的 provider/source 结构残差**；报告不把它冒充成“已经解决”。

---

## 2. Core `join`：语言层补的是值语义，不是 Mamba 特判

### 2.1 为什么属于 Core

上游 Mamba3 把两个 half tensor 先组成一个新的 minor extent-two 轴，再 reshape 成 interleaved 最后一维。这是值的组成方式；删掉它会改变结果下标，与 tile、warp、storage 无关。

参考实现核验结果：

- Triton 的 `tl.join(a, b)` 先广播两侧，再追加一个 extent-two minor 轴；
- `tl.interleave(a, b)` 本身就是 `join + reshape` 的语法糖；
- TileLang 当前没有同级原生 join primitive，不能据此把 target 拼写抬成 shared 机制，也不能宣称该 target 已支持。

实现沿唯一链路闭合：

- frontend canonical op：[tensor.py](../python/intent/frontend/lowering/intrinsics/tensor.py)；
- Kernel IR / Physical executable op：[IntentOps.td](../include/Intent/Dialect/Intent/IR/IntentOps.td)、[PlanOps.td](../include/Intent/Dialect/Plan/IR/PlanOps.td)；
- 轴 provenance 与 padding/validity 传播：[KernelFacts.cpp](../lib/Target/Common/Realization/KernelFacts.cpp)、[Proofs.cpp](../lib/Target/Common/Realization/Proofs.cpp)；
- Triton native spelling：[Operations.cpp](../lib/Target/Triton/Lowering/Handlers/Operations.cpp)。

frontend 允许广播，但 canonical `intent.join` 的两个 operand 在 emit 前已经被显式 broadcast 到同一类型；GPU verifier 要求 canonical operands 类型相等。因此这里是一份语义、不是 frontend 与 Core 的两份冲突合同。

### 2.2 两个独立使用点

[mamba.py](../examples/kernels/streaming/mamba.py) 中不是只改了触发问题的 forward：

- `mamba3_siso_forward`：query/key 两半都经 `join + reshape` 后整体写回；
- `mamba3_siso_step`：key state 和二维 SSM state 也经同一 Core op 形成完整值。

这证明新增能力不是只够一个 entry 使用的形状 matcher。

### 2.3 实测

| entry | 设备 | 旧 fixed ratio | 新 full-run ratio | 结果 |
|---|---:|---:|---:|---|
| `mamba3_siso_forward` | 5090 | 1.671851 | **0.688989** | 四次 half scatter 收敛为两次完整值写回后追平 |
| `mamba3_siso_forward` | H100 | 1.952458 | **1.109586** | 大幅收窄，余量见下文 |
| `mamba3_siso_step` | 5090 | 约 1.04 | 1.059536 | 绝对差 5.664 us |
| `mamba3_siso_step` | H100 | 1.065214 | 1.052613 | 绝对差 4.784 us |

forward 的 H100 余量做了 exact-trig A/B。上游文件明确写着它用 PTX `sin.approx/cos.approx` 以精度换速度，Intent DSL 使用跨 target 一致的 `I.sin/I.cos`：

- approximate source：约 `0.176432 / 0.159936 ms = 1.103141x`；
- 把 source 临时改成同样的 exact `tl.sin/tl.cos`：`0.171136 / 0.178304 ms = 0.959799x`。

临时修改已经删除，vendored source 未改。这个 A/B 说明 forward 的 H100 余量来自数值算法合同，而不是继续缺 interleave form。step 的 exact-trig A/B 仍约 `1.061x`，因此其约 5 us 余量不归因于 trig；它保留为微秒级参数/运行差异，不建立结构规则。

---

## 3. 精确 accumulator flow：作者表达式不再在 shared → provider 途中丢失

### 3.1 原问题

Flash Attention 的在线归一化写的是：

```text
scaled_previous = alpha * previous
next = scaled_previous + contract(probability, value)
```

旧 shared analysis 只在 add 的另一侧**直接等于** `previous` 时记录 loop-carried flow。因此这段普通 SSA pointwise 链在 KIR 中完好，但进入 Plan 时被压成 `accumulator_flow=none`；provider 只能发射独立 multiply、dot、add。

### 3.2 修复边界

[ContractionRealization.cpp](../lib/Target/GPU/Transforms/Value/ContractionRealization.cpp) 现在沿 add 的非-contract operand 做 use-def 回溯，并把两个不同事实分别记录：

- `accumulator_value`：上一轮原始 carry；
- `accumulator_input_value`：作者真正送进 contraction-add 的 SSA 值。

Plan verifier 要求 loop-carried flow 的 owner、update、previous value、input value 同时存在；provider 不再从邻近 pointwise 结构重推。

Triton provider pass 只在以下条件同时成立时选择 native accumulator form：

- flow 是 loop-carried；
- contract form 是 `direct` 或 `deferred_one`；
- input SSA 定义支配 contract；
- update 是被 Plan 精确绑定的 add。

terminal translation 才机械发出 `tl.dot(lhs, rhs, accumulator_input)`，并把原 add 绑定成 contract result alias。Mamba 中 accumulator input 定义在 contract 之后，dominance 不成立，因此不会被错误融合。

### 3.3 这项能力不是靠性能结果证明的

生成 Flash source 现在确实出现 `tl.dot(p, v, scaled_previous)`，但 H100 延迟没有因此追平：

- 5090：约 `2.689 / 2.756 ms = 0.976x`；
- H100：约 `1.963 / 1.740 ms = 1.129x`。

所以这项改动的交付是“Plan 不再丢作者 SSA 值、Triton 能委托原生 accumulator”，不是虚报为一次性能优化。

### 3.4 shared 改动暴露出的 TileLang 静默错误

TileLang 旧路径只看 `accumulator_flow=loop_carried`，直接把原始 previous fragment 作为 `T.gemm` 输出并原地累加。flow 扩展后，它忽略 `accumulator_input_value`，从而把 `alpha * previous` 错发成 `previous`；`dense_flash_attention` 最大误差为 3.5375。

修复位于 [TileLang Operations.cpp](../lib/Target/TileLang/Lowering/Handlers/Operations.cpp)：原地 accumulator 与 conditional-result buffer 复用都额外要求 input value ID 与 previous value ID 完全相同。修后：

- TileLang `dense_flash_attention` 数值恢复，`6.705328 / 3.732016 ms`；
- TileLang `dense_gemm`：`2.041280 / 2.313520 ms`，无退化；
- cuTile `official_fmha`：`5.004816 / 30.283760 ms`，无退化；
- cuTile `dense_gemm`：`2.034656 / 2.432000 ms`，无退化。

这里的经验很具体：新增一个 Plan 字段不等于 leaf 自动变正确；旧 leaf 若把宽泛 flow kind 当成充分条件，就会在新组合下暴露静默错误。

---

## 4. Embedding：补的是参数搜索盲区，不把它包装成 shared pass 创新

两类 embedding 的 generated kernel 都已经有正确 program decomposition，问题是 `program_m/program_n` profile 从未包含上游常用的 128×128 和整行 1×4096 数值。新增候选位于 [triton.py](../python/intent/runtime/tuning/triton.py)，只按 physical role 参与 autotune，不按 kernel 名或设备型号触发。

| entry | 5090 旧 → 新 | H100 旧 → 新 |
|---|---:|---:|
| `embedding_lookup` | 1.090993 → **0.996324** | 1.057418 → **1.001427** |
| `flaggems_embedding_lookup` | 1.078670 → **0.985556** | 0.995802 → **0.946891** |

非触发 entry 的定向结果：

- `index_select`：0.954081x；
- `flaggems_transpose_copy`：0.328486x；
- full-run dense GEMM：5090 0.997508x，H100 0.980766x。

因此候选没有造成已观察到的性能退化。但这仍然只是“补全 parameter space”，不是 shared compiler 学会了新的结构决策；报告不把它算成 pass policy 的性能创新。

---

## 5. Shared refinement pass 的真实决策空间

本节按“同一输入是否真有多个可选 physical program”审计，而不是按有没有独立 pass 类来打勾。

| pass | 当前真实空间 | 静默排除/固定项 | 本轮证据 |
|---|---|---|---|
| AutomaticBlocking | 根据 parallel/partition/ordered/scan/contract/ragged facts 选择 ownership、traversal、reduction、lane role；不同输入会走不同 typed 分支 | 给定同一组 facts 只有一个 canonical skeleton；conservative 模式会把非 source-visible range 全压成 `one` | 不是常量 pass，但没有“同一结构多个 blocking skeleton”的搜索 |
| AccessRanges | `ownership → traversal → reduction → lane` 的唯一优先级投影 | 找不到 selected range 直接失败，不搜索替代覆盖 | correctness projection，不是性能候选空间 |
| TransferRealization | load/returned atomic 的 result/coverage space 分支 | `materialization` 永远是 `direct` | 形式上是 pass，结构候选目前只有一个；这是明确的“假空间” |
| ContractionRealization | `scaled_direct/replay/direct/deferred_one/deferred_two`，shared/private operand residency，direct/deferred transfer | replay 条件不成立就排除；scaled+replay 明确拒绝 | 是当前最实质的 shared value/operation decision space；本轮新增 scaled carry use-def flow |
| ScanRealization | fragment access 与 workspace scalar access 二分 | inclusive only；构造初值固定 scalar carry/fragment result | 有真实二分，但 scan 算法/树形覆盖仍窄 |
| ValueRealization | tensor/scalar pointwise 二分 | reduction 固定 fragment、scan carry 固定 scalar、sparse spaces 固定 | 多数路径是常量填充，不能把“有 pass”误当成已经选择了 value realization |
| PrivateBufferResidency | scalar array / vector / workspace，读取 size、bit width、dynamic access、device register capacity | 给定 facts 后是一次贪心决定，不交 tuner | 有真实跨输入分支，但没有 placement 候选 A/B |
| PersistentTraversal | persistent true/false；true 后把 program axes 折成固定 worker traversal | 硬条件 `!ragged && parallel>=3 && multiTile>=2`；ragged 和较低维结构完全不进入空间 | 见下方实测；空间确实过窄，但简单放开不是正确答案 |
| BoundaryNeutralization | proof 成功/失败的二值 correctness decision | 无法证明就保留 padding；不猜 | 唯一合法解，不应该扩成性能搜索 |
| SearchSpace | 把非 fixed/one/row-vector 的 range role 暴露给 provider tuner | shared IR 只记录参数名；候选值在 Python profile；`num_stages` 没有由 shared resource fact 收窄 | 当前是 names-only parameter surface，不是 structural autotune |

### 5.1 Persistent 的实际扰动

做了两类 A/B，临时修改均已删除：

1. Flash Attention 把 `multiTile>=2` 放宽到 `>=1`：
   - 5090 基本持平；
   - H100 约 `2.117 / 1.766 ms = 1.199x`，比非 persistent 更差。
2. ragged grouped GEMM：
   - 仅删除 `!ragged` 仍不触发，说明 ragged 还在 parallel/multi-tile 统计阶段被排除；
   - 强制所有 contraction persistent 后，5090 cuTile 从约 `1.591 ms` 退到 `2.795 ms`。

因此两个结论同时成立：

- 当前 pass 的确静默排除了一类结构，审计不能再声称它覆盖了 ragged contraction；
- “把 persistent 打开”并不是 grouped GEMM 2.5×/5.6× 差距的答案，不能据此永久放宽 shared policy。

### 5.2 仍然像玩具的地方

这轮找到了三类不能靠形式完整性掩盖的问题：

- `TransferRealization` 的 materialization 只有 `direct`；
- `ValueRealization` 对 reduction/scan/sparse 的多数 placement 仍是常量；
- SearchSpace 只暴露 role 名，合法数值和资源组合主要靠 provider profile/JIT 淘汰，shared pass 没有形成资源约束过的参数域。

它们不是本轮为了“显得有产出”而随手扩写的对象。没有真实 A/B 证明某个新候选更好时，本轮保留现状并在这里明确登记。

另一个架构事实是：orientation、reduction axis、ragged route 已经由 common query 统一推导，三家 provider 并不是各自实现三套算法；但三家仍各自缓存同义 provider attr，部分 materializer 仍重复解析 index relation。这是 leaf 仍不够机械的剩余边界，本轮没有把与 Triton 性能无直接证据的组织清理混入实现。

---

## 6. 最终所有 `ratio > 1.05` 格子的状态

### 6.1 5090

| entry | full-run ratio | 最终定性与实测依据 |
|---|---:|---|
| `padded_rope_cache_update` | 1.291267 | 此前同机交替 A/B 已把旧表 1.98x 识别为离群；稳定余量约 1.23–1.29x。source 先选地址再执行一份 effect body，DSL 明确写三路控制流；已有候选与生成前后 A/B 没有稳定通用修法，本轮按任务边界不重追 |
| `scaled_fp8_splitk_gemm` | 1.706573 | H100 同一项 0.926776x，winner 跨设备反转；不能加设备分支或共享常量 |
| `mamba3_siso_step` | 1.059536 | 绝对差 5.664 us；join 已用于两个 state 输出；exact-trig A/B 仍约 1.061x，没有结构证据支持新增规则 |
| `mamba_chunk_state` | 1.341202 | 绝对差 5.088 us，H100 仅 1.024758x；属于短核跨设备幅度差，不建立一台机器专用规则 |
| `flaggems_batch_norm_training` | 1.342389 | 见 6.3：合法 reduction-tree A/B 无收益；source 与 DSL 的 ordered state decomposition 不同 |

### 6.2 H100

| entry | full-run ratio | 最终定性与实测依据 |
|---|---:|---|
| `flash_attention_forward` | 1.107355 | 没有追平；见 6.4，常见 shared/provider 解释已逐项 A/B 排除 |
| `padded_rope_cache_update` | 1.211838 | 同 5090；稳定 control/effect decomposition 差异，不用设备特化掩盖 |
| `mamba3_siso_step` | 1.052613 | 绝对差 4.784 us；exact-trig A/B 不解释该差距 |
| `mamba3_siso_forward` | 1.109586 | 上游 approximate trig 对比 exact trig 后反转为 generated 更快，属于数值算法合同差异 |
| `flaggems_triangular_solve` | 1.089301 | 绝对差 3.392 us，5090 generated 为 0.833333x；跨设备 winner 反转 |
| `flaggems_batch_norm_training` | 1.202976 | 同 5090；合法 tree 变化不追平 |
| `flaggems_softmax_backward` | 1.104171 | 绝对差 8.752 us，5090 generated 为 0.791867x；跨设备 winner 反转 |

### 6.3 BatchNorm：为什么本轮不再把它误写成“只差一条 physical tree”

当前 DSL 在每个 S chunk 内对 B×S-region 做 typed Welford reduce，再把 chunk scalar 通过 ordered state-stream 合并。source 则让 B×S physical lane array 跨 chunk 保持状态，最后才把 lane partials reduce 成 scalar。

本轮只试了不越过作者结构边界的替代：

- 把 all-axis tuple reduce flatten 成一维：5090 约 1.321x，H100 约 1.199x，没有实质关闭；
- 把同一个 reduce 按 axis 1、axis 0 分两级：5090 `1.446x`、H100 `1.308x`，更慢。

这两个结果否决了“只是 `tl.reduce(axis=None)` 拼写不佳”的解释。要生成 source 的 lane-array 形态，必须延后 DSL 中已经发生的 reduce、把 tensor partial 穿过 state-stream，再在循环外归约；这改变了作者写下的有序 carry 结构。它不能由 compiler 在 Physical Program 中偷偷完成。

因此这里要修的是 **baseline 算法对齐或作者 DSL 表达**，不是继续在 leaf 加 reduction spelling。当前表保留数字以暴露差异，但该 ratio 不能再被写成同算法下的纯 compiler 性能缺口。

### 6.4 H100 Flash：逐项排除，而不是一句“下层质量”

当前 generated 与 source 都已经具备：causal prefix/boundary 两段、descriptor/TMA、128 query tile、native QK/PV dot、fused accumulator。继续做了这些新 A/B：

| 替代形态 | H100 结果 | 判断 |
|---|---:|---|
| 当前最终形态 | `1.963136 / 1.772816 = 1.107355x` | 固定表结果 |
| source 实际 winner `M128×N64, 4 warps, 2 stages` 原样加入 generated tuner | `2.116960 / 1.747760 = 1.211242x` | 不是缺这个参数候选 |
| 所有 logical dimensions 改成 `tl.constexpr` | `2.046464 / 1.743584 = 1.173711x` | 不是缺 shape specialization |
| 保留 full-rank strided descriptor，匹配 source 的 4D descriptor 而不 flatten | `2.193360 / 1.751680 = 1.252146x` | 2D flatten 不是根因，full-rank 更慢 |
| 放宽 persistent | 约 1.199x | persistent 不是答案 |
| 原生 `tl.dot(..., accumulator)` | 约 1.129x | 语义/IR 更干净，但不关性能差距 |

此前还测过 `tl.range`、额外 stage/warp、TMA on/off、tile 组合，均无稳定可保留收益。

所以本轮能诚实给出的结论不是“下层质量”，而是：**H100 仍有 10.7% 的 provider/source program-shape 残差，已排除 descriptor rank、shape specialization、source winner 参数、persistent、accumulator primitive 和 loop spelling这些具体维度，但尚未定位到更小的结构事实。** 这一格没有被伪装成关闭，也没有为了数字引入 H100 分支。

---

## 7. 完整回归与失败边界

### 7.1 Triton full-run

两台机器同时启动完整 54-entry runner；每个 pass 行都完成 generated/source 数值比较，再记录 CUDA graph p50。

5090 非 pass：

- `modern_flash_attention_forward`：source/adapter 准备阶段需要 163840 B shared memory，超过设备 101376 B；
- `flash_attention_backward`：同一 source 资源边界；
- `legacy_flash_attention_bias`：vendored source 在 Triton 3.6 下数值不正确，保留 `source_compatibility_gap`。

H100 非 pass：

- `flash_attention_backward`：source provider 首次 JIT/launch 失败；
- `legacy_flash_attention_bias`：同一 source compatibility gap。

这些状态没有因本轮 compiler 改动恶化。

### 7.2 代表性跨 provider 回归

shared Plan 变更后额外跑了：

```bash
./examples/run/baseline-v2.sh cutile /tmp/cross-provider-cutile-5090.csv \
  official_fmha dense_gemm
./examples/run/baseline-v2.sh tilelang /tmp/cross-provider-tilelang-5090.csv \
  dense_flash_attention dense_gemm
```

第一次 TileLang attention 数值失败正是本轮发现并修复的 accumulator contract bug；修后四个 entry 均数值通过。没有把失败归为 target 波动。

---

## 8. 冻结前应保留的真实判断

1. **本轮真正扩大的 shared/language 能力**只有两项：typed join，以及精确到 SSA input 的 loop-carried contraction flow。embedding 是参数空间补全，不应混写。
2. **当前 pass pipeline 不是全假，也远未形成完整 compiler space。** Contraction、private residency、boundary proof 有真实跨输入分支；Transfer/Value 多数位置仍是常量，SearchSpace 仍是 names-only。
3. **persistent 的 ragged 排除是真问题，但不是当前 grouped GEMM 性能答案。** 只有找到另一种实测更好的 ragged ownership/traversal 结构，才能改 policy。
4. **provider leaf 仍可能把 shared fact 读窄。** TileLang accumulator 静默错误就是直接证据；“字段已进 Plan”不能代替跨 target 组合验证。
5. **不能把所有剩余差距都称为 compiler 缺口。** Mamba forward 是数值路径差异，BatchNorm 是 ordered state decomposition 差异，若继续公平比较应先在作者 DSL 层把算法对齐。
6. **也不能把 H100 Flash 称为已解释。** 它经过多维 A/B 后仍有 10.7%，当前最准确状态是“未定位的 provider/source 结构残差”，不是“下层就这样”。
