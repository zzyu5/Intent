# 5090 回归修复与 H100 全量复验报告

## 报告边界

本报告记录从 `deaa410` 到 `8e81210` 针对 5090 全量结果的修复，以及在当前代码基线上继续完成的 H100 全量复验。

5090 修复阶段只运行受影响算子的定向复现，没有重跑 5090 全量。H100 恢复空闲后，在提交 `6d4b2c6` 的独立临时快照上运行了完整的 104 个 runner × 3 个 provider；其中 6 个 runner 展开多个 case，最终形成 113 行、339 个 provider cell。验证入口始终只有：

```bash
examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

报告覆盖六个提交：

```text
21ce9f1 fix(frontend): import arg-reduction index type
79f4fc2 fix(tilelang): preserve fragment reshape semantics
5b54fef fix(plan): preserve transfer validity axes
3cf201b fix(tuning): isolate specialized query candidates
238c3b3 fix(emission): consume staged axis decisions
8e81210 data(baseline): record repaired 5090 cells
```

5090 固定表中的上游 source 数字没有重测、没有移动，只写回定向运行得到的 generated 状态和延迟。H100 表则由本次完整运行重新生成，generated 与能够公平调用的 upstream 都来自同一 H100 代码和环境基线。

## 本轮结论

全量暴露的两处数值错误已经闭合：

- TileLang MoE 最大误差从约 `8.7` 降到 `9.0594e-06`。
- cuTile Mamba chunk scan 最大误差从约 `0.148` 降到 `6.1035e-05`。

另外闭合了三类问题：

- arg-reduction 前端缺少 `i32` 导入导致的六个 provider failure；
- block-sparse attention 定向运行与全量结果不一致；
- staged/ragged 发射器重新拼装物理 tile，而不是读取 Physical Plan 已有决定。

修复没有增加按 kernel 名字或 kernel 类别分支。算法 IR 没有被改写；改动集中在前端名字绑定、Physical Plan 事实保存以及 target leaf 对这些事实的机械投影。

## 一、TileLang MoE 数值错误

### 现象

同一份 Kernel IR 和 Physical Plan 下：

- Triton 数值正确；
- cuTile 数值正确；
- TileLang 最大误差约为 `8.7`。

这排除了 DSL 算法整体错误，问题位于 TileLang 对局部张量形状和 fragment layout 的投影。

### 二分结果

- `8fbfc6d`：通过，最大误差约 `9.059e-06`；
- `6f40b3d`：失败，最大误差约 `8.33`。

回归点把原本显式逐元素复制的 fragment 扩维改成了：

```python
T.reshape(fragment, (M, 1))
```

形状元素数相同不代表 TileLang fragment 的物理 layout 可以直接 reshape。该写法能够编译和运行，但读到了错误的物理对应关系，因此形成静默数值错误。

### 修复

`SourceEmitter::emitReshape` 现在区分两类情况：

- rank-1 到 scalar：只有计划投影出的物理 extent 确实为 singleton 时，才发射 `[0]`；
- fragment 扩维：不再使用不安全的原生 `T.reshape` 快路径，保持显式逐元素复制语义。

这里没有为 MoE 建特例。修复针对的是 TileLang fragment reshape 的一般语义边界。

### 当前结果

| Provider | 最大误差 | generated p50 / p95 | 状态 |
|---|---:|---:|---|
| Triton | `9.0594e-06` | `8.6755 / 8.6986 ms` | PASS |
| cuTile | `9.0594e-06` | `10.2673 / 10.3704 ms` | PASS |
| TileLang | `9.0594e-06` | `11.4188 / 11.4463 ms` | PASS |

## 二、cuTile Mamba chunk scan 数值错误

### 现象

cuTile 的三个候选都能编译和执行，但最大误差约为 `0.148`；Triton 和 TileLang 正确。

这不是候选合法性问题，而是某个候选暴露了原先被较小 tile 隐藏的边界写错。

### 二分结果

- `cff0e27`：最大误差 `6.1035e-05`；
- `e45b84e`：最大误差约 `0.158325`。

`e45b84e` 改变了候选构成，使 cuTile 选择 `TILE_SIZE_M=128`。这个变化本身没有改变算法，却让最后一个 chunk 中超出逻辑行数的 lane 出现。

生成代码当时使用：

```text
global_row = chunk * S + row
check_bounds = global_row < L
```

它只检查整个输出张量的全局长度。最后一个 chunk 的无效 lane 仍可能满足 `global_row < L`，于是写进下一个 chunk 的合法行。

### 根因

Kernel IR 已经携带值对应的逻辑轴来源，但 transfer lowering 只保留了普通 boundary，丢掉了：

- 哪个 tensor axis 受逻辑有效区间约束；
- 该约束来自哪个 domain node。

cuTile leaf 只能从外部 view shape 重新推断边界，无法恢复 chunk 内的逻辑 validity。

### 修复

Physical Plan 的 `TransferOp` 新增：

```text
validity_tensor_axes
validity_domain_nodes
```

Plan builder 从 transferred value 的逐轴来源生成这两组绑定。Plan verifier 检查：

- 两个数组长度一致；
- tensor axis 不重复；
- validity domain 必须属于该 transfer 的 boundary domain。

cuTile 的 scatter 和 unique-store 发射直接消费这些绑定，生成精确 mask。对于 staged ragged member，validity 直接投影成已有的 `member_mask`，不再走普通 `axisIndices` 路径重新构造。

### 当前结果

```text
cuTile Mamba chunk scan
generated/reference max error = 6.103515625e-05
p50 = 0.0205 ms
p95 = 0.0226 ms
status = PASS
```

## 三、arg-reduction 的前端名字错误

### 现象

以下六格在进入后端之前失败：

- cross entropy × Triton/cuTile/TileLang；
- sorted nucleus cutoff × Triton/cuTile/TileLang。

报错是结构化 intrinsic 中 `i32` 未定义。

### 根因与修复

`python/intent/frontend/lowering/intrinsics/structured.py` 只导入了 `i16`，但带下标归约仍使用 `i32` 保存位置。

修复只补齐：

```python
from intent.language.dtypes import i32
```

同时核查了该模块中的 dtype 引用，没有发现第二个引用未导入名字的路径。

### 当前结果

| Kernel | Triton p50 / p95 | cuTile p50 / p95 | TileLang p50 / p95 |
|---|---:|---:|---:|
| cross entropy | `0.5135 / 0.5165 ms` | `0.5109 / 0.5190 ms` | `0.5095 / 0.5127 ms` |
| sorted nucleus cutoff | `0.0200 / 0.0227 ms` | `0.0240 / 0.0267 ms` | `0.0259 / 0.0269 ms` |

六格数值均通过。

## 四、block-sparse attention 的结果不一致

### 结论

这不是自动调优随机性，也不是未初始化数据导致的不稳定。

在修复前连续重跑，TileLang 都稳定失败于：

```text
T.reshape/view shape check failed. 1, 32
```

在记录“定向三后端通过”的实际提交 `2626597` 上复现，TileLang 仍然稳定失败。因此此前的定向结论不准确，后续全量不是随机反转。

### 修复

问题与 MoE 共用同一处 reshape 语义缺口：输入类型带动态 extent，但 Physical Plan 已经把它实现为 singleton fragment。旧发射器只检查静态类型的维度 `1`，没有读取物理 extent。

现在 rank-1 到 scalar reshape 根据计划投影出的实际 extent 判断。物理 singleton 发射 `[0]`，其它情况明确诊断，不再生成模糊的 reshape。

### 当前结果

```text
TileLang block-sparse GQA decode
generated/reference max error = 0.05853271484375
p50 = 0.1020 ms
p95 = 0.1021 ms
status = PASS
```

修复后重复运行保持通过。

## 五、staged/ragged 发射中的信息丢失

### 找到的问题

Physical Plan 的 `StageAxisOp` 已经分别保存：

- member extent、tile、worker axis；
- feature extent、tile、worker axis；
- reduction extent、tile。

但三条 target leaf 仍在 staged contraction、grid、offset、load/store 和 atomic 路径里直接拼写：

```text
BLOCK_SIZE_M / BLOCK_SIZE_N / BLOCK_SIZE_K
TILE_SIZE_M / TILE_SIZE_N / TILE_SIZE_K
```

当前 planner 恰好把三种角色映射到这些名字，所以常见样本能运行；但 emitter 实际依赖的是名字约定，而不是 stage 自己的计划决定。出现多 stage、不同 tile role 或另一种 target spelling 时，这条路径会重新决定物理结构。

### 修复

三个 emitter 在 `prepareRaggedStages()` 中从 `StageAxisOp` 派生并缓存：

```text
stageMemberTiles
stageFeatureTiles
stageReductionTiles
```

这些只是 target spelling 的派生索引，不是新表示。以下路径改为统一消费它们：

- stage grid；
- member tile 数与 member offsets；
- feature offsets；
- staged contraction 的 fragment shape；
- reduction block 循环；
- staged gather/members；
- unique store 与 atomic merge。

当一个操作同时属于多个 stage，而对应 member tile 不一致时，TileLang 明确报错，不再随意取某个全局常量。

### 定向验证

MoE 三后端通过后，又运行了 grouped GEMM 的三种形态：

- base；
- member/K/N 尾块；
- empty groups。

九个 grouped-GEMM case/provider 组合全部数值通过。当前 generated 延迟为：

| Case | Triton p50 / p95 | cuTile p50 / p95 | TileLang p50 / p95 |
|---|---:|---:|---:|
| base | `1.4138 / 1.4251 ms` | `1.4582 / 1.4992 ms` | `1.2757 / 1.2868 ms` |
| tail | `1.4313 / 1.4375 ms` | `1.4731 / 1.5053 ms` | `1.2848 / 1.3003 ms` |
| empty groups | `0.0860 / 0.0900 ms` | `0.0707 / 0.0829 ms` | `0.0471 / 0.0546 ms` |

这组验证同时覆盖紧凑 ragged、尾块、空组、分阶段收缩、唯一写和原子写，不是只为 MoE 验证一条特化路径。

## 六、候选空间污染

### 问题

运行时用一个 role 的候选补齐其它缺省 role 时，`query`/`query_*` 的特殊候选 `1`、`2` 会进入普通 tile role。

这些值是单查询解码形态的特化合法值，不应成为普通 completion candidate。它们进入其它 role 后会改变资源使用、触发无效候选，或者把原本稳定的 kernel 推向低质量配置。

### 修复

Triton、cuTile、TileLang 的 tuning runtime 都增加 `_completion_candidates(role)`：

- 显式调优 query role 时，仍保留 `1`、`2`；
- 用 query 候选补齐其它 role 时，过滤 `1`、`2`；
- 其它 role 的候选集合不变。

它改变的是候选补齐规则，没有增加设备型号分支，也没有自建 cost model。

## 七、性能回退核查

### LayerNorm backward

Triton 连续定向运行得到：

```text
p50 = 0.0816 ms
p50 = 0.0805 ms
```

没有复现全量表中的 `0.1388 ms`。当前 CSV 记录 `0.0805 / 0.0867 ms`。这不是持续存在的代码回退。

### ordered prefix

Triton 连续定向运行得到：

```text
p50 = 0.0388 ms
p50 = 0.0397 ms
```

没有复现旧表中的 `0.0622 ms`。当前 CSV 记录 `0.0397 / 0.0432 ms`。

### W4A8 TileLang

当前代码约为：

```text
p50 = 0.1612 ms
p95 = 0.1620 ms
```

为了区分代码回归和环境变化，又在历史提交 `8502f5b` 上使用当前 TileLang 环境重跑，得到约 `0.1608 ms`。历史提交在当前环境也没有复现旧表的 `0.1163 ms`，因此现有证据指向下层版本、缓存或运行环境差异，而不是仓库代码回归。

本轮没有为了追逐旧数字加入 TileLang 专属物理策略。

### reshape-cache split variant

两次定向结果的 p50 约为 `0.0368 ms` 和 `0.0396 ms`，原写法约 `0.0297–0.0300 ms`。

绝对差值约 `0.007–0.011 ms`。当前没有证据把它归因到某个共享判定，也没有出现数值错误，因此本轮没有为微秒级差异增加机制或算子特判。

## 八、继续检查“从形状重建计划事实”

本轮实际收敛了两类：

1. transfer validity 轴从 Kernel IR 的值来源进入 Physical Plan，再由 cuTile scatter/unique-store 消费；
2. staged member/feature/reduction tile 从 `StageAxisOp` 进入三个 target leaf，不再使用全局 M/N/K 拼写代替。

另外检查了 persistent/worker-reuse 路径中的：

```text
n_rows
n_cols
BLOCK_SIZE / TILE_SIZE
```

这里没有直接删除。核查结果是：

- `n_rows`、`n_cols` 是计划中 program/lane 维度的 target ABI 别名；
- `BLOCK_SIZE`/`TILE_SIZE` 是已选 tile 在目标语言中的拼写，具体候选交给现有下层配置选择；
- 相关 leaf 没有从 tensor shape 重新决定算法结构。

因此这些代码虽然较厚，但仍属于必要的目标接口与语法映射。为了让 emitter 看起来更薄而删除它们，反而会把合法的 target ABI 投影误判成信息丢失。

## 九、5090 修复阶段的验证范围

5090 修复阶段运行了以下受影响 repro，没有重新运行 5090 全量：

```text
cross_entropy              Triton / cuTile / TileLang
sorted_nucleus_cutoff      Triton / cuTile / TileLang
moe                        Triton / cuTile / TileLang
grouped_gemm               Triton / cuTile / TileLang
mamba_chunk_scan           重点复验 cuTile，另外确认已有 Triton/TileLang 结果
block_sparse_attention     重复复验 TileLang
layer_norm_backward        重复复验 Triton
ordered_prefix             重复复验 Triton
variant_reshape_cache_split 复验 Triton
w4a8_packed                复验 TileLang，并在历史提交复验
```

## 十、H100 完整矩阵

### 执行基线

H100 使用独立 `/tmp` 快照，不修改远端原有脏工作树。环境为：

```text
GPU       NVIDIA H100 80GB HBM3
Triton    3.6.0
cuTile    1.5.0
TileLang  0.1.13
LLVM/MLIR 20
```

远端默认 `/usr/bin/nvcc` 是 CUDA 11.5，不认识 TileLang 为 H100 选择的 `sm_90a`。机器已安装 CUDA 12.2，因此 TileLang 全量使用：

```text
PATH=/usr/local/cuda-12.2/bin:$PATH
CUDA_HOME=/usr/local/cuda-12.2
```

修正前产生的 TileLang 环境失败全部作废；随后完整重跑 TileLang 104 个 runner，没有把两套环境的结果混在同一张表里。

### 状态总览

| Provider | PASS | UNSUPPORTED | COMPILE_TIMEOUT | FAILED |
|---|---:|---:|---:|---:|
| Triton | 111 | 1 | 1 | 0 |
| cuTile | 109 | 3 | 1 | 0 |
| TileLang | 102 | 10 | 0 | 1 |
| 合计 | 322 | 14 | 2 | 1 |

唯一真实 failed 是 TileLang `fp8_mqa_logits`：生成路径进入了 CUTLASS FP8 MMA，但当前 H100 架构路径在下层 assertion 失败。它没有被伪装成 pass，也没有在没有明确能力检查的情况下改写成另一种算法。

compile-timeout 为：

- Triton `token_sparse_mla_prefill`：外层 900 秒上限耗尽；
- cuTile `token_sparse_mla_prefill`：下层 TileIR 编译器触发自身 10 秒 compile timeout。

明确 unsupported 包括：

- Triton/cuTile 没有原生 2:4 sparse contraction 投影；
- cuTile 在 `sm_90` 不支持 block-scaled kernel 所需的 `float8_e8m0fnu`；
- cuTile 不接受 FP8 MQA 的 runtime-sized matrix-M lane；
- TileLang 不支持 compare-exchange、二维联合覆盖范围 cooperative transfer、single-row contraction、token-sparse batched GEMM 以及若干 paged/split-K 组合。

这些项都没有走慢几个数量级的伪支持路径。

### 5090 修复的跨设备复验

以下本轮重点路径在 H100 上全部通过：

```text
cross_entropy            3/3 provider PASS
sorted_nucleus_cutoff    3/3 provider PASS
moe                      3/3 provider PASS
grouped_gemm             3/3 provider，base/tail/empty_groups 全部 PASS
mamba_chunk_scan         3/3 provider PASS
block_sparse_attention   3/3 provider PASS
```

因此 i32 导入、TileLang fragment reshape、transfer validity 和 staged-axis 消费都不是 5090 专属修复。

### 共享内存容量 A/B

TileLang `absorbed_mla_prefill` 在 5090 上因为候选需要 `102400 B`、设备可用 `101376 B` 而失败；同一份当前代码在 H100 上自动通过：

```text
p50 = 0.1032 ms
p95 = 0.1043 ms
```

这个 A/B 给出了明确责任边界：候选是否装得下由 TileLang 下层编译/调优器根据当前设备容量筛选。共享内存容量不需要进入我们的算法结构决策，也没有理由增加架构型号分支。

### 三后端赢家分布

按每个 kernel/case 的 generated p50 最小值统计，只计 `status=pass`：

| 设备 | Triton 唯一赢家 | cuTile 唯一赢家 | TileLang 唯一赢家 | 并列 | 无赢家 |
|---|---:|---:|---:|---:|---:|
| H100 | 52 | 30 | 27 | 3 | 1 |
| 5090 | 38 | 27 | 37 | 10 | 1 |

两台设备有 56/113 行的赢家集合不同；只比较两边都有唯一赢家的行，也有 45 行发生变化。这直接证明同一份算法与共享物理模型能吃到三种 surface language 在不同机器上的不同强项，而不是某一个 provider 固定占优。

### 与上游主场对照

H100 上有 33 个 kernel/case 能同时取得 generated 最优值与同范围 upstream 最优值：

```text
generated 最优更快：19 行
upstream 最优更快： 14 行
```

按 provider-cell 统计，51 个可比格中 generated 更快 26 格、upstream 更快 25 格。上游回到数据中心卡主场之后，我们在三 provider 中取最优的优势仍然存在，但不再表现为单边压倒。

### 相对旧 H100 表的能力变化

旧表只有 106 行，本次新增 7 行：attention backward、block-sparse attention、causal Conv1D backward、paged MLA decode、paged split-K attention、sparse 2:4 GEMM、varlen GQA decode logits。

已有 106 行中状态变化只发生在 TileLang：

- `mla_head_projection` 两个 case：unsupported → pass；
- `absorbed_mla_prefill`：unsupported → pass；
- `paged_attention`、`continuous_gqa_decode`、`splitk_attention_reduce`：旧慢路径 pass → 当前明确 unsupported；
- `fp8_mqa_logits`：unsupported → failed，说明能力检查仍比实际 target projection 宽。

H100 完整数字见 `report/baseline/kernel-performance-h100.csv`。

## 十一、当前状态与未闭合观察

正确性方面，本轮列出的两处静默数值错误、六格前端失败和 block-sparse 结果不一致均已闭合。

尚未形成代码归因的只有两项性能观察：

- W4A8 TileLang 的旧数字无法在当前环境、当前代码和历史代码上复现；
- reshape-cache split variant 存在约几微秒差异，但没有稳定的共享层回归证据。

这两项没有被包装成编译器已修复问题，也没有为它们引入 provider 特判。

H100 新增一项明确的 target 闭合问题：TileLang `fp8_mqa_logits` 目前通过了上层能力检查，却在 CUTLASS H100 FP8 MMA 路径失败。当前表如实记为 `failed`，没有提前改写算法或伪装成 unsupported。

另外，TileLang `shifted_row_copy` 首次编译耗时 161 秒但最终数值与运行通过；这是编译成本观察，不是运行能力失败。

本轮结束时主工作树代码提交完整，二分定位生成的 detached 临时 worktree 已确认无修改后删除。
