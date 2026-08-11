# Intent Kernel 编译器：全量回归修复与 42 × 3 核验

日期：2026-08-11  
问题基线：`643691e`（108 PASS / 17 FAIL）  
实现基线：`028fa34`  
核验入口：`./examples/run/repro.sh <provider> <repro>`

## 结论

- 旧全量核验中的 17 个失败，修复后成为 **16 个数值 PASS + 1 个明确 N/S**。
- 按上一轮矩阵实际执行了 125 条命令：Triton 42、cuTile 42、TileLang 41。结果为 **124 PASS、0 个意外 FAIL、1 个 EXPECTED_UNSUPPORTED**。
- 若把三后端与 42 个 repro 的理论笛卡尔积全部列出，则共有 126 格：**124 PASS、2 N/S**。两个 N/S 都在 TileLang：`atomic_compare_exchange` 缺少下层 CAS primitive，`conv2d` 当前不能把多轴间接读取投影为一个并行 fragment。
- 所有 PASS 都重新经过 Python DSL → canonical Intent Kernel MLIR → GPU Physical Plan → 目标源码 → 下层 JIT → GPU 执行 → reference 数值比较，没有复用旧生成物冒充当前结果。
- 旧时已经 PASS 的组合未发现稳定性能退化。三个仓库内硬性能门槛全部通过；全量日志中的一个 55% 异常经三次复测确认是孤立抖动。
- 本轮没有新增 kernel、kernel 名字分支、ragged 专用 realizer，也没有为 TileLang 增加串行慢路径。

## 全量结果

状态含义：

- `PASS`：生成、JIT、GPU 执行和数值对照全部通过。
- `N/S`：目标表面没有可接受的等价投影，编译器明确拒绝；不是运行后数值失败。

| Repro | Triton | cuTile | TileLang |
|---|---|---|---|
| `softmax` | PASS | PASS | PASS |
| `layer_norm` | PASS | PASS | PASS |
| `layer_norm_backward` | PASS | PASS | PASS |
| `embedding_backward_atomic` | PASS | PASS | PASS |
| `atomic_compare_exchange` | PASS | PASS | N/S |
| `rms_norm` | PASS | PASS | PASS |
| `fused_add_rms_norm` | PASS | PASS | PASS |
| `dropout_residual_rms_norm` | PASS | PASS | PASS |
| `logsumexp` | PASS | PASS | PASS |
| `cross_entropy` | PASS | PASS | PASS |
| `gemm` | PASS | PASS | PASS |
| `bf16_gemm` | PASS | PASS | PASS |
| `batched_gemm` | PASS | PASS | PASS |
| `batched_row_affine` | PASS | PASS | PASS |
| `quantized_gemm` | PASS | PASS | PASS |
| `dual_gemm` | PASS | PASS | PASS |
| `weight_only_int4` | PASS | PASS | PASS |
| `conv1d` | PASS | PASS | PASS |
| `conv2d` | PASS | PASS | N/S |
| `selective_scan` | PASS | PASS | PASS |
| `attention` | PASS | PASS | PASS |
| `attention_bias` | PASS | PASS | PASS |
| `varlen_attention` | PASS | PASS | PASS |
| `varlen_gqa_prefill` | PASS | PASS | PASS |
| `varlen_gqa_rope_prefill` | PASS | PASS | PASS |
| `paged_attention` | PASS | PASS | PASS |
| `online_softmax` | PASS | PASS | PASS |
| `ordered_prefix` | PASS | PASS | PASS |
| `moe` | PASS | PASS | PASS |
| `grouped_gemm` | PASS | PASS | PASS |
| `swiglu_forward` | PASS | PASS | PASS |
| `swiglu_backward` | PASS | PASS | PASS |
| `shifted_row_copy` | PASS | PASS | PASS |
| `grouped_query_head_add` | PASS | PASS | PASS |
| `scalar_table_lookup` | PASS | PASS | PASS |
| `matrix_transpose` | PASS | PASS | PASS |
| `boolean_reduction` | PASS | PASS | PASS |
| `value_select` | PASS | PASS | PASS |
| `record_fields` | PASS | PASS | PASS |
| `scalar_while` | PASS | PASS | PASS |
| `sorted_nucleus_cutoff` | PASS | PASS | PASS |
| `insertion_top_k` | PASS | PASS | PASS |

汇总：

| Provider | 数值 PASS | N/S | 意外 FAIL |
|---|---:|---:|---:|
| Triton | 42 | 0 | 0 |
| cuTile | 42 | 0 | 0 |
| TileLang | 40 | 2 | 0 |
| 合计 | 124 | 2 | 0 |

上一轮实际运行的 125 格没有包括早已声明 N/S 的 TileLang CAS；本轮仍实际运行 TileLang `conv2d`，确认它稳定命中精确能力诊断，因此“125 条执行命令”的汇总是 124 PASS + 1 expected N/S。

## 根因与修复

### 1. MoE / grouped GEMM：分阶段收缩驻留合同

影响：两个 repro × 三个 provider，共 6 格。

#### 精确根因

`b3d60f6` 为普通收缩加入了按生产者链决定共享驻留的逻辑，但把“必须是直接 `view_load`”检查放在“是不是 staged contract”之前。分阶段 ragged 收缩的两侧并不对称：

- 激活侧可以经过按 route 选择的 `gather`；
- 权重侧是按 expert 索引的 rank-3 `view_load`；
- staged 物理实现要求两侧都成为该 stage 的共享操作数。

激活侧因为不是直接 `view_load` 被误判为 private fragment。随后三个 emitter 的通用延迟加载判定又要求收缩两侧都已经是 shared，导致权重侧虽然保留了 rank-3 expert 选择，也没有进入 deferred load 表。最后报出的“缺少 expert-selected rank-three weight”是后果，不是 rank 或选择元数据在 canonical IR 中丢失。

二分结果：

- `8c90fab` 的 Triton `grouped_gemm` 通过；包含新驻留判定后的点失败；首个坏提交为 `b3d60f6`。
- TileLang 额外的 `inconsistent operand spaces` 在二分旧点已经存在，说明它不是 TileLang 下层新回归，而是同一共享物理分类从未在该投影上闭合。

#### 修复

- staged contract 先按 stage 的物理合同把两侧分类为 shared，再处理普通收缩的直接 load / producer-chain 分类。
- 通用 transfer deferral 对 staged contract 直接承认该 stage 已确定的 shared residency；普通 contraction 仍要求双方 shared 且存在相应 program grouping。
- 三个 emitter 继续使用同一个 `deferSharedContractionTransfer`，没有增加 MoE、grouped GEMM 或 provider 分支。

结果：6 格全部数值通过。cuTile 之前附带的“操作数存储空间不一致”随共享 Plan 分类修正而消失，不需要 cuTile 专属补丁。

分类：Triton/cuTile 是 `b3d60f6` 引入的回归；TileLang 是此前未闭合的投影合同，由同一共享修复闭合。

### 2. Triton 卷积：固定小域的物理范围被逻辑范围覆盖

影响：Triton `conv1d`、`conv2d`，共 2 格。

#### 精确根因

二分得到 `8c90fab` 通过、`7f690c5` 失败，首个坏提交为 `7f690c5`。Plan 一直正确地把静态 lane 域向上取整：滤波器范围 5 对应 `fixed_8`，范围 3 对应 `fixed_4`。回归发生在 Triton 的通用 lane 投影：它不再读取 `AxisOp.tile`，而是从逻辑维度重新调用 `physicalExtent(extent)`。静态常量不在动态 block-extent 表里，因此重新得到 5 / 3，生成非法的 `tl.arange(0, 5)` / `tl.arange(0, 3)`。

#### 修复

通用 lane 发射遵守 Plan：

- `fixed_*` 直接使用 `AxisOp.tile` 中已经确定的物理范围；
- 动态 row-vector 才通过 block-extent binding 取得动态的 power-of-two 范围。

没有检查卷积 op 或 kernel 名字。所有固定静态 lane 都获得同一语义。

结果：Triton `conv1d`、`conv2d` 数值通过；当前 p50 分别为 0.0082 ms、0.0696 ms。cuTile 卷积旧/新 p50 分别保持 0.0164 / 0.0164 ms 和 0.0614 / 0.0614 ms；TileLang `conv1d` 保持 0.0082 / 0.0082 ms。

分类：`7f690c5` 引入的共享轴投影回归。

### 3. cuTile 五个未定义动态符号：全局 reuse 门槛误伤其他轴

影响：`dropout_residual_rms_norm`、`cross_entropy`、`boolean_reduction`、`value_select`、`sorted_nucleus_cutoff`，共 5 格。

#### 精确根因

`7f690c5` 把 row-vector 的动态逻辑 extent 接入 block-extent 机制时，使用了 kernel 级别的 `hasWorkerReuse`：只要任意轴复用 worker，就跳过所有 row-vector block extent。上述 kernel 同时存在：

- 一个复用 worker 的轴；
- 另一个不复用 worker、需要 `PHYSICAL_N` 或 `PHYSICAL_V` 的 row-vector lane。

Plan 因全局门槛没有生成后一轴的 block extent，cuTile emitter 只能打印 `ct.arange(N)` / `ct.arange(V)`；这些逻辑符号既不是核参数，也不是物理 constexpr，于是在 cuTile HIR→IR 阶段成为未定义变量。

五个 kernel 在 `7f690c5` 之前都已经存在，所以这是一个共享回归，不是五个新算法从未支持。

#### 修复

block extent 按轴决定：只跳过 `choice.reuse == true` 的那个 row-vector；其他不复用轴各自生成 power-of-two `BlockExtentOp`。现有 cuTile kernel-header 与 launch binding 随即自动得到 `PHYSICAL_N` / `PHYSICAL_V`，没有为五个 kernel 补参数或默认值。

结果：5 格全部数值通过。按各 repro 自己标注的既定计时 scope，当前 p50 分别为 0.2392、0.8266、0.0036、0.1140、0.0220 ms；这些不同 scope 的绝对值不横向比较。旧轮均在 JIT 前失败，因此也不构造伪性能比值。

分类：`7f690c5` 引入的共享物理 extent 回归。

### 4. TileLang 三个 LayoutInference 失败：生成形态而非能力缺失

影响：`embedding_backward_atomic`、`value_select`、`sorted_nucleus_cutoff`，共 3 格。

这三项的旧失败都发生在 TileLang 0.1.13 的 `LayoutInference`，但并不是同一个算法缺口。

#### tensor conditional

`value_select` 与 `sorted_nucleus_cutoff` 都把 tensor `select` / `mask` 的结果原地复用为某个输入 fragment。Plan 中的 `reuse_operand` 表示该值在语义和 liveness 上允许复用，不要求每个 target 必须原地别名。TileLang 对这两条具体 producer/consumer layout 不能同时满足原地别名约束。

投影现在仍验证 Plan 给出的 reuse 候选是否合法，但 tensor conditional 在 TileLang 上分配 fresh fragment。算法、索引、边界和条件表达式不变，只是不向下层强加可选别名。两个 repro 均通过。

#### atomic value

`embedding_backward_atomic` 原先先把 `grad_output[token, :]` 整块复制到一个 fragment，再在 `T.Parallel` 中逐元素原子累加。该 fragment 只有 atomic value 一个消费者，没有能给它确定布局的目标 primitive，导致 layout inference 无解。

新的逐 op 投影只在以下 use-def 条件同时满足时延迟该 load：它有一个结果、唯一消费者是 `intent.atomic_add`、并且正好作为 atomic value operand；staged 程序不走这条路径。atomic handler 随后使用原 `view_load` 的精确 element address，生成：

```python
T.atomic_add(grad_weight[embedding, d], grad_output[token, d])
```

这没有重新决定 ownership、tile、有效区间或算法结构，只消除了无语义必要的临时 fragment；并行 lane 与边界谓词都来自原 Plan。没有采用串行 atomic fallback。

结果：三项数值通过；当前 p50 分别为 0.1468、0.0880、0.0261 ms。旧轮没有成功执行 kernel，不能比较旧 p50。

分类：此前未闭合的 TileLang 等价投影；没有证据把它们归为本轮新回归，也不是下层 primitive 的硬能力缺失。

### 5. TileLang `conv2d`：保留真实能力边界

`conv2d` 的 patch load 同时使用两个 tensor-valued 间接索引，形成 rank>2 的 broadcasted indirect footprint。当前 TileLang surface 不能把它投影成一个并行 fragment。最新实际诊断仍为：

```text
'intent.view_load' op requires a multi-axis broadcasted indirect read footprint
that TileLang cannot project as one parallel fragment
```

这里没有改写成逐元素串行 load，也没有在 emitter 里重新推导卷积 halo、ownership 或 layout。那会把 TileLang 投影层变成第二个编译器，并产生“能跑但结构和性能错误”的假支持。

分类：从未支持的目标能力边界。`repro.sh tilelang conv2d` 仍保留为可手动执行的 capability repro，但不计为数值支持。

## 性能复核

### 现有硬门槛

仓库当前只有三项会在超过 upstream 5% 时直接失败；本轮全部通过：

| 项目 | 计时 scope | generated p50 | upstream p50 | 比值 |
|---|---|---:|---:|---:|
| Triton GEMM | kernel-only / CUDA Graph | 2.0900 ms | 2.0731 ms | 1.0081× |
| cuTile softmax | kernel-only / CUDA Graph | 0.3686 ms | 0.3688 ms | 0.9996× |
| cuTile GEMM | kernel-only / CUDA Graph | 2.0985 ms | 2.2511 ms | 0.9322× |

### 不规则收缩当前值

这些项旧轮没有运行到 kernel，下面只报告当前值，不声称相对旧 FAIL 的“加速”：

| Provider | MoE end-to-end p50 | Grouped GEMM 主例 end-to-end p50 |
|---|---:|---:|
| Triton | 8.7395 ms | 1.4392 ms |
| cuTile | 10.2760 ms | 1.4599 ms |
| TileLang | 11.4556 ms | 1.2747 ms |

### 新旧共同 PASS 的扫描

对旧轮和本轮共同 PASS 的主 p50 做了扫描。初次全量日志中只有三项增长超过 5%：

- cuTile LayerNorm backward：0.0822 → 0.1276 ms；
- cuTile ordered prefix：0.0395 → 0.0417 ms；
- TileLang ordered prefix：0.0235 → 0.0271 ms。

复测结果：

- cuTile LayerNorm backward 连续三次为 0.0829、0.0822、0.0830 ms，恢复旧值；0.1276 ms 是孤立抖动。
- cuTile ordered prefix 连续三次为 0.0392、0.0398、0.0417 ms，跨过旧值两侧，绝对波动约 0.002 ms。
- TileLang ordered prefix 连续三次为 0.0266、0.0193、0.0193 ms，短核波动大于初次差值，后两次优于旧值。

因此没有证据表明本轮实现造成稳定性能退化。全量串行运行的单次 p50 仍保留为原始观察，但不把一次短核抖动当作代码回归。

## 设计取舍

1. **staged 是收缩的物理合同，不是 producer 类别。** 先兑现 stage 已经确定的 residency，再用 producer-chain 判断普通收缩；反过来会把合法的 gather operand 错分。
2. **物理 extent 是逐轴事实。** 一个轴复用 worker 不能抹掉另一个轴的 block extent；这与逐轴角色分配的架构一致。
3. **发射器读取 Plan，不从逻辑 shape 重做决定。** Triton 固定小域回归正是 emitter 从 shape 重建物理范围造成的。
4. **可复用不等于必须别名。** TileLang 可以拒绝一个可选的 fragment alias，让下层 layout inference 选择 fresh fragment；这没有改变算法或机器级 ownership。
5. **等价投影优先于串行兜底。** atomic 直接读取原始 element 是机械 use-def 投影；把并行 atomic 改成 serial 只会制造慢而模糊的“支持”。
6. **真实能力边界保持显式。** TileLang `conv2d` 与 CAS 都明确 N/S。当前 TileLang 仍有 40 个数值 PASS，且新增修复只在逐 op 投影中，没有形成第二套 realizer，因此没有理由因这两项边界移除整个后端。

## 提交

- `bb20b26 restore shared physical projection contracts`
  - staged contraction 的共享驻留与 transfer deferral；
  - 逐轴 row-vector block extent；
  - Triton fixed lane 读取 Plan tile。
- `028fa34 repair tilelang layout projections`
  - tensor conditional 使用 fresh fragment；
  - 单一 atomic-value load 直接投影到原始 element address。

本轮实现提交只包含上述必要代码。工作区中原先存在的旧报告目录迁移没有混入这些提交。
