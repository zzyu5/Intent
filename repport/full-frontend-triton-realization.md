# Intent Kernel 编译器：10 Kernel 性能实现与共享 Ragged Realization 报告

## 1. 本报告的范围

本报告覆盖从 4 个代表性 kernel 扩展到 10 个 kernel，并进一步把这 10 个 kernel 的性能与上游高性能源码对齐的阶段。

当前覆盖：

| 结构 | Kernel |
|---|---|
| row reduction / normalization | softmax、LayerNorm、RMSNorm、logsumexp |
| tiled contraction | GEMM、dual GEMM |
| stateful streaming | attention、online softmax |
| ragged / irregular | MoE、grouped GEMM |

每个 kernel 都经过同一条编译链：

```text
Python Intent DSL
  → canonical Intent Kernel MLIR
  → KernelFacts / ScheduleStructure
  → shared GPU realization + search-space schema
  → Triton / cuTile / TileLang target projection
  → per-op source emission
  → target compiler / target tuner
  → real CUDA execution
```

本轮最终状态：

- 10 × 3 = 30 条公开 repro 全部数值 PASS；
- 三个 provider 都从同一份 canonical Kernel MLIR 和 GPU Plan 发射；
- realization 与 emission 中没有 kernel 名称分支；
- grouped GEMM 的 generated 最优 p50 为 `1.3129 ms`，上游三者最优为 `1.4260 ms`；
- MoE 的 generated 最优 p50 为 `8.8470 ms`，上游三者最优为 `9.3524 ms`；
- generated 的逐 kernel 赢家分布在 Triton、cuTile、TileLang 三者，而不是一个 provider 包办；
- 除没有任何等价上游 baseline 的 logsumexp 外，每个 kernel 的最佳 generated 均与最佳 upstream 同水平或更快；
- 工作树中没有额外 benchmark 结果文件、测试目录或隐藏验证链。

本阶段的两个核心提交是：

| Commit | 核心作用 |
|---|---|
| `a382e4a` | 保留 ragged stage 的真实 dtype、workspace 和终端写入语义 |
| `0f9e56c` | 建立 contiguous/indexed canonical ragged 关系，并实现 compact offset-tile traversal |

## 2. 性能工作的判断方法

这次没有从 kernel 名称出发添加优化，而是将 generated source 与上游同语言源码并排比较，并把差异分成三类。

### 2.1 算法不同

如果上游本身使用另一种算法，差异应进入 DSL 源码，而不是 target emitter。

本轮最重要的例子是 grouped GEMM。原来的 DSL 实际表达的是：

```text
任意 member permutation
  → indirect gather x
  → per-group contraction
  → scatter 回原始 row
```

而三个上游 grouped GEMM 的核心输入都是按 group 连续排列的矩阵段。它们不是同一个算法接口。继续用 permutation workload 比较，会把“真正 grouped GEMM”和“任意 indexed ragged GEMM”混在一起。

因此 grouped GEMM DSL 被改成连续 ragged 段；任意 index-map 语义仍由 MoE 保留。

### 2.2 同一算法，但 generated 缺少结构特化

这类差异应由共享 realization 从结构性质推出。

本轮真正产生决定性收益的是 ragged program grid。原 generated grid 使用：

```text
group_count × ceil(max_group_rows / M_tile)
```

每个 group 都按最大 group 的 tile 数启动 program。短 group 多出来的 program 虽然 member mask 全假，仍会进入完整 K-loop，并执行无效 contraction。

当前 grouped GEMM 的 group sizes 为：

```text
256, 256, 512, 1024, 2048, 2048, 1024, 1024
```

当 `M_tile = 128` 时：

```text
旧矩形 grid：8 × ceil(2048 / 128) = 128 个 M tile
真实有效量：sum(ceil(group_size / 128)) = 64 个 M tile
```

一半 program 都是空 work。这是从 offsets/ragged ownership 推导出的普遍结构问题，不是 grouped GEMM 名称特例。

### 2.3 结构相同，只是 tile 参数不同

这类差异继续交给 Triton、cuTile、TileLang 自己的 tuner。

realizer 只产生：

```text
合法 tuning key
合法参数 role
源码结构已经确定的 traversal
```

它不选择 winner，也没有自建 cost model。

## 3. 从 FlashAttention 源码中学到了什么

本轮重点阅读了：

```text
source/triton/flash-attention/attention/fused/flash_attn_triton.py
```

它的 forward kernel 集中展示了三类内容。

### 3.1 应属于 DSL 的算法语义

- online maximum / denominator / accumulator recurrence；
- causal mask；
- optional vector/matrix bias；
- mixed QKV/bias dtype；
- output 与 log-sum-exp 多输出；
- query/key traversal 的算法次序。

当前非 causal attention DSL 已经表达了 online state stream 与 QK/PV 两次 contraction。当前 repro 使用 `causal=False`，因此 causal control flow、bias 和多输出没有被本轮性能数字覆盖，不能把它们写成已经验证。

### 3.2 应属于 realization 的结构特化

- 已知整除时收紧 boundary；
- causal 时缩短 stream 上界，不启动必然无效迭代；
- program ownership 与 stream tile 的组合；
- 根据结构消除空 work。

本轮曾验证 shape-divisibility boundary specialization 是否能直接改善现有 Triton attention。生成代码数值正确，但 attention p50 从约 `4.95 ms` 变为约 `5.00 ms`。当前 Triton 下层已经能更好地处理原 masked form，因此没有保留会让 emitter 变厚且使性能退化的实现。

最终保留的优化是有明确结构收益的 compact ragged traversal。

### 3.3 不应复制的上游 workaround

- 为编译器 bug 增加的临时 buffer；
- store 后立即 load 的 workaround；
- 手工传入的大量 stride/cache-key 参数；
- 为特定编译器版本凑指令融合的表达重排；
- 下层应负责的寄存器分配、layout、指令选择。

这些没有上移进 Kernel IR 或 GPU Plan。

## 4. Canonical Ragged 语义为什么必须重做

### 4.1 原来的欠缺

原 `intent.ragged` 固定接收：

```text
outer + offsets + indices
```

这里把两件不同的事混在一起：

1. ragged member position 的完整 domain；
2. position 到真实数据 row 的可选映射。

没有 `indices` 时，原 schema 不仅失去映射，还失去 member 总 extent，因此无法正确表达连续 grouped GEMM。

### 4.2 当前 canonical schema

当前 canonical operation 为：

```text
intent.ragged(
  outer,
  members,
  offsets,
  optional indices
)
```

四部分语义分别是：

| 字段 | 语义 |
|---|---|
| `outer` | group/expert 的逻辑 domain |
| `members` | 所有 ragged position 的逻辑 universe 和总 extent |
| `offsets` | 每个 outer entity 拥有的 position 区间 |
| `indices` | 可选的 position → 数据 row 映射 |

因此同一个 Kernel IR 概念现在能表达两种关系。

MoE 使用 indexed ragged：

```python
R = member_routes.shape[0]
groups = I.ragged(
    outer=I.domain(0, E),
    members=I.domain(0, R),
    offsets=route_offsets,
    indices=member_routes,
)
```

grouped GEMM 使用 contiguous ragged：

```python
R, _ = x.shape
groups = I.ragged(
    outer=I.domain(0, G),
    members=I.domain(0, R),
    offsets=group_offsets,
)
```

`I.members(member_region)` 的语义随 canonical relation 自然确定：

- 有 index map：读取 `indices[position]`；
- 无 index map：直接使用 `position`。

这不是 target 决定，而是 Kernel IR 的逻辑关系。

### 4.3 Common facts 如何保留这个关系

`RaggedRelationFact` 当前分别保存：

```text
outerSource
memberSource
outerDomain
offsets
optional indices
memberDomains
```

member physical axis 的 shape symbol 来自显式 `memberSource`，不再由某个 target emitter 猜 `indices.shape[0]`。

这也修正了旧 surface emitter 中把 offsets shape `G_PLUS_1` 错当成 member extent 的可能性。MoE 两 stage workspace 和 grouped GEMM member tile 现在都从同一 canonical `R` 绑定获得。

## 5. Ragged Stage 语义如何被完整保留

### 5.1 Stage dtype 由 Kernel IR 决定

MoE 的第一 stage 为：

```text
f16 input × f16 W1
  → f32 accumulator
  → ReLU
  → explicit f16 cast
  → f16 stage workspace
```

第二 stage 再读取该 f16 workspace，与 f16 W2 contraction，并以 f32 累加。

三后端 emitter 不再使用“第一 stage f16、后续 stage f32”之类的 ordinal 规则，而是读取 canonical contraction operand/result element type。

### 5.2 TileLang allocation 使用物理 tile extent

TileLang 原 ragged path 会把 stage-local fragment/shared allocation 写成完整逻辑 feature `F`，造成远超 tile 的局部 buffer。

当前 stage-local allocation 使用：

```text
TILE_SIZE_M × TILE_SIZE_N
TILE_SIZE_M × TILE_SIZE_K
TILE_SIZE_K × TILE_SIZE_N
```

逻辑 workspace 的全局 shape 仍为 `R × F`；局部 tile shape 和全局逻辑 shape 不再混淆。

### 5.3 Unique write 与 reduction write 分离

grouped GEMM 的每个连续 member 只写一次，因此 DSL 使用：

```python
I.scatter_unique(...)
```

realization 将其变成 direct unique store。三后端分别发射普通 store/scatter，不使用 atomic。

MoE 的多个 route 可以累加到同一 token，因此继续使用：

```python
I.scatter_reduce(..., combine=I.add)
```

它才被 GPU Plan 兑现为 relaxed device-scoped atomic add。

这项区分来自 Kernel IR 的写入语义，不来自 kernel 名称。

### 5.4 Stage wrapper 由 terminal semantics 决定输出初始化

- unique store 输出使用 `empty`；
- additive atomic 输出使用 `zeros`；
- stage workspace dtype/shape 来自 canonical value；
- 每个 stage 只 materialize Plan 中明确列出的输入和输出。

## 6. Compact Offset-Tile Traversal

### 6.1 Plan 中只决定一次

GPU Plan 的 `ragged` operation 当前允许两种 traversal：

| Traversal | 适用 canonical relation | 物理意义 |
|---|---|---|
| `expert_offset_ranges` | indexed ragged | outer-major rectangular offset ranges |
| `compact_offset_tiles` | contiguous ragged | 把各 group 的有效 member tiles 压成连续 worklist |

当前 MoE 仍使用 indexed `expert_offset_ranges`，grouped GEMM 使用 `compact_offset_tiles`。

三个 target projector 只复制这一物理决定；surface emitter 不能重新选择 traversal。

### 6.2 Compact grid 的计算

wrapper 从 offsets 得到：

```python
route_lengths = tuple(
    int(length)
    for length in (offsets[1:] - offsets[:-1]).tolist()
)
```

对于 target tuner 选中的 `M_TILE`：

```text
total_route_tiles = sum(ceil(length / M_TILE) for length in route_lengths)
```

grid 变成：

```text
feature_tiles × total_route_tiles
```

不再使用：

```text
feature_tiles × group_count × max_route_tiles
```

`M_TILE` 仍由各 target tuner 选择。realizer 只决定 grid 必须使用 compact worklist，不固定 tile winner。

### 6.3 Kernel 内如何恢复 group ownership

每个 program 收到一个扁平 `route_tile_id`。它扫描很小的 group prefix：

```text
tile_cursor = 0
for candidate_group:
    group_tiles = ceil(group_length / M_TILE)
    if route_tile_id in [tile_cursor, tile_cursor + group_tiles):
        expert = candidate_group
        local_route_tile = route_tile_id - tile_cursor
    tile_cursor += group_tiles
```

然后恢复：

```text
route_begin
route_end
member position
feature tile
```

再执行完全相同的 contraction 和 terminal store。

### 6.4 三个 surface 如何打印同一决定

Triton：

- `G` 作为该结构的 `tl.constexpr`；
- 使用 `tl.program_id`、`tl.load`、`tl.cdiv`、`tl.where`；
- grid lambda 根据 target config 的 `BLOCK_SIZE_M` 计算 compact tile 数。

cuTile：

- group count 和 shape 为 `ConstInt`；
- 使用 `ct.bid`、`ct.load`、`ct.cdiv`、`ct.where`；
- exhaustive tuner 的每个 config 使用自己的 `TILE_SIZE_M` 计算 grid。

TileLang：

- `ROUTE_LENGTHS` 作为 builder-time tuple；
- builder 根据每个 autotune config 的 `TILE_SIZE_M` 计算 `total_route_tiles`；
- TIR body 使用显式 local scalar 和 `T.if_then_else` 恢复 group ownership。

三条代码的语法不同，但没有三份 traversal 决策。

## 7. 共享编译架构没有被性能工作破坏

### 7.1 Python frontend

Python 仍只负责：

- Python AST、closure 和 constexpr；
- dtype、symbol、shape、domain、region 的 lowering 临时状态；
- 带源码位置的诊断；
- 直接构造 canonical Intent Kernel MLIR。

不存在独立 typed Python Kernel IR，也不存在 Python physical-plan emitter。

### 7.2 Common realization

共享部分仍按 operation/region facts 工作：

```text
ABI analysis
domain and axis provenance
region structure
boundary proof
contraction facts
state-stream facts
ragged relation facts
schedule structure
physical schedule decision
Plan construction
```

没有 `if kernel == "moe"` 或 `if kernel == "grouped_gemm"`。

### 7.3 GPU Plan 与 search space

resolved realization 保存：

- ownership；
- mapping；
- traversal；
- storage/residency；
- boundary semantics；
- stream/ragged/stage 结构；
- matrix primitive role；
- unique/atomic terminal mechanism。

search space 只保存：

```text
shape-dependent key
target-neutral parameter roles
```

候选值、排序和 winner 继续由三个下层 tuner 负责。

### 7.4 Target emission

三个 emitter 仍只负责：

- capability/binding 检查；
- Plan concept 到目标语法的映射；
- target 要求显式写出的 buffer/scalar/tile materialization；
- per-op emission；
- target runtime 编译与运行接线。

本轮没有新增 kernel 专用 realizer、emitter 或 tool。

## 8. 统一 Repro 与计时口径

公开入口为：

```bash
./examples/run/repro.sh \
  <triton|cutile|tilelang> \
  <softmax|layer_norm|rms_norm|logsumexp|gemm|dual_gemm|attention|online_softmax|moe|grouped_gemm>
```

每条命令都会：

1. 从 DSL 重新 lower canonical MLIR；
2. 调用唯一 C++ 工具 `intent-compile`；
3. 构造 shared GPU realization；
4. 投影并打印 target source；
5. 由对应 target 环境编译/JIT；
6. 实际 launch CUDA work；
7. 对 generated、upstream 和 PyTorch reference；
8. 输出数值误差与 p50/p95。

公共 benchmark 使用 CUDA Events：

- 普通 kernel：25 次 warmup、100 次采样；
- MoE：5 次 warmup、20 次采样；
- p50/p95 是 GPU event elapsed time 的分位数；
- 初次源码生成、JIT 和 autotune 不在稳定采样中；
- callable 内排入 GPU stream 的额外 gather、cat、index-add、merge 会被计入；
- Python/CPU wall-clock 开销不会被 CUDA Events 完整反映。

所以这些数字是 wrapper 所排入 GPU 的工作时间，不应一概解释为单一 kernel body 时间。

## 9. 30 条数值结果

下表为 generated 与 PyTorch reference 的最大绝对误差。

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | `2.7939677e-09` | `1.8626451e-09` | `2.7939677e-09` |
| LayerNorm | `1.9073486e-06` | `1.9073486e-06` | `2.8610229e-06` |
| RMSNorm | `2.8610229e-06` | `1.9073486e-06` | `1.9073486e-06` |
| logsumexp | `9.5367432e-07` | `9.5367432e-07` | `9.5367432e-07` |
| GEMM | `0.0` | `0.0` | `0.0` |
| dual GEMM | `7.8125e-03` | `7.8125e-03` | `7.8125e-03` |
| attention | `3.0517578e-05` | `3.0517578e-05` | `3.0517578e-05` |
| online softmax | `3.7252903e-09` | `1.8626451e-09` | `1.8626451e-09` |
| MoE | `9.0594403e-06` | `9.0594403e-06` | `9.0594403e-06` |
| grouped GEMM | `1.9679070e-03` | `1.9679070e-03` | `1.9679070e-03` |

结果为 30/30 PASS。

dual GEMM、MoE、grouped GEMM 最终包含 f16 输出或 f16 stage handoff，因此误差尺度不能与全 f32 row reduction 直接比较。

## 10. 完整性能矩阵

表中每项为：

```text
generated p50/p95；upstream p50/p95
```

单位均为毫秒。`—` 表示当前 source 中没有等价 upstream baseline。

| Kernel | Triton | cuTile | TileLang |
|---|---:|---:|---:|
| softmax | `0.3540/0.3569；0.3558/0.3581` | `0.3579/0.3602；0.3580/0.3601` | `0.3492/0.3519；0.3638/0.3676` |
| LayerNorm | `0.1758/0.1778；0.1725/0.1737` | `0.1799/0.1821；0.3427/0.3455` | `0.1739/0.1758；—` |
| RMSNorm | `0.1768/0.1796；0.1747/0.1785` | `0.1799/0.1806；—` | `0.1737/0.1771；0.3499/0.3508` |
| logsumexp | `0.1635/0.1642；—` | `0.1696/0.1708；—` | `0.1614/0.1616；—` |
| GEMM | `2.1216/2.1244；2.1060/2.1101` | `2.3088/2.3127；2.2966/2.3057` | `2.1265/2.1421；2.3651/2.3729` |
| dual GEMM | `0.6977/0.6986；0.7740/0.7755` | `0.7205/0.7211；0.8092/0.8107` | `0.7349/0.7356；0.9212/0.9241` |
| attention | `4.9558/4.9599；4.9766/4.9845` | `5.0011/5.0130；66.7623/71.0124` | `4.8971/4.9087；6.6958/6.7171` |
| online softmax | `0.3826/0.3860；0.3560/0.3567` | `0.3499/0.3526；0.3580/0.3601` | `0.3806/0.3828；0.3649/0.3668` |
| MoE | `8.8470/8.8823；10.2980/10.3131` | `10.2704/10.3705；9.6117/9.6193` | `11.7792/11.8270；9.3524/9.3619` |
| grouped GEMM | `1.3129/1.3175；1.8848/1.9239` | `1.4552/1.4619；1.8720/1.8815` | `1.3756/1.3852；1.4260/1.4314` |

## 11. 每个 Kernel 的跨 Provider 最优值

最佳值按 p50 选择。

| Kernel | 最佳 generated | 最佳 upstream | generated / upstream | 结论 |
|---|---:|---:|---:|---|
| softmax | TileLang `0.3492` | Triton `0.3558` | `0.9815×` | generated 更快约 1.9% |
| LayerNorm | TileLang `0.1739` | Triton `0.1725` | `1.0081×` | 同一水平，generated 慢约 0.8% |
| RMSNorm | TileLang `0.1737` | Triton `0.1747` | `0.9943×` | generated 更快约 0.6% |
| logsumexp | TileLang `0.1614` | — | — | 无 upstream，不能给相对结论 |
| GEMM | Triton `2.1216` | Triton `2.1060` | `1.0074×` | 同一水平，generated 慢约 0.7% |
| dual GEMM | Triton `0.6977` | Triton `0.7740` | `0.9014×` | generated 更快约 9.9% |
| attention | TileLang `4.8971` | Triton `4.9766` | `0.9840×` | generated 更快约 1.6% |
| online softmax | cuTile `0.3499` | Triton `0.3560` | `0.9829×` | generated 更快约 1.7% |
| MoE | Triton `8.8470` | TileLang `9.3524` | `0.9460×` | generated 更快约 5.4% |
| grouped GEMM | Triton `1.3129` | TileLang `1.4260` | `0.9207×` | generated 更快约 7.9% |

generated 赢家分布：

| Provider | 赢得的 kernel |
|---|---|
| Triton | GEMM、dual GEMM、MoE、grouped GEMM |
| cuTile | online softmax |
| TileLang | softmax、LayerNorm、RMSNorm、logsumexp、attention |

这说明三条 surface 不是只做到“都能运行”；下层工具的不同强项确实进入了最终结果。

## 12. 两个共享机制的实际收益

### 12.1 Ragged stage semantics 对 MoE 的收益

相对于上一份报告中的 generated p50：

| Provider | 旧值 | 当前值 | 降低 |
|---|---:|---:|---:|
| Triton | `13.0976` | `8.8470` | 约 32.5% |
| cuTile | `16.1568` | `10.2704` | 约 36.4% |
| TileLang | `21.6398` | `11.7792` | 约 45.6% |

收益来自正确的 stage dtype handoff、tile-local allocation、workspace shape 和 terminal write 语义，而不是 MoE 专用 emitter。

### 12.2 Compact traversal 对连续 grouped GEMM 的收益

在 canonical contiguous grouped GEMM 已经跑通、但尚未加入 compact traversal 时，generated p50 为：

```text
Triton  2.3570 ms
cuTile  2.6591 ms
TileLang 2.4847 ms
```

加入 compact traversal 后：

| Provider | rectangular grid | compact grid | 降低 |
|---|---:|---:|---:|
| Triton | `2.3570` | `1.3129` | 约 44.3% |
| cuTile | `2.6591` | `1.4552` | 约 45.3% |
| TileLang | `2.4847` | `1.3756` | 约 44.6% |

三条路径同时获得约 44%–45% 的收益，符合“同一个机器决策只做一次”的预期。

## 13. Baseline 的来源与公平性边界

| Kernel | Triton baseline | cuTile baseline | TileLang baseline | 口径说明 |
|---|---|---|---|---|
| softmax | standalone fused softmax | standalone softmax | online-softmax source adapter | 接近等价 |
| LayerNorm | FlashAttention LayerNorm | NVIDIA cuTile LayerNorm | unavailable | TileLang 无 baseline |
| RMSNorm | Liger Triton RMSNorm | unavailable | normalization kernel + 外部 weight multiply | TileLang 是组合 baseline |
| logsumexp | unavailable | unavailable | unavailable | 只能报告 generated |
| GEMM | standalone matmul | standalone matmul | standalone matmul | 接近等价 |
| dual GEMM | 两次 matmul + epilogue | 同类组合 | 同类组合 | generated 是 fused kernel，fusion 粒度不同 |
| attention | fused attention | FMHA wrapper | TileLang MHA | cuTile wrapper 开销污染严重 |
| online softmax | fused/整行 softmax | softmax | online-softmax adapter | 算法结构不完全一致 |
| MoE | grouped GEMM 组合 + merge | fused-MoE 组合 | grouped GEMM 组合 + merge | 不是统一的 standalone kernel 边界 |
| grouped GEMM | list views + grouped kernel + cat | list views + grouped kernel + cat | packed contiguous A + offsets kernel | TileLang 最接近 canonical contiguous ABI |

必须保留的解释边界：

1. cuTile attention upstream p50 为 `66.7623 ms`，主要受 source wrapper 影响，不能解释为 generated kernel 快 13 倍；
2. dual GEMM generated 把两个 contraction 和 epilogue 融合，baseline 是两个上游调用的组合；
3. online softmax 的两遍 state-stream 算法与部分整行 baseline 不相同；
4. MoE baseline 的排序、分组、多个 kernel 和 merge 边界在三个 provider 中不同；
5. grouped GEMM 当前算法已统一为 contiguous ragged，但 Triton/cuTile 的上游 API 返回 matrix list，因此仍需要 `torch.cat` 适配；
6. logsumexp 三个 provider 都没有等价上游 source，不能用 PyTorch reference 代替“高性能 baseline”做性能结论。

当前 unavailable 的 5 个组合是：

```text
Triton  + logsumexp
cuTile  + RMSNorm
cuTile  + logsumexp
TileLang + LayerNorm
TileLang + logsumexp
```

## 14. 为什么这些结果证明编译器在发射

### 14.1 Kernel IR 仍是唯一算法 IR

normalization、contraction、stream、ragged 的算法差异都在 DSL 与 canonical Kernel MLIR 中表达。Python 没有第二份 typed kernel graph，target emitter 也不识别 kernel 名称。

### 14.2 物理机制由 facts 组合产生

| 结构事实 | Shared physical mechanism |
|---|---|
| one program + one vector domain | persistent row-strided mapping |
| two tiled program axes + contraction axis | grouped 2D tile mapping |
| query ownership + ordered contraction stream | multi-axis stream |
| row ownership + ordered state stream | row stream |
| indexed ragged + contraction stages | expert offset ranges |
| contiguous ragged + contraction stages | compact offset tiles |

这些是结构判定，不是 kernel 枚举。

### 14.3 新增机制没有新增 kernel 专用文件

本轮修改发生在已有的：

```text
frontend lowering
canonical Intent op schema
common KernelFacts
GPU Plan build
target ragged dialect verifier
三个既有 source emitter 的 ragged concept spelling
统一 repro adapter
```

没有产生：

```text
GroupedGemmRealizer.cpp
MoeEmitter.cpp
FlashAttentionMatcher.cpp
```

### 14.4 第五个同类 kernel 的验收方式仍成立

再增加一个 kernel 时：

- realization/emission 不应新增 kernel 文件；
- 只有在出现新 op 或新物理 concept 时增加 handler/capability；
- 若只是已有 domain、stream、contract、ragged、store 的新组合，应自然走现有路径。

## 15. 当前准确边界

| 边界 | 当前状态 |
|---|---|
| causal attention | Kernel DSL 中存在相关算法表达，但当前 10×3 repro 固定 `causal=False`，未形成完整性能闭环 |
| attention bias / LSE 多输出 | 从上游源码确认是需要的语言能力，当前 benchmark kernel 尚未要求，未提前造接口 |
| 多 ragged relation | facts/Plan 容器可索引多个；当前 GPU ragged staging 明确接受一个 canonical relation |
| indexed ragged compact traversal | 当前 compact 机制由 contiguous relation 触发；indexed MoE 保持 expert-offset traversal |
| provider-local MoE parity | 跨 provider 最佳 generated 已快于最佳 upstream，但 cuTile/TileLang generated 仍慢于各自 provider baseline |
| logsumexp baseline | 三个 provider 都缺少 standalone source，因此只有 generated/reference 数据 |
| CPU / RVV | 当前报告只覆盖 GPU 和三个 GPU surface |
| layout / register / instruction selection | 继续委托给下层 compiler，不在上层复制 |
| cost model | 不存在；winner 由目标 tuner 选择 |

## 16. 最小复现

任意单条：

```bash
./examples/run/repro.sh triton grouped_gemm
./examples/run/repro.sh cutile moe
./examples/run/repro.sh tilelang attention
```

完整矩阵的合法 kernel 名称为：

```text
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
```

每条命令都从 DSL 重新生成目标代码并实际运行，没有 test 目录、pytest fixture 或预生成结果兜底。

## 17. 最终结论

这一阶段真正形成的不是 10 份 kernel 模板，而是一组能被不同算法组合的编译机制：

```text
canonical logical domains
explicit member universe
optional ragged index map
row/tiled/stream/ragged ownership
boundary proof
state carry
contraction stages
stage dtype and workspace semantics
unique versus reducing terminal writes
compact ragged tile traversal
target-neutral tuning roles
three target projections
```

性能工作的核心收益也来自这些机制：MoE 三后端的 generated p50 相比旧报告降低约 32%–46%，连续 grouped GEMM 三后端通过同一个 compact traversal 同时降低约 44%–45%。

当前 30 条 repro 已经证明：同一份算法 IR 与同一份 GPU physical decision 可以由 Triton、cuTile、TileLang 分别渲染，并让三个下层工具各自在不同 kernel 上成为赢家。没有 baseline 的组合和不公平的 wrapper/composition 比较仍被明确标出，没有用比值掩盖。
