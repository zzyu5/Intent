# 陌生算法与等价写法检验报告

## 结论

当前检验不是从编译器已有能力反推语料，而是先照公开实现或真实模型结构选算法，再直接写 DSL。两批陌生算法和一批等价写法的结果如下：

| 检验组 | Triton | cuTile | TileLang | 合计 |
|---|---:|---:|---:|---:|
| 第一批 10 个陌生算法 | 10/10 | 10/10 | 9/10 | 29/30 |
| 10 个已有算法的等价写法 | 10/10 | 10/10 | 9/10 | 29/30 |
| 第二批 10 个陌生算法 | 10/10 | 8/10 | 9/10 | 27/30 |
| 合计 | 30/30 | 28/30 | 27/30 | 85/90 |

第二批没有通过的三格都属于整行 scan 的下层编译问题：`nonzero_compact × cuTile`、`unique_consecutive × cuTile` 在 cuTile 编译器的既定限时内未完成，`unique_consecutive × TileLang` 的候选首次 JIT 超过十分钟。三者都已经完成 Kernel MLIR、Physical Plan 和目标源码生成；当前如实记为 `downstream_fail`，没有增加按算子分支，也没有把串行慢路径伪装成支持。

全量固定表现在包含 72 个 kernel、79 个 case、3 个 provider，共 237 格：

| 状态 | 数量 |
|---|---:|
| 数值通过 | 230 |
| 明确不支持 | 3 |
| 下层失败 | 4 |

完整数字在 `report/baseline/kernel-performance.csv`。既有 source baseline 的测试逻辑和数字没有改动；第二批没有可直接拆出的同算法 kernel-only adapter，因此 source 数字留空，不拿 PyTorch reference 或相邻算法冒充上游时间。

## 一、第二批算法是怎样选的

| Kernel | 原算法中的关键结构 | 本轮实际压到的边界 |
|---|---|---|
| `nonzero_compact` | 筛选、前缀计数、数据相关输出数 | 固定上界输出加逐行有效计数；scan 与 scatter 同时存在 |
| `unique_consecutive` | 相邻比较、run id、run length | 多输出、原子返回值参与后续索引、整行 scan |
| `moe_align_block` | count → prefix → scatter → block owner | 作者显式编排四次调用；中间计数决定后续地址和有效区间 |
| `nested_ragged_pool` | document → sentence → token 两级变长层次 | 两个嵌套不规则关系共同决定同一访问 |
| `adamw_update` | 参数、梯度、两份状态原地更新 | 五个以上异构输入输出、别名与多 InOut 状态 |
| `adafactor_update` | 行状态、列状态、全局均值、参数更新 | 作者显式编排三次调用；不同秩广播和多段归约 |
| `reshape_and_cache` | slot map 间接定位 KV cache | 输入输出别名语义、间接多轴原地写入 |
| `group_norm_silu_backward` | 归一化与激活反向融合 | bf16/f32 混合、多输出、不同秩广播、原子累加 |
| `batched_cholesky` | 数据依赖三角递推 | 原地矩阵更新、动态标量索引、顺序依赖 |
| `batched_householder_qr` | Householder 反射与尾部更新 | 原地矩阵、多结果、循环内归约与 rank-2 动态索引 |

这里有一条必须诚实限定：当前“数据决定下一阶段”已经能通过作者显式多调用和预分配上界表达，数据相关的计数、地址和有效区间会真实流入后续 kernel；但运行时计数还不能改变下一次 launch 的物理 program 数，也没有按计数动态分配恰好大小的输出。编译器没有越过单次调用边界替作者安排 launch 或张量布局，这个边界被保留了。

## 二、公开结构参考

source 仍按语言、上游和算子职责保存，文件不为 Intent DSL 改写：

| 算法 | 公开结构参考 | 对照性质 |
|---|---|---|
| nonzero compact | FlagGems `nonzero` | 同类压缩实现 |
| unique consecutive | FlagGems `unique_consecutive` | 同算法、多阶段实现 |
| MoE block alignment | TileGym cuTile MoE alignment | 同算法流水线 |
| AdamW | FlagGems fused Adam | 同类原地优化器更新 |
| reshape and cache | FlagGems `reshape_and_cache` | 同算法间接 cache 写入 |
| batched Cholesky | FlagGems `linalg_cholesky` | 同类分解 |
| batched Householder QR | MAGMA batched QR | 同算法家族 |
| nested ragged pool | 未找到可直接复用的高质量单-kernel实现 | 不造假对照 |
| Adafactor | 未找到可直接复用的高质量 GPU kernel | 不造假对照 |
| fused GroupNorm + SiLU backward | 未找到完全同算法的公开 kernel | 不用相邻 GroupNorm 冒充 |

这些是选择性保留的上游源码文件，不是完整仓库 checkout；FlagGems benchmark 脚本和 MAGMA 内核仍依赖未复制的 upstream 公共模块，因此本轮只把它们作为结构参考，没有把它们描述成当前 checkout 可独立运行的 baseline。这些源码的作用是约束算法结构和高性能写法，不是把相近实现的数字塞进性能表。

## 三、第二批 10 × 3 结果

表中 `K` 是生成端只计 kernel launch 的口径，`E` 是完成整个作者流水线所必须的 GPU 工作。所有 p50/p95 单位均为毫秒。

| Kernel | Scope | Triton p50/p95 | cuTile p50/p95 | TileLang p50/p95 |
|---|:---:|---:|---:|---:|
| nonzero compact | K | 0.8270 / 0.8281 | downstream fail | 1.6404 / 1.6456 |
| unique consecutive | E | 1.5498 / 1.5513 | downstream fail | downstream fail |
| MoE block alignment | E | 0.1594 / 0.1682 | 0.0757 / 0.0824 | 0.0799 / 0.0921 |
| nested ragged pool | K | 0.2279 / 0.2292 | 0.6399 / 0.6412 | 0.2782 / 0.2797 |
| AdamW update | K | 0.1188 / 0.1198 | 0.1188 / 0.1208 | 0.1188 / 0.1199 |
| Adafactor update | E | 0.2673 / 0.2695 | 0.2696 / 0.2735 | 0.2560 / 0.2580 |
| reshape and cache | K | 0.0302 / 0.0321 | 0.0302 / 0.0321 | 0.0302 / 0.0323 |
| GroupNorm + SiLU backward | E | 0.0616 / 0.0706 | 0.0508 / 0.0557 | 0.0361 / 0.0450 |
| batched Cholesky | K | 0.0365 / 0.0391 | 0.0348 / 0.0369 | 0.0317 / 0.0349 |
| batched Householder QR | K | 0.1188 / 0.1188 | 0.1761 / 0.1762 | 0.1167 / 0.1173 |

这些延迟证明生成 artifact 可实际运行并对数值；没有同算法上游 kernel-only adapter 的格子不能用于声称超过上游。

## 四、这一批怎样修改了编译器

### 1. 原子更新的返回值成为普通 SSA

`unique_consecutive` 需要用 atomic add 返回的旧值作为紧接着的写入下标。作者已经在源码写下了这个数据依赖，原实现却把 atomic 当成无结果副作用，前端在这里丢了信息。现在 `intent.atomic_add` 具有一个结果，分析只验证地址和值，三个 leaf 都把目标原语返回的旧值绑定回同一个 SSA；没有另建 unique 路径。

### 2. 不规则关系从“各有一个”变成可以嵌套消费

`nested_ragged_pool` 同时有 document→sentence 和 sentence→token 两个关系。共享 Facts 允许外层 selector 来自父级不规则成员，并在 relation lookup 时按 member axis 消歧；Surface Plan 直接用父成员的运行时索引读取下一层 offsets。三个 emitter 只读取同一份层级关系和动态 begin/end，不自行判断这是 nested ragged。

### 3. source axis 与 result axis 被彻底分开

`new_axis`、广播和 scalar gather 同时出现时，旧 leaf 会把结果维度编号误当成 source view 维度，从 shape 或逻辑维度重建地址。cuTile 和 TileLang 现在都维护独立的 source/result 游标；地址、精确 mask 和 physical extent 都从 access relation 与 Plan 取值。这个修复同时覆盖 GroupNorm 广播和 cache 的多轴间接写入。

### 4. 标量 transfer 具有明确的物理所有者

顺序递推里的 scalar store/atomic 以前可能被 TileLang 的所有线程重复执行。Plan 已经知道标量域是否被 packed；TileLang leaf 现在只投影这项决定：未 packed 的标量 transfer 由 thread 0 执行，packed 域则使用计划中的 lane/tail predicate。动态地址的边界谓词也直接作用于精确地址，不再从周围 tensor 形状猜。

### 5. padding 绑定到消费者，不绑定到生产者名字

同一个中间值可能先经过逐元素链再进入 sum、max 或收缩。共享 realizer 现在按具体 consumer 选择恒等填充值，并让 physical expansion 排除纯 `new_axis`。Emitter 只在 Plan 明确给出 fill 或物理扩张时打印填充；`fill=none` 不再被 TileLang 擅自扩成逻辑边界 mask。

### 6. InOut 的调优候选互不污染

原地算法第一次暴露出调优语义错误：候选依次运行在前一候选已经修改过的状态上。cuTile 只在候选试跑时 clone InOut，正式 launch 保持原对象；TileLang 的 autotune supplier 为每个候选提供 fresh clone，并在编译结束后释放临时输入。这个决定属于运行接线，不进入 Kernel IR 或 Physical Plan。

### 7. 测量区间不再把状态恢复算进 kernel

runner 的 `prepare` 在 CUDA start event 之前恢复 AdamW、Adafactor、Cholesky 和 QR 的原始状态。MoE 对照按每个 expert 的有效分段比较，而不是把全局排序误当成算法合同。这个改动只修正第二批测量，没有动既有 source baseline 口径。

## 五、失败归因与仍未闭合的边界

### 整行 scan 的共享 realization 仍然缺一格

两个 compaction 算法把 scan 直接展开为完整行上的顺序递推。Triton 能编译，TileLang 对较小的 nonzero 能编译但代价很高，cuTile 两项都在下层编译限时内失败，TileLang 的 unique 也出现不可接受的首次 JIT 时间。证据指向的是同一个结构问题：当前 Physical Plan 只给出了“整行有序”，没有给出通用的 chunked scan、块内 scan 与块间 carry 组合。

这不是 cuTile 或 TileLang 应该各自决定的，也不是 `nonzero` / `unique` 两个算子分支。正确落点只能是共享物理决策：由算法 IR 的 scan 依赖确定顺序语义，由 realizer 选择分块与 carry 结构，再让目标投影原语或显式循环。此轮没有用 target 私有特判绕过，因此三格保留为下层失败。

### 数据相关输出目前是上界加计数，不是动态分配

nonzero、unique 和 MoE 都真实产生运行时计数，但输出 buffer 仍由调用者按静态上界预分配。MoE 的后续调用会消费 count/prefix 形成真实地址，却仍以静态上界启动。让 runtime 计数改变下一次 launch 或分配大小属于调用编排层，不应被单-kernel 编译器悄悄决定；当前需要作者在普通 Python 外层显式做这件事。

### 公开 source 与性能 adapter 是两件事

第二批已经保存能找到的原始高质量 source，但尚未为它们建立同算法、同 kernel 数量、同计时范围的 runtime adapter。因此 CSV 只记录 generated 数字。缺少可比 adapter 不是编译失败，也不能用 PyTorch 端到端数字补空。

## 六、架构边界自查

- 新算法没有让 realization 或 emission 目录新增按 kernel 分类的文件，也没有按 entry 名选择策略；算法类别仍不是调度入口。
- 三个 target 共用 Kernel IR、Facts、Physical Plan 和 Surface Plan。目标 leaf 新增的是 op 映射、能力检查与运行接线，没有重新推导所有权、ragged 层级或 padding。
- owner-private workspace 的线性地址已有共享 `projectPrivateWorkspaceOffset`，旧报告中“三个 leaf 各拼一份”的判断已经过时；本轮没有恢复重复路径。
- cuTile 最终 launch 不 clone InOut；clone 只属于候选隔离。TileLang 同理，目标接线没有改变作者可观察的别名语义。
- TileLang 在第二批通过 9/10，并准确暴露了 source/result axis、标量所有者和 fresh autotune input 三个共享或接线问题。它仍是能力投影层，不是第二套编译器。
- 保留下来的 target 较厚代码都在打印目标 API、组织 autotune 或声明能力子集；审计没有发现按算法名字重做物理决定的路径。

## 七、可复现入口

每一格都仍由同一条人工 repro 入口生成目标代码、实际 launch 并对数值：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <kernel>
```

第二批 kernel 名为：`nonzero_compact`、`unique_consecutive`、`moe_align_block`、`nested_ragged_pool`、`adamw_update`、`adafactor_update`、`reshape_and_cache`、`group_norm_silu_backward`、`batched_cholesky`、`batched_householder_qr`。
