# Intent Kernel 编译器多轮推进与当前状态

## 报告范围

这份报告覆盖从 `06b2969` 到当前实现提交 `8502f5b` 的连续推进，以及随后 `7025343` 对 RTX 5090 D 全量结果的固定。时间范围是 2026-08-13 00:21 至 2026-08-14 11:28。

这不是单独一轮 source 清点，而是下面六条工作线交错推进的结果：

1. 把设备事实和测量接线从单机经验改成可跨机器运行的合同；
2. 用 `source/` 中已有上游实现补齐 12 个 source-driven case；
3. 用 MLA、FP8 和 token-sparse attention 压出 broadcast、间接 gather、batched contraction 与下层编译成本边界；
4. 并排阅读三个目标源码，修掉错误原语和慢速伪支持；
5. 补入注意力、因果卷积和块稀疏注意力的反向/多调用流水线；
6. 最后补 2:4 稀疏、分页 MLA、分页 split-K 和变长 GQA decode，并完成 RTX 5090 D 全量复验。

从基线到实现提交，非 `report/` 代码一共修改 55 个文件，约 `+7228/-926`。固定 case 数从 89 增加到 113；增加的不是 24 条后端专用路径，而是 24 个新的 `(kernel, case)`，它们继续经过同一份 canonical Kernel MLIR、共享 GPU Physical Plan 和三个目标叶子。

## 一、起点、过程与当前落点

| 时点 | RTX 5090 D 固定 case | H100 固定 case | 这一阶段的含义 |
|---|---:|---:|---|
| `3c0f176` | 89 | 89 | 首次完成两机同 schema 全量，验证设备事实与下层 tuner |
| `6f40b3d` | 101 | 101 | 从已有 source 补入 12 个 case |
| `cff0e27` / `b55f21a` | 106 | 106 | 补入吸收式 MLA、token-sparse MLA、FP8 MQA 与 batched contraction |
| `2626597` / `8502f5b` | 代码已新增 7 case | 尚未重新部署 | 反向、block sparse、分页与结构化稀疏进入代码 |
| `7025343` | 113 | 106 | 5090D 完成当前代码全量；H100 被外部任务阻塞，保持此前固定表 |

当前真正成立的状态是：

- RTX 5090 D 表对应当前 113 个 case 的一次完整运行；
- H100 表包含此前已经真实运行或定向补测的 106 个 case，但不是最新 113-case 代码的全量；
- `source/` 的候选已全部审视，仍有 2 个独立算法明确“尚未落地”，不能说 source 全部实现；
- 当前全量暴露了 3 组真实数值/前端回归，报告保留运行时真值，没有在测完后悄悄修表。

## 二、贯穿所有轮次的架构合同

实际主链保持为：

```text
Python DSL source
  -> Python frontend lowering
  -> canonical Intent Kernel MLIR
  -> shared Kernel facts
  -> GPU Physical Plan MLIR
  -> Triton / cuTile / TileLang target projection
  -> 下层 JIT / autotuner
  -> 真实 GPU 运行与数值对照
```

### 1. Python frontend

Python 只负责语法、constexpr、符号/shape/region 的 lowering 临时状态和源码诊断，然后直接构造 canonical MLIR。当前没有 typed Python Kernel IR、Python Physical Plan 或 Python target emitter。

这几轮新增到前端的内容只有作者确实需要陈述的算法语义：

- `contract(..., batch=...)`：批量轴是算法中两个 operand 的逻辑对应关系；
- `I.sparse_contract_2to4(...)`：作者明确声明这是一条结构化 2:4 稀疏收缩；
- FP8/E8M0、i8/u8/i64 等真正进入算法 ABI 的类型；
- 已有 index relation、broadcast、ragged relation、state stream 和 helper 调用的完整保留。

### 2. Canonical Kernel IR

Kernel IR 是唯一算法真理。分页 attention、GQA、MLA、split-K、反向和 block sparse 没有各自专用的 kernel op；它们由已有的逻辑域、索引关系、ragged relation、ordered state、contract、reduce、load/store 与多函数调用自然组合。

这一阶段唯一新增的算法 op 是 `intent.sparse_contract`。原因是 dense contraction 与结构化稀疏 contraction 的数学输入合同不同：后者包含 compressed values、metadata 和 RHS，并且稀疏格式是作者算法的一部分，不能让后端从名字或 shape 猜。

### 3. Shared facts 与 Physical Plan

共享 facts 现在保存 domain、extent、parallel ownership、boundary/fill、tensor indexing、value axis、ordered/serial/reduction/contraction domain、scan/state stream、ragged relation、whole-view load/scatter、dense/sparse contraction、logical buffer 和 access range。

Physical Plan 只保存“有多个合法实现时选了哪个”的机器决定，包括：

- 每个逻辑轴承担哪些物理角色；
- ownership、traversal、reduction、lane、access footprint 等用途各自对应哪个 range；
- program order、worker fold/reuse、persistent traversal；
- buffer residency、padding、transfer、stage 与 sparse/dense contraction 的物理绑定；
- 不改变源码结构的 tile/occupancy/stage 等候选搜索空间。

Plan 没有 `rowwise`、`attention`、`moe` 或其他 kernel category，也没有一个“kernel 属于哪一类”的入口字段。绑定通过 Kernel IR node/value ID 完成，不依赖 Python 变量名或函数名。

### 4. 三个目标叶子

目标侧只能做四类事情：能力检查、Plan 概念到目标语法的映射、ABI/runtime 接线、逐 op handler。表达不了时在进入下层 JIT 前拒绝，不保留数量级更慢的替代算法冒充支持。

最近几轮的所有新分支均以 canonical op、axis role、index relation、Plan binding 或 target capability 为条件。`lib/` 中没有按 `absorbed_mla`、`token_sparse_mla`、`paged_mla`、`paged_splitk`、`varlen_gqa` 或 `sparse_2to4` 名字决定 lowering 的路径。

### 5. 关键代码落点

| 层 | 主要文件/符号 | 本轮承担的职责 |
|---|---|---|
| 设备能力 | `python/intent/targets/gpu/device.py::resolve_gpu_device` | 通过 CUDA Driver API 取得 SM、shared memory、register 与 matrix capability |
| 前端 structured lowering | `python/intent/frontend/lowering/intrinsics/structured.py::_contract` | 校验并生成 canonical contract/batch relation |
| 前端 sparse lowering | `structured.py::_sparse_contract_2to4` | 校验 compressed/metadata/RHS ABI 并生成 sparse contract |
| Kernel IR schema | `include/Intent/Dialect/Intent/IR/IntentOps.td` | 定义唯一新增的 `intent.sparse_contract` 算法 op |
| Plan schema | `include/Intent/Dialect/Plan/IR/PlanOps.td` | 定义 axis/range/transfer/buffer/scan/contract/sparse/stage/autotune 等物理绑定 |
| Plan verifier | `lib/Dialect/Plan/IR/PlanOps.cpp` | 拒绝 role/range/residency/schema 不一致，不依靠 emitter 兜底 |
| Index relation | `lib/Target/Common/Analysis/IndexRelation.cpp` | 区分 structured/data-dependent index，并闭合 unit-scalar projection |
| Shared facts | `lib/Target/Common/Realization/KernelFacts.cpp` | 保存 batch、multi-domain provenance、ragged/stream、dense/sparse contraction 等算法事实 |
| GPU axis decision | `lib/Target/GPU/Realization/Plan/Decisions.cpp` | 逐轴分配 role/range/program mapping，处理 ordered+reduction 与 batch axis |
| GPU plan build | `lib/Target/GPU/Realization/Plan/Build.cpp` | 生成 buffer residency、transfer、stage、dense/sparse contraction 等已选决定 |
| Target op projection | `lib/Target/{Triton,CuTile,TileLang}/Emission/Handlers/Operations.cpp` | 逐 op 读取 Kernel IR + Plan，选择目标原语拼写 |
| Target capability/runtime | `lib/Target/{Triton,CuTile,TileLang}/Emission/Source/Emitter.cpp` | 在 JIT 前拒绝目标无法表达的组合，并生成 wrapper/autotune 接线 |
| DSL 语料 | `examples/kernels/{backward,contraction,streaming,...}` | 作者算法源码；多调用关系由 Python 外层明确编排 |
| 唯一 repro | `examples/run/repro.sh` 与 `examples/repro/common/extended.py` | 编译 DSL、运行生成 artifact、对 reference/upstream、输出 p50/p95 |

这张表也给出本轮判断“该改哪一层”的实际依据：作者语义缺失改 frontend/Kernel IR；多个合法物理方案中的选择改 Plan；同一 Plan 在一个 surface 上拼写低质只改目标 handler；下层没有等价能力则在 target capability 处拒绝。

## 三、第一阶段：把“同一份算法换机器”真正跑起来

对应提交：

- `a51ca74 fix(target): query GPU capabilities through CUDA driver`
- `03ab117 fix(repro): make cross-device measurement portable`
- `dd0768e fix(cutile): isolate autotune candidate failures`
- `3c0f176 docs(report): record H100 cross-device realization`

### 1. 设备事实不再依赖 PyTorch wrapper

GPU device resolver 改用 CUDA Driver API 查询：

- compute capability；
- multiprocessor 数；
- 每个 SM 的 shared-memory 上限；
- 每个 SM 的 register 上限；
- matrix-unit capability。

查询失败直接带 CUDA error name 报错，不用缺省值兜底。这样 H100 上较旧 PyTorch 没暴露 `shared_memory_per_multiprocessor` 时，不会把 Python wrapper 的字段集合误当成设备合同。

当时实测设备事实：

| 设备事实 | RTX 5090 D | H100 80GB HBM3 |
|---|---:|---:|
| SM 数 | 170 | 132 |
| shared memory / SM | 102400 B | 233472 B |
| registers / SM | 65536 | 65536 |
| L2 | 100663296 B | 52428800 B |
| warp size | 32 | 32 |

这些事实的消费边界也被明确区分：SM 数影响 persistent/row launch 的实际 program 数；L2 决定 CUDA Graph 重放前的 cache-flush buffer；register budget参与 private-buffer residency；shared memory 参与下层候选合法性。被查询不等于全部参与结构决策。

### 2. repro 接线可跨机器

`examples/run/repro.sh` 增加仓库外 build root、CMake generator、MLIR/LLVM package 与 Python executable 的显式环境入口。H100 可以使用独立的 Triton、cuTile、TileLang 0.1.13 环境，不把机器路径、虚拟环境或编译缓存写进项目。

原先 softmax/GEMM runner 中固定的“generated 不得比 source 慢 5%”失败门被删除。这个阈值是某一台机器上的性能关系，不是 correctness。repro 只对数值失败，性能写入 CSV，由同机器同 scope 数字判断。

### 3. cuTile 候选失败隔离

H100 上 `attention_bias` 曾出现整个 search space 无有效配置，但独立进程逐候选检查后发现 4 个候选中 3 个正确、1 个触发 `misaligned address`。问题是单候选的异步 CUDA 错误污染同一 context，使后续候选一起失败。

生成的 cuTile wrapper 接入下层已有的 `single_run_timeout_sec`，让候选首次运行进入隔离 worker。没有删除候选、修改 Plan 或添加 H100 分支。修复后 4 个候选全部完成，数值最大误差 `3.0517578125e-05`，p50/p95 为 `5.0203/5.3904 ms`。

### 4. 第一轮跨设备结论

89 个 case、267 个 provider-case 在两台机器上分别完成：264 个数值 PASS，3 个 TileLang 明确 unsupported，没有 downstream failure。

同一 search space 在两机选出不同的下层参数，例如：

| Kernel | RTX 5090 D cuTile winner | H100 cuTile winner |
|---|---|---|
| attention | K=32, Q=64, ctas=1, occupancy=2 | K=128, Q=128, ctas=2, occupancy=2 |
| GEMM | K=32, M=128, N=128, occupancy=2 | K=64, M=128, N=128, occupancy=4 |
| online softmax | N=1024, occupancy=2 | N=512, occupancy=4 |
| selective scan | N=512, occupancy=4 | N=1024, occupancy=2 |

逐行最低 generated p50 的赢家分布也不同：

| 设备 | Triton 独胜 | cuTile 独胜 | TileLang 独胜 | 含并列的行 |
|---|---:|---:|---:|---:|
| RTX 5090 D | 23 | 24 | 34 | 8 |
| H100 | 35 | 22 | 31 | 1 |

89 行中有 40 行赢家集合变化。这说明“逐 kernel 在三家取最优”不是一句假设；同时也说明配置不能从一台机器直接搬到另一台。

没有出现“一条结构规则无法同时服务两台机器”的证据，因此没有建立源码形态搜索，也没有架构代号分支。private-buffer budget 的两个经验除数没有被真正扰动，因为两机 registers/SM 恰好相同；这项当时就保留为未被跨设备证明的规则。

## 四、第二阶段：从已有 source 补入 12 个 case

对应提交：`6f40b3d feat(compiler): close source-driven kernel coverage`。

这一轮不自行构造新算法，而是先阅读已经放进 `source/` 的实现，再照其算法结构写 DSL。固定表从 89 增加到 101 个 case。

| 新记录 | 上游结构 | 压到的语言/编译器能力 |
|---|---|---|
| `causal_conv1d` | causal depthwise Conv1D + bias + SiLU | 左负偏移、非对称边界、depthwise reduction |
| `continuous_gqa_decode` | continuous KV 上的 GQA decode | ordered stream、多对一 head mapping、stream 内 contraction |
| `mla_prefill` | content/position 双 score causal MLA | 两个 contraction 共享 stream、在线归一化 |
| `mamba_chunk_scan` | chunk selective scan | 高元融合、组广播、causal scan、前态项 |
| `w4a8_packed` | int8 activation × packed signed int4 | 位运算、符号扩展、循环内 unpack 与 contraction |
| `embedding_forward_lookup` | embedding row gather | data-dependent tensor index 与调用前置条件 |
| `block_scaled_matmul` | 每 32 个 K 元素共享 E8M0 scale | E4M3/E8M0、外层 stream 与独立 inner reduction |
| `splitk_attention_reduce` | partial output/LSE 合并 | max、exp2 权重与加权归约 |
| `fp8_gemm/e4m3` | FP8 GEMM | E4M3 ABI、FP32 accumulation |
| `fp8_gemm/e5m2` | FP8 GEMM | E5M2 ABI、FP32 accumulation |
| `index_select_rows` | row index-select | 一维 data-dependent gather |
| `scaled_index_add` | unique indexed scaled update | data-dependent scatter 与原位更新 |

### 1. structured index 与 data-dependent index 分开

旧 facts 把所有 tensor index 压成一个 `tensor_indirect` 布尔值，导致叶子不知道下标是作者写出的 SSA/quasi-affine 表达式，还是从外部 tensor 读取的数据。

当前共享事实与 `intent_plan.transfer.tensor_indexing` 明确区分：

- `none`：没有 tensor index；
- `structured`：能沿 SSA/use-def 回到逻辑轴与受限整数表达式；
- `data_dependent`：来自外部数据，必须走真实 gather/scatter。

三个叶子直接消费这个分类。structured index 保留作者表达式；data-dependent index 使用目标原生 gather/scatter。没有按 embedding、index-select 或 W4A8 名字分支。

### 2. 同一轴同时承担 ordered 与 reduction

W4A8 首轮暴露了静默错编：stream 已生成全局 K 偏移，同一轴进入 contraction reduction 时又被初始化为局部零，三个后端都重复读取第一个 K tile。

Plan 用 stream axis 与 inner reduction axis 的 node relation明确两种情况：

- 同一节点：保留 stream 的全局块偏移；
- 不同节点：为独立 inner axis 建本地 range。

修复后 W4A8 三后端 generated/reference 误差为 0，continuous GQA、MLA、paged/varlen attention 等所有消费者也重新通过。这里修的是轴角色组合，不是量化 kernel 特判。

### 3. pointwise 结果轴与 staged feature tile 进入 Plan

pointwise result 的逐轴 provenance 进入 `intent_plan.pointwise.axis_nodes`，reshape、broadcast 与 structured index 不再从结果 shape 反猜逻辑轴。

同一轮还发现 cuTile grouped GEMM 的 staged scatter 从外部 view shape 重建 feature index，形成完整 `N=4096` 地址，而值只有 `(128,64)` tile。当前 staged scatter 直接读取 stage plan 的 member routes、`offs_feature` 与 `feature_mask`。三后端 grouped GEMM 的 base、tail 与 empty group 都重新数值通过。

### 4. 类型合同补齐

语言、MLIR type、ABI 与三个 target 补入 E4M3、E5M2、E8M0FNU、u8/i8/i64。Triton 没有 E8M0 tensor dtype时使用 u8 storage，cast op 根据自己的输入类型机械兑现 `2^(x-127)`，不沿 producer 链猜来源。

cuTile 在 H100 SM90 对 E8M0 明确 unsupported；没有改成另一个 dtype 或添加架构型号分支。

### 5. 当时的运行结果

- H100 新增 36 个 provider-case 中 35 个 PASS，唯一边界是 cuTile/SM90 的 E8M0；
- RTX 5090 D 当时 32/36 个已运行格 PASS，另外 4 格受同机外部显存占用阻断，写为 `not_measured`；
- 9 个算法与调用 scope 可对齐的 provider-case 接入真实 source 数字，其余保持空白。

## 五、第三阶段：MLA、FP8 MQA 与 batched contraction

对应提交：

- `cff0e27 feat(compiler): lower current MLA and FP8 operators`
- `e45b84e fix(compiler): project sparse MLA batch contractions`
- `b55f21a docs(report): record targeted cross-device closure`

固定 case 从 101 增加到 106。

### 1. 选入的公开算法结构

| 记录 | 保留的作者算法结构 |
|---|---|
| `mla_head_projection/query_absorb` | 每个 head 将 no-PE query 从 128 维投影到 512 维 latent 空间 |
| `absorbed_mla_prefill` | latent content dot 与独立 64 维 RoPE dot 相加，causal online normalization |
| `mla_head_projection/value_reconstruct` | 512 维 latent output 投影回 128 维 value 空间 |
| `token_sparse_mla_prefill` | 外部 selected-token tensor 驱动多轴 cache gather，同时输出 output/max/LSE |
| `fp8_mqa_logits` | FP8 Q×KV、FP32 MMA、ReLU、head weight/reduction、KV scale 与 range mask |

吸收式 MLA 保留三次作者调用，没有把 query projection、attention 和 value reconstruction 偷偷融合。token-sparse MLA 保留作者的数据依赖 token selection；没有改成 dense attention 以迁就编译器。

### 2. 任意秩 broadcast 与多轴间接 gather

`head_weight[query, head, None]`、`valid_token[:, :, None]` 等 `new_axis` 原本只在少数固定 rank 模式下可发射。共享 relation 现在接受“一个或多个 `new_axis`，其余项为 full slice”的通用合同，三个叶子分别映射 reshape/expand 语法。

token-sparse cache 的访问是 `[Q-tile, selected-token, latent-channel]`，index 来自外部 selected-token tensor。三个目标现在直接从同一个 index relation、transfer boundary 与作者 valid predicate 生成 pointer/gather、bounds 和 fill；不从物理轴重新拼地址。

### 3. bool 运算映射表补齐

sparse selection 和 key range 需要 bool tensor 上的 `logical_and`/`logical_or`。前端和 Kernel IR 已经有该语义，缺的是 capability 与三个目标的逐 op 拼写。本轮补的是映射表，不是新的分析机制，也没有把 bool op 改写成控制流。

### 4. row-vector 与 deferred contraction 只读物理轴

FP8 MQA 首轮生成的 Triton 源码出现未定义 `BLOCK_SIZE`，根因是 row-vector 宽度没有从 Plan 的 physical-axis projection 读取。当前非复用 row-vector 使用 `next_power_of_2(logical extent)` 的物理投影，reuse-worker 轴使用已选 tile。

cuTile deferred contraction accumulator 原先也读取逻辑 `axis.tile`，现改为同一 physical-axis tile。defer 条件只由“单一 contraction consumer、两侧直接外部 load、Plan 决定 shared、boundary共同绑定唯一 reduction axis”这组事实决定，不看周围 kernel 长相。

### 5. `batch` 是 Kernel IR 语义，不是后端猜测

公开 token-sparse MLA 的核心是 batched `Q @ focused_kv^T` 和 `S @ focused_kv`。原 DSL 只能写 broadcast multiply + reduce，三个下层得到巨大逐元素表达式并长时间编译。

前端新增：

```python
I.contract(lhs, rhs, reduce=((lhs_k, rhs_k),), batch=((lhs_b, rhs_b),))
```

它完成四件事：

1. 检查 batch axis 唯一、维度兼容且不与 reduction 重叠；
2. 将 batch relation 写入 canonical `intent.contract`；
3. shared facts 保留 batch axis provenance，axis assignment 不把它误认为 M/N 自由轴；
4. Triton/cuTile direct contract 机械投影为三维 `tl.dot`/`ct.mma`。

TileLang 0.1.13 没有当前模型可映射的 batched GEMM leaf，因此在 emission 阶段明确 unsupported，不展开成逐元素慢路径。

改后单一 Triton 候选已通过数值：output/max/LSE 最大误差分别为 `1.2207e-4`、`6.33e-8`、`9.54e-7`。但完整 autotune 在两机仍超过五分钟；cuTile 单候选也可能触发 `tileiras` 编译超时。因此结论是：算法表达和 direct primitive 投影已经闭合，剩余约束是下层编译成本，状态写 `compile_timeout`。

### 6. 删除三份叶子重建

同一阶段系统审计了 emitter 从 shape、role name 或逻辑轴重建已有物理事实的路径，删除了三类重复：

- private/scan workspace owner extent 不再通过 `owner -> program_order -> roleDimensions` 重拼，直接读 Plan owner node；
- scan workspace extent 不再回 Kernel IR 找 axis，直接读 shared physical binding 的 `axisDimensions`；
- staged ragged member 不再遍历 runtime operation 反推，直接读 `StageAxisOp(role="member")`。

保留的厚路径只有真正的目标拼写：result shape 是 Kernel IR 语义，`BLOCK_SIZE_M/N/K` 等是已选 StageAxis tile 的 surface 名字，不是 target 再次做决定。

### 7. H100 的资源 A/B

`absorbed_mla_prefill` 在 RTX 5090 D 上需要 102400 B shared，而该 kernel 可用 101376 B，无法启动。同一源码在 H100 上直接通过，p50/p95 `0.0556/0.0568 ms`；cuTile 为 `0.2861/0.2878 ms`。

这个 A/B 说明 shared-memory 容量影响候选可实现性，但当前下层已经按设备筛选。没有把容量变成新的算法结构决定，也没有按设备型号选路径。

## 六、第四阶段：并排读三家源码，修叶子而不是扩 Plan

对应提交：`ebb1c3f fix(tilelang): use native transpose and reject scalar contraction fallback`。

### 1. TileLang transpose 原语选错

三家读同一 Plan：Triton 用 `tl.permute`，cuTile 用 `ct.permute`，TileLang 却展开成 `T.Parallel` 下逐元素赋值。TileLang 0.1.13 已有 `T.transpose(src,dst)`，因此只改目标叶子拼写。

- 数值最大误差：0；
- TileLang p50：`0.6053 -> 0.0916 ms`；
- 与 Triton `0.0926 ms`、cuTile `0.0943 ms` 回到同一水平。

### 2. 删除分页注意力的单行 contraction 伪支持

TileLang 原 leaf 将 M=1 contraction 展开成 products fragment、逐元素乘法与 `T.reduce_sum`，形成约 5.9 倍离散度。实际核验 TileLang 0.1.13 后确认没有可用 GEMV/matvec/dot；`T.gemm` 要求 M 可被 16 整除，补 M=16 又与 fragment layout 冲突。

因此删除慢路径，在原 `intent.contract` 源码位置报 unsupported。原 `1.3210 ms` 不再冒充支持，也没有把 TileLang 的 M=16/layout 条件上移到共享 Plan。

### 3. 试过但撤回的方案

- cuTile Viterbi 的 1×1 load 改零维 scalar load：数值通过但无性能改善，撤回；
- dynamic private vector residency：cuTile 编译数分钟仍未完成，撤回；
- 连续去重用 raw store 替代 scatter：与原实现等价，没有解释 3 倍差距，撤回；
- NMS 共享 32-bit bit-pack：Triton 从 27.7411 退到 30.8760 ms，而且逼近另一种算法结构，完整撤回。

这轮的原则不是“叶子越薄越好”，而是：如果叶子只读取 Plan 并映射成目标原语，它可以厚；如果它在重新选择算法、ownership 或 tile，就必须删除。

## 七、第五阶段：反向、多输出与作者多调用编排

对应提交：`2626597 Add source-driven backward kernel pipelines`。

### 1. 注意力反向

作者侧保留三个 kernel：

- `attention_backward_delta`：从 output 与 dO 得到 row delta；
- `attention_backward_dkdv`：以 K/V block 拥有外层工作，跨 Q block 累积 dK/dV；
- `attention_backward_dq`：重算 score/probability，得到 dQ。

这不是把 forward 倒放一遍。ownership 方向改变，中间概率重算而不保存，dK/dV 与 dQ 由不同拥有者计算。调用次数与中间张量由作者的 Python 外层编排，编译器仍一次只编译一个 kernel function。

### 2. 因果卷积反向

DSL 保留 partial gradient kernel 与 reduce kernel，由作者编排两个调用。它压到输入/权重/偏置多结果、跨程序归约和 causal offset 的反向传播，没有让编译器自行决定拆几次 launch。

### 3. block-sparse attention

块选择和 streaming attention 同时存在；索引映射来自作者的 block table，因果/有效范围仍由逻辑读取终点表达。它不因为是“多调用 pipeline”而被跳过。

### 4. 为陌生反向补的共享能力

- source provenance 从单 domain 扩展到多 domain，pointwise operand/shape label 可以传播多个轴来源；
- tensor 可以作为 sequential loop-carried state，facts 检查 carrier/yield provenance 稳定；
- TileLang 支持 tensor loop carrier 的 `T.copy` yield；
- rank-1 unit tensor 到 scalar 的机械读取闭合；
- 同一 contraction operand 被重复使用时，TileLang 将作者已决定的 tile copy 到 shared 并同步，再交给 `T.gemm`，避免多个 fragment layout 互相污染。

5090D 当前结果：

| 记录 | Triton p50/p95 | cuTile p50/p95 | TileLang p50/p95 |
|---|---:|---:|---:|
| `attention_backward` | 0.1584/0.1625 | 0.2041/0.2046 | 0.1781/0.1798 |
| `causal_conv1d_backward` | 0.2607/0.2613 | 0.4736/0.4748 | 0.2658/0.2671 |
| `block_sparse_attention` | 0.1031/0.1063 | 0.0580/0.0647 | failed |

前两条三后端数值通过；block-sparse 的 TileLang 仍是 generated failure，不能算完成。

## 八、第六阶段：分页、split-K 与结构化稀疏

对应提交：`8502f5b Add source-driven sparse and paged kernels`。

### 1. 新增四条独立记录

| 记录 | 上游结构依据 | 使用/新增的能力 |
|---|---|---|
| `varlen_gqa_decode_logits` | 变长 GQA decode + sink/block logits | ordered stream、ragged relation、多对一 head mapping、标量 block maxima |
| `paged_mla_decode` | TileLang paged MLA | 分页间接映射与 state stream 组合 |
| `paged_splitk_attention` | xFormers split-K | 作者侧 partial producer + 既有 reducer 两次调用 |
| `sparse_2to4_gemm` | TileLang `gemm_sp` | canonical sparse contract + target capability |

分页/split-K/GQA 都复用已有 canonical op；只有 2:4 稀疏增加新 op。

### 2. 2:4 sparse contract 的端到端闭环

前端 `I.sparse_contract_2to4` 检查：

- compressed values、metadata、RHS 都是 rank-2；
- compressed/RHS dtype 相同；
- metadata 是 i16；
- format 固定为 `two_of_four`。

Canonical `intent.sparse_contract` 进入 shared facts。Physical Plan 增加 `intent_plan.sparse_contract`，验证 compressed/metadata/RHS shared、accumulator private fragment 等已确定的物理绑定。

TileLang leaf 机械分配 compressed `(M_tile,K_tile/2)`、metadata `(M_tile,K_tile/16)` 与 RHS shared tile，再调用 `T.gemm_sp`。Triton/cuTile 当前 target model 没有同层级稀疏矩阵原语，在 emission 前明确 unsupported；没有 dense fallback。

### 3. ragged member 与 ordered role 的组合

同一个 ragged relation 可能有多个 member domain。此前只给直接标记 ordered 的那个 member 传播角色，分页 relation 与 state stream 组合时可能丢 ownership。当前只要 relation 中任一 member ordered，所有 member domain 在共享 axis decision 中保留 ordered 关联。它依据 relation/role，不依据 paged attention 名字。

### 4. unit tensor scalar projection

private block maximum 等标量值可能以每维 extent=1 的 tensor 存在。Index relation 新增 `extract_unit_scalar`：只有静态全 0 index 且 source 每维 extent 都为 1 时成立。三个目标消费这一 binding；例如 TileLang 用 `T.reduce_sum(source)` 机械抽取标量。

这不是“看到 shape=1 就猜标量”的兜底，而是有明确 index relation 和 Plan binding 的受限投影。

### 5. 5090D 结果

| Kernel | Triton | cuTile | TileLang | 公平 source |
|---|---:|---:|---:|---:|
| `paged_mla_decode` | 0.1163/0.1183 | 0.2219/0.2232 | unsupported | 无：ABI 不同 |
| `paged_splitk_attention` | 0.1709/0.1718 | 0.2738/0.2754 | unsupported | 无：xFormers adapter 不是同 scope 双 kernel |
| `sparse_2to4_gemm` | unsupported | unsupported | 0.2300/0.2318 | TileLang 0.2332/0.2335 |
| `varlen_gqa_decode_logits` | 0.0682/0.0726 | 0.1050/0.1061 | unsupported | 无 |

所有数字是 p50/p95 ms。只有 sparse TileLang 满足算法、调用次数和 scope 对齐，因此只有这一格接 source。

## 九、Source 清点最终结论

`source/` 当前共有 159 个文件。排除 runtime、test、helper、cost model、support plumbing 与同一算法的多 target 拼写后，候选已经全部逐项审视；“尚未审视”数量为 0。

### 已经落成新 DSL 记录

- 本报告第四节的 12 个 source-driven case；
- MLA/FP8 的 5 个 case；
- attention backward、causal Conv1D backward、block-sparse attention；
- 2:4 sparse GEMM、varlen GQA decode logits、paged MLA 与 paged split-K。

### 仍未落地，不能写成“无需处理”

| 上游算法 | 当前最早的真实边界 |
|---|---|
| DeepSeek V3.2 radix top-k | raw bit reinterpret；可移植 workgroup shared mutable buffer、barrier 与 thread/workgroup 语义 |
| vLLM fused top-k/top-p | 作者管理全局 workspace、数据依赖 compaction/control、多轮 pivot search |

它们分别不同于当前 insertion top-k 和独立 nucleus cutoff，必须保留为未落地算法。

### 确认不应独立建记录

- TileLang persistent MLA：与 paged MLA 数学相同，persistence/grid barrier/SM-sized launch 是物理执行方式；
- Liger fused linear cross entropy：Python 外层 chunk/multi-call 编排，inner CE 已有记录；
- `*_runtime.py`、稀疏输入压缩 helper、autotune cost model、import/support plumbing；
- 同一算法在 Triton/cuTile/TileLang 的不同拼写。

因此“source 清点完成”的准确含义是：**0 个未审视候选，2 个已确认独立但尚未落地的算法，其余均已有算法记录或确认不是独立单 kernel。**

## 十、RTX 5090 D 当前全量

统一手动入口：

```bash
examples/run/repro.sh <triton|cutile|tilelang> <runner>
```

实际遍历 104 个 runner 名 × 3 个 provider，共 312 次独立运行。多 case runner 展开后为 113 个 `(kernel,case)`、339 个 provider 单元格。

### 执行级统计

| 状态 | 数量 |
|---|---:|
| PASS | 287 |
| FAIL | 13 |
| TIMEOUT | 1 |
| UNSUPPORTED | 11 |

### case 单元格统计

| Provider | pass | failed | compile_timeout | unsupported |
|---|---:|---:|---:|---:|
| Triton | 109 | 2 | 1 | 1 |
| cuTile | 107 | 3 | 1 | 2 |
| TileLang | 98 | 4 | 0 | 11 |
| 合计 | 314 | 9 | 2 | 14 |

runner 与 case 单元格统计不同，是因为 GEMM、batched GEMM、varlen attention、grouped GEMM、FP8 GEMM 和 MLA projection 等一次 runner 会输出多个 case。

完整日志位于 `/tmp/intentdsl-full-local.DL6gTj/`，没有进入项目。

### 相对上一固定表的 10 个状态变化

| Kernel | Provider | 旧状态 -> 当前状态 | 已知事实 |
|---|---|---|---|
| `cross_entropy` | 三家 | pass -> failed | 确定回归：structured intrinsic 只导入 i16，`arg_reduce.max` 仍使用 i32，前端触发 `NameError` |
| `sorted_nucleus_cutoff` | 三家 | pass -> failed | 与上项相同的 arg-reduce import 回归 |
| `moe` | TileLang | pass -> failed | generated/reference 最大误差 8.7247，上游/reference 正常 |
| `mamba_chunk_scan` | cuTile | pass -> failed | 有 3 个候选运行，但 generated/reference 最大误差 0.1483154297 |
| `continuous_gqa_decode` | TileLang | pass -> unsupported | 删除单行 contraction 串行慢路径后的真实能力边界 |
| `splitk_attention_reduce` | TileLang | pass -> unsupported | 同一单行 contraction 能力边界 |

这里合计 10 个 provider 单元格：两个三后端前端失败、两个单后端数值回归、两个 TileLang 能力状态收紧。

`token_sparse_mla_prefill` 维持 Triton/cuTile `compile_timeout`、TileLang `unsupported`，不是本轮新回归。

### 性能变化

明显变慢的代表项：

- `layer_norm_backward` Triton：0.0969 -> 0.1388 ms（+43.2%）；
- batched GEMM cuTile：NT +13.3%，TT +25.1%；
- `selective_scan` TileLang：+14.8%；
- `w4a8_packed` TileLang：+38.6%；
- `variant_reshape_cache_split` Triton：+39.0%；
- `mla_head_projection/query_absorb` cuTile：+16.3%；
- `ordered_prefix` Triton：+60.3%。

明显变快的代表项：

- `block_scaled_matmul` cuTile：0.1775 -> 0.0703 ms（-60.4%）；
- `roi_align` TileLang：-49.3%；
- `moe_align_block` TileLang：-43.8%；
- `variant_moe_product_domain` Triton：-41.8%；
- `grouped_gemm/empty_groups` TileLang：-36.8%；
- `mamba_chunk_scan` Triton：-28.3%；
- `variant_rope_index`：Triton -22.8%，cuTile -26.9%。

短核的百分比可能只对应数微秒，本轮没有把每个波动归因成 compiler change；CSV 固定实际数字，状态回归单独列出。

### 当前赢家分布

| 赢家 | 上一固定表 | 当前 5090D 表 |
|---|---:|---:|
| Triton | 30 | 37 |
| cuTile | 28 | 27 |
| TileLang | 38 | 36 |
| Triton + cuTile | 2 | 2 |
| Triton + TileLang | 2 | 4 |
| cuTile + TileLang | 1 | 1 |
| 三者并列 | 4 | 3 |
| 无可运行 provider | 1 | 3 |

三家仍各自赢下大量记录，但当前分布包含已知回归，只描述这个代码/运行基线，不单独作为架构优越性的结论。

## 十一、测量口径在这些轮次中如何保持

CSV 的 scope 没有混用：

- `K`：双方预分配 input/output/workspace，计时区间只含相同数量 kernel launch；
- `E`：融合、多调用或 API 无法拆内核时，比较用户取得结果必须付出的相同端到端 GPU scope；
- `R`：算法必须读取 runtime metadata，例如变长 offsets/page table，双方把这部分放在相同一侧。

短 kernel 使用 CUDA Graph 重放；cache flush 大小从设备 L2 查询，不用固定字节数。为适配上游 ABI 的 copy/gather/merge 不会只算在一侧。算法、调用次数或 scope 不同的上游实现保持 source 列为空，PyTorch reference 只做数值正确性，不冒充高性能 baseline。

H100 早期 89-case 主场对照中，41 个可比 source provider-case 以 1% 内为持平：

| 设备 | generated 胜 | source 胜 | 持平 |
|---|---:|---:|---:|
| RTX 5090 D | 20 | 16 | 5 |
| H100 | 18 | 20 | 3 |

上游回到数据中心卡后 generated 优势收窄但没有消失。这证明“当前设备重新生成”不是自动胜利，也没有用三家最优掩盖逐格上游差距。

## 十二、没有保留的错误方向

这些轮次不只增加代码，也删除或拒绝了几条看似能让数字更好、实际破坏分层的方向：

1. 没有为 H100/5090D 或架构代号写分支；设备差异通过属性或下层 tuner 消化。
2. 没有因为两机 winner 不同就建立新的源码形态搜索；尚无一条结构规则被证明必须二选一。
3. 没有把 TileLang 的 batched GEMM、M=16 或 layout 字段搬进共享 Plan。
4. 没有用 products+reduce、串行 joint footprint 或 dense fallback 冒充目标支持。
5. 没有把 attention projection、partial/reduce 或 backward stages 偷偷融合；调用次数属于作者。
6. 没有保留无收益的 cuTile scalar-load/raw-store/private-vector A/B。
7. 没有保留使 Triton 更慢且改变算法结构的 NMS bit-pack。
8. 没有为报告记录 tuner winner 再造一份选择表示；下层没有稳定公开 winner 时就不复制第二份真理。

## 十三、提交时间线

| 提交 | 时间 | 性质 | 内容 |
|---|---|---|---|
| `a51ca74` | 08-13 00:21 | compiler | CUDA Driver API 设备能力查询 |
| `03ab117` | 08-13 00:21 | repro | 跨机器 build/env 接线，移除固定性能失败阈值 |
| `dd0768e` | 08-13 01:34 | target runtime | cuTile autotune 候选失败隔离 |
| `3c0f176` | 08-13 01:54 | report/data | 89-case H100/5090D 跨设备固定表 |
| `6f40b3d` | 08-13 15:02 | compiler/source | 12 个 source-driven case 与共享 index/axis/Plan 闭环 |
| `cff0e27` | 08-13 18:21 | compiler/source | MLA、FP8 MQA、broadcast/gather/row projection |
| `e45b84e` | 08-13 23:53 | compiler | batched contraction 与 token-sparse MLA direct primitive 投影 |
| `b55f21a` | 08-14 00:14 | report/data | 定向跨设备 A/B、表格空洞和 Plan 重建审计 |
| `ebb1c3f` | 08-14 01:15 | target leaf | TileLang native transpose，删除 scalar contraction 慢路径 |
| `2626597` | 08-14 06:11 | compiler/source | attention/causal-conv backward 与 block-sparse attention |
| `8502f5b` | 08-14 10:37 | compiler/source | 2:4 sparse、paged MLA、paged split-K、varlen GQA decode |
| `7025343` | 08-14 11:28 | report/data | 当前 5090D 113-case 全量与 source 清点 |

其中 8 个提交修改 compiler/source 机制，1 个修改 repro 可移植接线，3 个只固定报告/数字。

## 十四、当前没有闭合的地方

### 已定位但尚未修复

- `cross_entropy` 与 `sorted_nucleus_cutoff`：同一个 i32 import 前端回归，影响 6 个 provider 单元格；
- TileLang `moe`：generated 数值回归；
- cuTile `mamba_chunk_scan`：generated 数值回归；
- TileLang `block_sparse_attention`：生成/运行失败，尚未收敛为更窄的 op/Plan 原因。

### 明确 target/downstream 边界

- TileLang 单行 contraction、batched contraction、二维联合 access footprint 和若干 fragment layout 组合；
- Triton/cuTile 当前结构化 2:4 target model；
- cuTile runtime-sized matrix-M FP8 MQA；
- token-sparse MLA 完整 autotune 首次编译成本。

### 语言/算法仍未覆盖

- DeepSeek radix top-k；
- vLLM fused top-k/top-p。

### H100 当前阻塞

H100 现被另一项 vLLM/评测任务占用。本轮等待脚本已停止，没有创建远端 snapshot，没有构建或运行最新 7 条记录，也没有触碰对方进程。

`report/baseline/kernel-performance-h100.csv` 与本轮前快照逐字节一致，仍是此前 106-case 真实数据。它不能回答最新 113-case 代码的跨机器赢家与回归。后续 H100 运行只在人工通知机器空闲后触发。

## 十五、当前文件与提交状态

- RTX 5090 D 当前固定表：`report/baseline/kernel-performance.csv`；
- H100 既有固定表：`report/baseline/kernel-performance-h100.csv`；
- 编译器/source 最后实现提交：`8502f5b`；
- 5090D 全量与上一版报告提交：`7025343`。

本报告用 Git 时间线、当前代码、两份 CSV 和真实 repro 日志重建了整段推进；没有把旧 H100 数字重跑或改写，也没有为了让结论完整而补造未执行的结果。
