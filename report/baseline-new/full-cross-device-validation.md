# baseline-new 两机全量核验与性能状态

## 1. 本轮结论

本轮在 RTX 5090D 与 H100 上并行执行了 baseline-new 的全部 registry entry，并将最终结果写入六张固定表：

- `triton-5090.csv` / `triton-h100.csv`
- `cutile-5090.csv` / `cutile-h100.csv`
- `tilelang-5090.csv` / `tilelang-h100.csv`

六张表都由当前代码、当前固定 case 和各自设备上的真实 source callable 重新运行得到，没有从旧表复制缺失数字。主流程被单个 source 异常或超长首次编译截断时，后续 entry 使用独立进程继续；最终每个 registry entry 都有一行结果。

结果证明 V2 重构后的主链没有出现跨三个 target 的共同数值错误，但 baseline-new 还不能被描述成“性能已经收敛的成熟编译器”：

- Triton 与 cuTile 的大多数 entry 已能形成真实数值和性能对照；
- TileLang 仍只有 15/30 entry 能完整编译、数值对照和计时；
- 已通过 entry 中仍有若干明确的 Physical Program 结构缺口；
- 少数行的 source 与 generated 使用不同物理分解，只能作为端到端结构参照，不能把 ratio 当成同结构代码质量结论。

## 2. 执行方式与口径

两台机器的全量主流程同时启动，互不等待：

```bash
./examples/run/baseline-v2.sh <provider> <output.csv>
```

H100 使用同一工作树快照、独立 build root 和三套 provider 环境。Triton source 异常会在 factory 构造期直接终止 runner；TileLang `mhc_pre` 的第一个候选编译超过 218 秒、300 秒内仍不能形成可执行候选。为了不让一格吞掉后续结果，本轮对被截断后的 entry 逐项执行同一个 runner：

```bash
python -m repro.v2.runner <provider> \
  --compiler <intent-compile> \
  --output <single-entry.csv> \
  --kernel <entry>
```

逐项执行仍走完整的 DSL → Kernel IR → Physical Program → target source → 下层编译 → 数值对照 → 计时链路，不是替代实现。首次编译 300 秒仍未返回的 entry 记录为 `compile_failed`，不填性能数字。

## 3. 六表全量结果

| provider / device | registry | pass | compile_failed | pass 中 ≤1.05× | pass 中 >1.05× |
|---|---:|---:|---:|---:|---:|
| Triton / 5090 | 31 | 29 | 2 | 18 | 11 |
| Triton / H100 | 31 | 30 | 1 | 22 | 8 |
| cuTile / 5090 | 30 | 30 | 0 | 20 | 10 |
| cuTile / H100 | 30 | 28 | 2 | 11 | 17 |
| TileLang / 5090 | 30 | 15 | 15 | 8 | 7 |
| TileLang / H100 | 30 | 15 | 15 | 6 | 9 |

这里的“≤1.05×”只说明表中计时 ratio；是否属于严格的同结构对照，还要经过第六节的算法和调用边界审计。`ratio` 由 runner 使用未截断的 p50 相除后再格式化，因此不保证恰好等于 CSV 中两列已四舍五入显示值的商。

### 3.1 Triton

5090 上 29/31 通过。失败项：

- `flash_attention_backward`：不是 generated 编译失败，而是 Meta source 在准备 forward state 时要求 163840 B shared memory，超过设备的 101376 B；固定 D128 case 不能通过改小 D 来规避。
- `block_sparse_gqa_decode`：generated 需要 133120 B shared memory，超过 101376 B；H100 同一 Kernel IR/Plan 能通过，说明这是当前候选在设备资源上的合法性缺口。

H100 上 30/31 通过。`flash_attention_backward` 的 Meta source 在当前 Triton 上把 bf16 probability operand 与 fp16 `do` 送入同一次 `tl.dot`，下层以 dtype mismatch 拒绝。两机的失败根因不同，但都发生在 source callable，不能拿其它 source 或 D64 workload 替代后继续声称是原 entry。

### 3.2 cuTile

5090 上 30/30 通过。H100 上 28/30 通过：

- `block_scaled_gemm`：source 与 generated 都依赖 SM100 的 E8M0 scaled MMA；H100 SM90 不具备该能力。这是设备/target 能力边界，不是搜索空间少了一个 tile。
- `sparse_mla_prefill`：所有 generated candidate 都在 tileiras 的 10 秒编译上限内失败，报 `No valid config found in search space`。相同算法在 5090 上能编译运行，因此当前证据指向 H100 工具链的编译成本/候选可编译性，不是 Kernel IR 表达缺口。

### 3.3 TileLang

两台设备都是同一组 15 pass / 15 compile_failed。相同失败集合说明多数问题与设备无关，集中在 TileLang target projection 与共享 Physical Program 尚未闭合的结构上。

| 失败 entry | 当前最早失败点 | 性质 |
|---|---|---|
| `block_sparse_gqa_decode` | query group 的 matrix-M 为 4，TileLang MMA 要求 16 的倍数 | target layout / ownership packing 缺口 |
| `gqa_decode` | 18 个 target candidate 全部未通过编译或验证 | target candidate / 下层质量边界 |
| `varlen_gqa_decode_logits` | single-row contraction | TileLang 0.1.13 无高质量机械投影 |
| `paged_mla_decode` | single-row contraction | 同上 |
| `conv2d` | contraction 通过 tensor loop carrier 重复累加 | target accumulation projection 缺口 |
| `deepgemm_fp8_2xacc` | reduction pair 未保留唯一 region identity | 共享 contraction provenance 缺口 |
| `linear_attention_forward` | broadcast 的 tensor-shape region 没有已选 TileLang tile binding | target range consumption 缺口 |
| `retention_forward` | `T.Parallel` 中 layout infer conflict | target layout realization 缺口 |
| `mhc_pre` | 300 秒内不能完成候选首次编译 | 下层编译成本，不是假能力声明 |
| `varlen_block_causal_attention` | runtime-bounded domain 没有 ordinary scalar sequential loop | 共享 Physical Program 缺口 |
| `native_sparse_attention_forward/decode` | single-row contraction | TileLang 0.1.13 能力子集 |
| `gqa_attention_backward` | contraction 通过 tensor loop carrier 重复累加 | target accumulation projection 缺口 |
| `fp8_lighting_indexer` | FP8 contraction 的 matrix-M 是 runtime lane extent | target MMA 能力子集 |
| `grouped_gemm_backward` | ragged reduction axis 无 masked bulk-copy 投影 | target ragged contraction 能力子集 |

single-row contraction 没有退回逐元素标量归约。历史结果已经证明“能跑但慢一个数量级”比明确 unsupported 更有害；当前 leaf 在没有原生矩阵/归约投影时直接拒绝。

## 4. 本轮修复的真实问题

### 4.1 恢复固定 workload 与 source provenance

最终表恢复并坚持原合同：

- FlashAttention forward/backward 使用 D128，不用 D64 绕开资源问题；
- forward 仍对官方 Triton fused-attention source，backward 仍对 Meta Applied-AI `flash_bwd`；
- TileLang block-sparse GQA 保持 KVH8，不改成 KVH2 来凑 MMA 的 M=16；
- native-sparse 保留作者原来的二维 block-index 表达，不改写成只取第 0 个元素。

H100 官方 Triton forward 的 source runtime 原先默认打开 warp specialization，而上游测试在非 Blackwell 设备明确使用 `False`。adapter 现在按设备 capability 使用上游已有选择：5090 开启、H100 关闭。它只选择 source 自己支持的 target mode，不改变 generated 算法。

### 4.2 静态 reshape 的精确 provenance

native-sparse 的 `[1, BLOCK, D] → [BLOCK, D]` reshape 暴露了一个共享分析缺口：分析在读取 reshape source 之前，就要求结果中的静态 `64/128` label 已全局绑定到逻辑轴。

修复没有恢复“按相同 extent 猜一根轴”的旧 fallback。现在只有 reshape 局部允许正静态 extent 作为临时占位，随后必须按 source use-def 顺序与精确元素乘积完成合并；最终轴身份仍来自 reshape source。符号 label 不能精确解析时仍然诊断。

### 4.3 TileLang 多维 `assume_in_bounds`

作者已在 Kernel IR 中声明二维 key-index tile 对目标 view 合法，TileLang leaf 却把 `assume_in_bounds` 限死为 rank-1。rank-1 单元素确实需要额外抽出标量名字；rank>1 tensor 只需保留原表达并消费假设，不需要生成新的计算。

修复把“假设是否受支持”和“是否需要标量化”分开。两个 native-sparse entry 因而继续推进到真正的 single-row contraction 能力边界。

### 4.4 entry-block 常量 SSA 绑定

TileLang emitter 原先对 entry block 常量直接早退，没有为后续 use 建立 value binding。现在常量先完成类型/属性检查并形成表达式，entry block 只是不打印语句，但会绑定 SSA result。该修复不依赖具体 kernel。

### 4.5 row-vector 最终物理范围

TileLang 的 row-vector range 过去只有 active role 的局部读取被替换成最终 physical extent，其他按 purpose/level 读取同一范围的消费者仍可能拿到逻辑名字。现在在 target materialization 开始时统一把所有 row-vector range 更新为已选 block extent，后续 leaf 只读同一份绑定，不再从形状重建。

### 4.6 可重算结构事实不进入 Plan

“contract result 与前一轮 tensor carry 相加并再次 yield”是从 Kernel IR 唯一推出的结构事实，不是多个合法方案中的选择。本轮把这项识别从 TileLang 私有、依赖 `python_add` 拼写和 emitter 临时名字的匹配，收敛成共享的派生查询。Plan 没有新增第二份算法字段；TileLang 只用这个查询声明当前 target 无机械投影。

### 4.7 其它闭合

- fully-static kernel 可以有 tunable parameters 而没有 specialization dimension；`intent_plan.autotune` 现在允许空 key，但 parameter 仍必须合法。
- contraction 的 M/N ownership 仍要求来自不同逻辑轴；当前 Plan 只有一份 ownership range，不能在没有额外表示的情况下把同一轴静默物化成两份独立矩阵坐标。
- TileLang ragged contraction reduction 在没有 masked bulk-copy 投影时提前拒绝，不再生成忽略 group offset 的静默错误代码。
- MHC GEMM+RMS 的 DSL 显式让 body 看见 token×column region；没有把逐 token vector×matrix 自动升级为 block GEMM。是否使用 split-K、多 stage 和 workspace 仍是待实现的 Physical Program 结构决定。
- Mamba chunk-state 使用真实的累计负衰减输入。gated-delta 的 query/key 使用相同的有界量级；原 adapter 只缩放 query，未缩放 key，在固定 S2048 case 上 generated 与 source 都有超过 95% 的输出成为非有限值，且溢出发生位置不同，无法形成有效的数值对照。该修正不改变 shape、dtype、算法或计时 scope。
- TileLang MHC 的 source-only bf16 rounding 与 residual-matrix transpose 位于计时外，用于对齐 source ABI；generated 输入和 DSL 算法没有改变。

## 5. 修复后的定向回归

共享 reshape 修复后，两台机器都重新执行了：

- Triton `rope_qk`
- cuTile `rope_qk`
- TileLang `w4a8_gemm`

六个结果全部数值通过。native-sparse forward/decode 也在两机重新执行，均稳定到达 single-row contraction 的同一显式诊断。没有新增 test 目录、pytest 或 fixture；验证仍是可手动执行的 baseline repro。

## 6. 性能状态与归因

### 6.1 Triton

最大的严格性能信号：

- `rotary_embedding`：5090 1.38×、H100 3.29×。upstream 明确联合打包 sequence/head，而 generated 仍使用通用 B/S/H ownership；这是物理 program mapping 缺口。
- `mamba_chunk_state`：5090 1.68×、H100 1.34×；`mamba_chunk_scan` 在 H100 1.33×。upstream 有 head-block、warp/stage family，当前 Plan 没形成相同 target-local 参数空间。
- `scaled_fp8_splitk_gemm`：5090 1.30×、H100 0.94×。跨设备 winner 反转，说明 tile/warp/stage 是参数性选择，应交给下层实测，不能写设备型号分支。
- FlashAttention forward：5090 1.12×、H100 1.53×。算法一致，但 source 有成熟的 block/warp specialization；generated 尚未达到同等 target program 质量。

`paged_gqa_decode` 与 `paged_mla_decode` 的大 ratio 不能只归为 leaf 拼写：generated 是单条 page/token online stream，上游 vLLM 是 split-KV partial + reduction 两阶段，并包含 query-head packing、MLA 512+64 分离和专用 warp/stage。MLA 的 15–16× 说明 Physical Program 缺少真实结构选择；由于调用分解不同，这两行不能作为“同结构慢 15×”来解读。

### 6.2 cuTile

明确的 Physical Program 缺口：

- `absorbed_mla_decode`：5090 12.76×、H100 110.65×；
- `splitk_mla_decode`：5090 9.83×、H100 63.32×；
- `recurrent_gated_delta`：3.79× / 4.00×；
- `mhc_gemm_rms_scale`：3.23× / 3.56×。

MLA source 使用完整的 tile/head packing、TMA、persistent/split topology；generated 没有形成等价物理 skeleton。gated-delta source 在 standard/persistent、BLOCK_V、TMA 和 occupancy 间联合实测。MHC source 是 split-K partial + finalize 两阶段，而 generated 仍是一份物理 kernel；这里需要 stage decision，不是再堆一个 tile 常数。

窗口注意力的剩余差距来自遍历边界：generated 仍进入部分完全位于窗口外的块再 mask，source 在块级直接收紧 lower/upper traversal。attention backward 同理，source 为 dK/dV 选择因果下界，generated 遍历完整 Q 再中和。

同时必须看到 source 的设备配置质量并不稳定：cuTile `dense_gemm` 的 source 在 H100 是 13.22 ms、在 5090 是 2.47 ms；official FMHA source 在 5090 是 32.20 ms、H100 是 2.56 ms。ratio <1 不能自动宣传成 Intent 普遍更快，它也可能说明公开 source 的固定配置在该设备上失配。

### 6.3 TileLang

已通过项中的主要差距：

- `block_causal_attention`：2.97× / 4.66×；
- `block_sparse_gemm`：1.98× / 3.16×；
- `w4a8_gemm`：1.81× / 2.00×；
- `varlen_gqa_prefill`：1.52× / 1.90×；
- `mamba_chunk_scan`：1.16× / 3.95×。

这些 entry 的 generated/source 都已经使用 TileLang 原生 copy/GEMM/reduction 语法。剩余差距集中在 target-local tile、pipeline、layout、register/shared resource realization，以及 source 固定 specialization 的设备适配，不应把 TileLang source 的字段表上移成共享 Plan 字段。

## 7. 尚未关闭的问题

### 7.1 必须由共享 Physical Program 补齐

1. split-KV / split-K 多 stage program 的结构选择与 intermediate 生命周期；
2. query-head/KV-head 联合 packing、MLA latent/rope 分段；
3. ordered/ragged stream 的上下界收紧与 reverse traversal；
4. runtime-bounded domain 的合法 scalar sequential mapping；
5. contraction reduction pair 在复杂 state/FP8 组合中的唯一 region identity；

### 7.2 TileLang target 能力子集

1. 高质量 single-row contraction；
2. tensor loop-carried contraction accumulator；
3. runtime-lane FP8 MMA；
4. ragged reduction masked bulk copy；
5. block-sparse GQA 的跨组 M packing；
6. 部分复杂 state-stream 的 layout realization。

这些能力没有用串行慢路径冒充支持。

### 7.3 下层或 source 工具链

1. Meta FlashAttention backward 在两机当前工具链各有不同的 source compile failure；
2. H100 cuTile sparse MLA 的 tileiras candidate 编译超时；
3. TileLang `mhc_pre` 首次候选编译成本超过可接受范围；
4. 部分 target autotuner 候选会崩溃或全部无效，隔离 worker 能保护后续 candidate，但不能让无合法候选的 entry 变成 pass。

## 8. 当前状态

baseline-new 的六张运行矩阵已经完整产出，V2 重构后的主链在两台机器和三门 target language 上得到了真实压力验证。它现在可以诚实说明“哪些算法已经可编译、数值正确并有性能数字”，也可以精确说明未通过项断在哪一层。

但完成表格不等于性能闭环：Triton/cutile 的若干真实结构选择仍缺，TileLang 的通过率仍为 50%，且有几行不是严格同物理分解。后续若继续推进，应以第七节的具体边界为单位；不能用改 workload、换 source、缩小 case、串行 fallback 或 kernel-name 分支把表做绿。
