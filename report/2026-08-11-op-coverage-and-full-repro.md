# Intent Kernel 编译器：全算子与全后端核验

日期：2026-08-11  
代码基线：`6f72d76`  
核验范围：当前 `examples/kernels/` 的全部 DSL kernel，以及 `examples/run/repro.sh` 声明支持的全部 provider 组合。

## 结论

- 当前语料包含 **34 个源码文件、48 个 `@intent.kernel` 定义、42 个公开 repro 名称**。
- 声明支持的矩阵为 Triton 42、cuTile 42、TileLang 41，共 **125 个有效组合**。TileLang 的 compare-and-swap 因下层没有 CAS primitive，明确不在支持矩阵内。
- 本次从干净提交 `6f72d76` 出发，顺序执行了全部 125 个组合：**108 PASS，17 FAIL，通过率 86.4%**。
- 通过项都完成了 DSL lowering、canonical Kernel MLIR、Physical Plan、目标源码生成、GPU 执行和 reference 数值比较。
- 17 个失败项全部停止在 Intent emission 或下层 JIT 编译阶段；**没有出现 kernel 已执行但数值错误的情况**。
- 42 个 repro 中，**32 个在各自声明支持的全部 provider 上通过**；10 个至少有一个 provider 编译失败。

| Provider | 声明支持 | PASS | FAIL | 通过率 |
|---|---:|---:|---:|---:|
| Triton | 42 | 38 | 4 | 90.5% |
| cuTile | 42 | 35 | 7 | 83.3% |
| TileLang | 41 | 35 | 6 | 85.4% |
| 合计 | 125 | 108 | 17 | 86.4% |

## 核验口径

每个有效组合都使用仓库现有入口：

```bash
./examples/run/repro.sh <triton|cutile|tilelang> <repro>
```

执行过程没有并发占用 GPU，也没有增加测试目录、fixture 或 pytest。每条命令重新经过当前 compiler，并由既有 runner 生成模型级输入、运行生成 kernel、执行数值 reference。全量执行的墙钟时间包含 CMake 检查、下层首次 JIT、autotuning 和 benchmark，不能当作 kernel latency；本报告因此只把它用于判断是否完成闭环，不把不同 scope 的时间混成性能结论。

## 全部 DSL kernel

下表列出当前 48 个 kernel 定义。一个 repro 可能编译多个 kernel、多个阶段或多个 constexpr 变体。

| 算子族 | DSL kernel 定义 | 对应 repro / 组合方式 |
|---|---|---|
| Activation | `swiglu_forward` | `swiglu_forward` |
| Backward | `embedding_backward_atomic` | `embedding_backward_atomic` |
| Backward | `layer_norm_backward_rows`, `layer_norm_backward_reduce` | `layer_norm_backward`，两阶段 partial + reduce |
| Backward | `swiglu_backward` | `swiglu_backward` |
| Contraction | `gemm` | `gemm` |
| Contraction | `bf16_gemm` | `bf16_gemm` |
| Contraction | `quantized_gemm` | `quantized_gemm`，bias + activation + residual + int8 输出 |
| Contraction | `batched_gemm_nn`, `batched_gemm_tn`, `batched_gemm_nt`, `batched_gemm_tt` | `batched_gemm`，四种布局 |
| Contraction | `gated_dual_gemm` | `dual_gemm` |
| Contraction | `weight_only_int4_matmul` | `weight_only_int4`，循环内解包与分组反量化 |
| Convolution | `conv1d_same` | `conv1d` |
| Convolution | `conv2d_same` | `conv2d` |
| Indexing | `shifted_row_copy` | `shifted_row_copy`，带偏移索引 |
| Indexing | `grouped_query_head_add` | `grouped_query_head_add`，多对一头映射 |
| Indexing | `scalar_table_lookup` | `scalar_table_lookup`，张量标量索引 |
| Layout | `matrix_transpose` | `matrix_transpose` |
| Loss | `cross_entropy_forward`, `cross_entropy_backward` | `cross_entropy`，前向 + 反向 pipeline |
| Normalization | `stable_softmax` | `softmax` |
| Normalization | `weighted_layer_norm` | `layer_norm` |
| Normalization | `weighted_rms_norm` | `rms_norm` |
| Normalization | `fused_add_rms_norm` | `fused_add_rms_norm` |
| Normalization | `dropout_residual_rms_norm_forward`, `dropout_residual_rms_norm_backward_data` | `dropout_residual_rms_norm`，可重现 RNG 的前后向 pipeline |
| Normalization | `row_logsumexp` | `logsumexp` |
| Pointwise | `batched_row_affine` | `batched_row_affine`，Cartesian parallel domain |
| Pointwise | `alternating_signed_indices` | `value_select` |
| Pointwise | `paired_sum_product` | `record_fields`，named SSA record |
| Pointwise | `integer_log2_floor` | `scalar_while`，两项 carried state |
| Position | `rotary_embedding_flat` | `varlen_gqa_rope_prefill` 中分别处理 Q/K，再接 attention |
| Ragged | `ragged_grouped_gemm` | `grouped_gemm` |
| Ragged | `moe_expert_ffn` | `moe` |
| Reduction | `row_boolean_reduction` | `boolean_reduction` |
| Sampling | `sorted_nucleus_cutoff` | `sorted_nucleus_cutoff`，scan + arg-reduce |
| Sampling | `insertion_top_k` | `insertion_top_k`，private buffer + nested sequential loop |
| Streaming | `flash_attention_fwd` | `attention` |
| Streaming | `flash_attention_bias_fwd` | `attention_bias` |
| Streaming | `flash_varlen_attention_fwd` | `varlen_attention`，causal=False/True 两个变体 |
| Streaming | `flash_varlen_gqa_prefill` | `varlen_gqa_prefill`；也被 RoPE pipeline 复用 |
| Streaming | `paged_gqa_decode_attention` | `paged_attention` |
| Streaming | `streamed_online_softmax` | `online_softmax` |
| Streaming | `ordered_product_prefix` | `ordered_prefix`，二维 Cartesian ordered domain |
| Streaming | `selective_state_scan` | `selective_scan` |
| Synchronization | `claim_zero_slots` | `atomic_compare_exchange` |

## 全部 repro × provider 结果

状态码：

- `PASS`：生成、GPU 执行和数值对照全部通过。
- `CU-SYM`：cuTile 生成源码使用了未绑定的动态符号 `N` 或 `V`。
- `TR-P2`：Triton 收到非二次幂 `tl.arange` extent。
- `RAGGED-STAGE`：ragged staged contraction 的权重选择或存储空间合同没有闭合。
- `TL-INDIRECT`：TileLang 不能投影该多轴间接读取 footprint。
- `TL-LAYOUT`：TileLang 下层 LayoutInference 找不到合法 layout。
- `N/S`：下层无等价 primitive，编译器明确不声明支持。

| Repro | Triton | cuTile | TileLang |
|---|---|---|---|
| `softmax` | PASS | PASS | PASS |
| `layer_norm` | PASS | PASS | PASS |
| `layer_norm_backward` | PASS | PASS | PASS |
| `embedding_backward_atomic` | PASS | PASS | **TL-LAYOUT** |
| `atomic_compare_exchange` | PASS | PASS | **N/S** |
| `rms_norm` | PASS | PASS | PASS |
| `fused_add_rms_norm` | PASS | PASS | PASS |
| `dropout_residual_rms_norm` | PASS | **CU-SYM** | PASS |
| `logsumexp` | PASS | PASS | PASS |
| `cross_entropy` | PASS | **CU-SYM** | PASS |
| `gemm` | PASS | PASS | PASS |
| `bf16_gemm` | PASS | PASS | PASS |
| `batched_gemm` | PASS | PASS | PASS |
| `batched_row_affine` | PASS | PASS | PASS |
| `quantized_gemm` | PASS | PASS | PASS |
| `dual_gemm` | PASS | PASS | PASS |
| `weight_only_int4` | PASS | PASS | PASS |
| `conv1d` | **TR-P2** | PASS | PASS |
| `conv2d` | **TR-P2** | PASS | **TL-INDIRECT** |
| `selective_scan` | PASS | PASS | PASS |
| `attention` | PASS | PASS | PASS |
| `attention_bias` | PASS | PASS | PASS |
| `varlen_attention` | PASS | PASS | PASS |
| `varlen_gqa_prefill` | PASS | PASS | PASS |
| `varlen_gqa_rope_prefill` | PASS | PASS | PASS |
| `paged_attention` | PASS | PASS | PASS |
| `online_softmax` | PASS | PASS | PASS |
| `ordered_prefix` | PASS | PASS | PASS |
| `moe` | **RAGGED-STAGE** | **RAGGED-STAGE** | **RAGGED-STAGE** |
| `grouped_gemm` | **RAGGED-STAGE** | **RAGGED-STAGE** | **RAGGED-STAGE** |
| `swiglu_forward` | PASS | PASS | PASS |
| `swiglu_backward` | PASS | PASS | PASS |
| `shifted_row_copy` | PASS | PASS | PASS |
| `grouped_query_head_add` | PASS | PASS | PASS |
| `scalar_table_lookup` | PASS | PASS | PASS |
| `matrix_transpose` | PASS | PASS | PASS |
| `boolean_reduction` | PASS | **CU-SYM** | PASS |
| `value_select` | PASS | **CU-SYM** | **TL-LAYOUT** |
| `record_fields` | PASS | PASS | PASS |
| `scalar_while` | PASS | PASS | PASS |
| `sorted_nucleus_cutoff` | PASS | **CU-SYM** | **TL-LAYOUT** |
| `insertion_top_k` | PASS | PASS | PASS |

## 17 个失败的准确归因

### 1. Ragged staged contraction：6 项

涉及 `moe` 和 `grouped_gemm` 的三个 provider。

- Triton 与 cuTile 在 Intent emission 阶段报：`staged contraction requires one expert-selected rank-three weight`。
- 两个 DSL kernel 的 ABI 权重本来都是 rank-3，并以 `weight[group, :, :]` / `weight[expert, :, :]` 选择；当前 staged contract 路径没有把这种选择兑现成 emitter 所要求的 rank-3 expert-selected view。具体是 deferred load、view rank 还是 selection metadata 先丢失，现有诊断尚未分解到单一子条件。
- TileLang 更早报 `staged TileLang contraction has inconsistent operand spaces`；它要求 staged lhs/rhs 为 shared、accumulator 为 fragment，因此除共享的选择问题外，还有目标存储空间投影缺口。
- 三者都没有进入下层 kernel 执行。

### 2. cuTile 动态符号未绑定：5 项

涉及 `dropout_residual_rms_norm`、`cross_entropy`、`boolean_reduction`、`value_select`、`sorted_nucleus_cutoff`。

- Intent compiler 已生成 cuTile 源码。
- cuTile JIT 在 HIR→IR 期间报 `Undefined variable N used`，cross entropy 对应 `Undefined variable V used`。
- 生成 kernel 内存在 `ct.arange(N)` / `ct.arange(V)`，但相应符号没有成为 kernel 参数或 constexpr binding。
- 这是同一个 cuTile kernel-header/symbol binding 问题，不是五个算法各自的缺口。

### 3. Triton 非二次幂 lane extent：2 项

涉及 `conv1d` 和 `conv2d`。

- 生成代码分别包含 `tl.arange(0, 5)` 和 `tl.arange(0, 3)`。
- Triton JIT 明确要求 `arange` 范围为二次幂，因此在编译阶段失败。
- cuTile 的一维、二维卷积均通过，TileLang 的一维卷积通过，说明卷积 DSL 算法与共享索引关系可以成立；Triton 需要把静态窗口 extent 物理扩到 8/4，并继续使用已有有效区间与填充值机制屏蔽扩展 lane。

### 4. TileLang 多轴间接读取：1 项

仅涉及 `conv2d`。

- Intent TileLang emitter 拒绝一个 rank>2、同时含多个 tensor indirect index 的 patch load。
- 当前明确错误为 `multi-axis broadcasted indirect read footprint ... cannot project as one parallel fragment`。
- 该项没有进入 TileLang JIT；这是 TileLang 投影能力边界，而不是数值失败。

### 5. TileLang layout inference：3 项

涉及 `embedding_backward_atomic`、`value_select`、`sorted_nucleus_cutoff`。

- 三项都已生成 TileLang 源码。
- 下层 TileLang CUDA pipeline 在 `LayoutInference` 阶段报 `no available layout found`。
- 生成 kernel 未执行，因此当前只能确认下层布局搜索失败；不能把它归因成数值、算法或统一的 canonical IR 错误。

## Canonical op 覆盖

当前 `IntentOps.td` 定义 **53 个 canonical op**。

三后端已有纵向 handler/结构消费闭环的 48 个为：

`constant`, `dim`, `domain`, `domain_product`, `partition`, `indices`, `region_end`, `assume_in_bounds`, `parallel`, `ordered`, `state_stream`, `if`, `for`, `while`, `yield`, `condition`, `view_load`, `view_store`, `reshape`, `transpose`, `broadcast`, `full`, `zeros`, `make_record`, `extract`, `unary`, `binary`, `compare`, `select`, `cast`, `mask`, `reduce`, `arg_reduce`, `scan`, `contract`, `ragged`, `ragged_outer`, `ragged_member`, `members`, `gather`, `scatter_unique`, `scatter_reduce`, `buffer`, `buffer_load`, `buffer_store`, `atomic_add`, `random`, `return`。

部分后端闭环：

- `atomic_cas`：Triton、cuTile 可运行；TileLang 当前 surface 没有 CAS primitive，明确拒绝。

尚未纵向闭环：

- `call`：frontend 能生成 `intent.call`，但 helper body、Facts、Plan 和 emitter ABI 尚未闭合。
- `break`, `continue`：frontend 与 canonical terminator 已有，Facts/Plan/emitter 尚未消费；Triton 当前表面也没有可核验的等价 primitive。
- `fence`：frontend 与 canonical op 已有，但三个下层没有共同的 ordering/scope 语义，不能降成 no-op 或偷换成单个 atomic ordering。

## 本轮新增闭环

本轮以五个独立提交完成：

| Commit | 闭环 |
|---|---|
| `7f690c5` | Cartesian logical domain product |
| `599f2e0` | scalar atomic compare-and-swap；TileLang 明确 capability rejection |
| `64f0ede` | named SSA record fields |
| `9e12668` | scalar carried-state `while` |
| `6f72d76` | Cartesian `ordered` iteration；逐轴 ordered role 与嵌套串行投影 |

这些新增 repro 的当前结果：

- `batched_row_affine`、`record_fields`、`scalar_while`、`ordered_prefix`：三个 provider 全部通过。
- `atomic_compare_exchange`：Triton、cuTile 通过；TileLang 按真实下层能力不声明支持。

## 当前事实边界

- “有 handler”不等于任意形状都能跑；本次 17 项正是 target capability、symbol binding、staged space/rank 合同尚未覆盖的实例。
- 本轮没有拿失败项的旧结果冒充当前结果，也没有把下层编译失败记为数值 PASS。
- `/tmp` 中的逐项原始日志是本次运行的临时证据，不进入仓库；本报告保留了矩阵、首个准确诊断和分层归因。
