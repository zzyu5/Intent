# 第四轮：跨设备全量回归与编译空间收尾

## 结论

这一轮使用同一份代码基线，在 RTX 5090 D 与 H100 上并行启动完整矩阵，依次运行各自的 Triton、cuTile、TileLang provider，并更新 `report/baseline-new/` 下六张固定表。

结果首先说明两件事：

1. **前三轮叠加后没有数值错误。**六张表共 256 个 provider entry，209 个完成 generated/source 首次 launch、数值对照和计时，未出现 `numerical_failed`；当前矩阵也不再有 `physical_program_failed`。
2. **“所有可比项都在 source 的 1.05 倍内”尚未完成。**209 个通过项中 139 个在 1.05 内，70 个超出。Triton 已接近收口；cuTile 与 TileLang 仍有成组的 provider-native form、shared execution policy 和下层首次编译成本缺口。不能把这轮状态写成“成熟完成”，但失败边界与下一阶段的责任层已经可以逐项说清。

这轮没有为了把表做绿而加入 kernel-name matcher、设备型号分支、compiler-private multi-launch、串行伪支持或新的测试脚手架。没有证据支持的临时 A/B 没有进入主路径。

## 一、测量口径与并行执行

### 1.1 唯一代码基线

本机第一次启动时发现使用的是重建前的旧 `intent-compile`，那一批中间结果全部作废。本机随后重新构建当前工作树；H100 使用同步的同一工作树与独立构建。最终六张 `final2` 表都来自当前 compiler，不混入旧二进制或前三轮定向 CSV。

每个 `pass` 的含义不是“能编译”：runner 先执行 generated/source 初次 launch 与 CUDA 同步，再做数值对照，最后分别执行 3 次 warmup 和 100 次计时。只有整条链完成才写入 p50 与 ratio。

### 1.2 并行方式

两台机器同时启动；同一张 GPU 内三个 provider 串行，避免不同下层编译器与 benchmark 互相争抢设备。完成时间提供了独立的并行证据：

| 设备 | Triton 表完成 | cuTile 表完成 | TileLang 表完成 |
|---|---|---|---|
| RTX 5090 D | 09:42:11 | 09:53:14 | 10:13:52 |
| H100 | 09:42:48 | 09:51:43 | 10:06:34 |

两张 Triton 表只相差约 37 秒，随后两机各自继续 cuTile/TileLang；没有等待另一台机器完成。5090 总体比 H100 晚约 7 分钟，主要来自 TileLang 的三个 300 秒 worker timeout。

## 二、全量矩阵

### 2.1 总表

ratio 统计只包含 `status=pass` 且 source 数字存在的行。本轮所有 pass 行都有真实 source 数字。

| provider / device | entries | pass | non-pass | ratio <= 1.05 | ratio > 1.05 |
|---|---:|---:|---:|---:|---:|
| Triton / 5090 | 54 | 51 | 3 | 44 | 7 |
| Triton / H100 | 54 | 52 | 2 | 43 | 9 |
| cuTile / 5090 | 37 | 35 | 2 | 21 | 14 |
| cuTile / H100 | 37 | 34 | 3 | 12 | 22 |
| TileLang / 5090 | 37 | 19 | 18 | 11 | 8 |
| TileLang / H100 | 37 | 18 | 19 | 8 | 10 |
| **总计** | **256** | **209** | **47** | **139** | **70** |

固定表：

- `report/baseline-new/triton-5090.csv`
- `report/baseline-new/triton-h100.csv`
- `report/baseline-new/cutile-5090.csv`
- `report/baseline-new/cutile-h100.csv`
- `report/baseline-new/tilelang-5090.csv`
- `report/baseline-new/tilelang-h100.csv`

### 2.2 相对上一份固定表的主要变化

#### Triton

Triton 的 status 没有退化。前三轮的作者级算法对齐在全量中保住了：

- `paged_mla_decode`：5090 `14.01x -> 0.728x`，H100 `16.27x -> 0.665x`；
- `paged_gqa_decode`：5090 `1.273x -> 0.950x`，H100 `2.564x -> 1.009x`；
- `splitk_paged_attention`：5090 `1.133x -> 0.840x`，H100 `0.862x -> 0.378x`；
- `qkv_projection_pipeline`：5090 `1.060x -> 0.982x`，H100 `0.933x -> 0.869x`；
- `flaggems_roll`：5090 `1.064x -> 1.000x`；
- `flaggems_batch_norm_training`：5090 `1.466x -> 1.340x`，H100 `1.322x -> 1.198x`；
- `mamba3_siso_forward`：5090 `1.784x -> 1.672x`，H100 `2.330x -> 1.952x`。

#### cuTile

- `chunk_gated_delta` 两机由 `physical_program_failed` 变为数值通过；这验证了 runtime domain provenance、ordered role 与 state physical range 的共享修复。
- dense attention / MLA 联合候选的收益在全量保住：H100 `dense_attention_forward 1.789x -> 1.258x`，`mla_prefill 4.635x -> 1.981x`；5090 分别为 `2.312x -> 1.955x`、`2.101x -> 1.671x`。
- `absorbed_mla_decode`、`sliding_window_attention` 明显改善，但仍未闭合；`splitk_mla_decode` 在 H100 仍为 `129.15x`，不能宣称 provider 已成熟。

#### TileLang

以下 entry 从失败变为数值通过：

- `mamba_chunk_scan`：两机 `worker_timeout -> pass`；
- `varlen_block_causal_attention`：两机 `worker_timeout -> pass`；
- `per_token_fp8`：两机从 compiler/provider program 失败变为 pass；
- `fp8_lighting_indexer`：5090 `compiler_invocation_failed -> pass`；
- `mhc_post`：两机从 JIT/initial-launch 失败变为 pass。

以下变化不是退化，而是更精确的能力边界：

- `bitnet_int2_decode` 原来以 `3.29x/4.15x` 的慢路径通过；当前在 provider program/JIT 前明确拒绝，因为 TileLang provider 尚无 logical contraction row `<16` 的 subwarp SIMT GEMV form。删除慢路径比保留“能跑”状态更诚实。
- `block_sparse_gqa_decode` 从下层首次 launch 失败前移到 provider program 的同一 subwarp form 诊断。
- `linear_attention_forward/backward` 从 provider program 失败进入下层 JIT，但所有候选仍无法编译；这说明 shared/provider program 已能表达，剩余边界已下移，不能继续称为语言表达失败。

## 三、本轮全量真正暴露并修掉的共享问题

### 3.1 runtime domain 不再丢失 SSA provenance

此前 runtime-bounded ordinary loop 的 start/stop/step 在 Physical Program 中退化成不可执行的字符串 extent，三个 leaf 只能重建或失败。本轮增加 `intent_plan.domain_extent_binding`：保存 canonical axis node 与 source SSA value node，不复制表达式字符串；common lowering 统一把这段 typed provenance 投影成目标整数表达式。

结果：

- cuTile `chunk_gated_delta` 两机从 `physical_program_failed` 恢复为数值通过；
- 三个 provider 不再各自从逻辑 shape 或符号名猜 runtime bound；
- 非 unit-step 仍由既有 legality 明确拒绝，没有用默认 step 兜底。

### 3.2 boundary fill 的权威来源收回 common proof

masked-load fill 以前在构建 `KernelFacts` 时按 op 顺序立即推导，consumer 尚未全部出现时会误判；GPU `BoundaryNeutralization` 又维护了第二套相似的 domain proof。本轮改为：

1. 先完整建立 KernelFacts；
2. 再对 unresolved masked load 做 fixed-point fill inference；
3. domain-aware neutralization proof 只有 common `Proofs.cpp` 一份权威实现；
4. GPU pass 只查询证明结果、删除冗余 padding 或写入 `consumer_neutralized` 决定。

它覆盖 direct/scaled contract、reduce/scan、shape-only gather 与 state carry，不按 kernel 名识别 attention 或 gated delta。

### 3.3 selected physical range 不再由 leaf 从逻辑结果 shape 回猜

common lowering 增加 `selectedResultAxisPhysicalRange`，三个 materializer 读取同一个已选 physical range。它修掉了 state value axis、pointwise result 与 row-vector width 在不同 leaf 中各自读取逻辑 extent 的分叉。

### 3.4 compiler-created Cartesian grid 的 unique-write 合法性

自动引入的 program axis 如果不被 non-atomic external write 消费，会让多个 program 重放同一个 unique write。本轮 shared decision 对这种轴选择 single-ownership tile；它是正确性约束，不是性能 heuristic，也不读取 kernel 名称。

当前实现仍是保守形态：在 Physical Program 尚无 per-effect ownership guard 时，只能把自动轴收成唯一 ownership。后续若要放松，应先在 Plan 中表达逐 effect ownership，而不是在某个 kernel 上跳过检查。

### 3.5 provider-local legality 只留在 provider

- Triton：descriptor candidate 现在携带具体 block-axis 约束；runtime 只对 descriptor 候选检查 block shape，pointer 候选不受影响。
- cuTile：exact direct-store divisibility 由 cuTile provider form 标记并在 runtime shape 已知时裁剪；动态 ordinary range 显式投影成 `ct.int32`；grid/program binding 读取 `physicalAxisTile`，不再读 raw tile。
- TileLang：role/axis dimension 统一使用目标 spelling；physical result range 由 Plan query 提供；subwarp SIMT GEMV 缺口在 provider 层早拒绝。

这些都没有向 shared Plan 写入 Triton descriptor、cuTile exact-store 或 TileLang thread topology 字段。

## 四、通过但超过 1.05 的完整责任审计

这一节区分三种证据：

- **已闭合原因**：已有同环境 A/B、源码结构对照或跨设备反转，足以说明为什么不应改 shared rule；
- **已定位未实现**：责任层明确，但需要新的 Core/physical/provider form；
- **尚未单项闭合**：只有 family-level 证据，不能伪装成“原因已验证”。

第三类意味着本轮设定的最终性能门槛尚未达到。

### 4.1 Triton

| entry | 5090 | H100 | 定性 |
|---|---:|---:|---|
| `embedding_lookup` | 1.091 | 1.057 | 5090 定向复测 `1.091x`；generated 已是与 Liger 同形的 `program_m x program_n` 2D gather，8--9% residual 尚未由单项 A/B 关闭 |
| `flaggems_embedding_lookup` | 1.079 | 0.996 | 与上一行使用同一 generated kernel；两种 source 结构不同，5090 residual 稳定，不能写成 shared correctness 问题 |
| `padded_rope_cache_update` | 1.984 | 1.221 | 5090 full-run generated 离群；见 4.2。稳定 residual 为约 1.23--1.28x、绝对 5--8 us，effect/address 与 tuner 候选已对齐 |
| `scaled_fp8_splitk_gemm` | 1.703 | 0.918 | 两机 winner 反转；禁止写设备分支或固定 shared rule |
| `mamba_chunk_state` | 1.392 | 1.028 | 5090 绝对差约 5.6 us，H100 接近持平；无稳定 shared 缺口证据 |
| `mamba3_siso_forward` | 1.672 | 1.952 | 已定位：Core 缺 pair join/interleave，generated 只能四个 half scatter/store；TMA、stage、maxnreg、trig 均已 A/B，不能闭合 |
| `flaggems_batch_norm_training` | 1.340 | 1.198 | 已定位：算法已为 Welford，Physical Program 缺 source 的 strip-mined lane-array generic-reduction tree |
| `dense_gemm` | 0.998 | 1.078 | H100 定向复测为 `1.319984/1.322368 = 0.998x`，全量该行是运行态离群，不是 codegen 回归 |
| `flash_attention_forward` | 0.977 | 1.129 | H100 的 tile、TMA、dot、额外 stage 与 `tl.range` 均做过 A/B；没有稳定可保留的单参数/form |
| `mamba3_siso_step` | 1.037 | 1.065 | H100 绝对差约 5.9 us；5090 在门槛内 |
| `flaggems_triangular_solve` | 0.846 | 1.090 | 两机 winner 反转，H100 绝对差约 3.4 us |
| `flaggems_softmax_backward` | 0.792 | 1.096 | 两机 winner 反转，H100 绝对差约 8.1 us |

### 4.2 5090 padded RoPE 的专门复核

全量表记录 `0.075808/0.038208 = 1.984x`。这看起来像本轮 shared ownership 修复造成的退化，因此做了同机交替 A/B：

| 次序 | compiler | generated | source | ratio |
|---|---|---:|---:|---:|
| 1 | current | 0.038240 | 0.030880 | 1.238 |
| 2 | pre-rebuild binary | 0.038032 | 0.030336 | 1.254 |
| 3 | current | 0.041328 | 0.033568 | 1.231 |
| 4 | pre-rebuild binary | 0.038496 | 0.030192 | 1.275 |

两个 compiler 生成的 Physical Program 与 Triton source 逐字相同：grid 都是 `(ceildiv(64, BLOCK_SIZE_M), 48, B)`，B/head ownership 为 one，half axis 为 `pointwise_lane/fixed_64`。所以本轮没有以 kernel 特判回撤 unique-write legality；全量中的 `0.0758 ms` 归为长时间矩阵环境下的运行态离群。固定 CSV 仍保留原始全量值，没有用定向好数字覆盖。

### 4.3 cuTile

| family / entries | 5090 ratio | H100 ratio | 责任与证据 |
|---|---|---|---|
| dense/MLA attention：`dense_attention_forward`, `mla_prefill` | 1.955, 1.671 | 1.258, 1.981 | provider causal-prefix 与 `program_m x stream` 联合候选已实测带来 15--59% 改善；residual 仍是 provider-native form/候选质量 |
| decode/attention：`grouped_flash_decode`, `attention_sink_prefill`, `sliding_window_attention` | 1.329, 1.631, 2.977 | 2.563, 1.412, 2.255 | Plan 已有 stream/ragged/prefix；family-level 证据指向 provider form，但三项未逐项完成独立 A/B |
| `attention_backward` | 1.097 | 1.753 | 多 kernel/atomic ownership 已由作者表达；H100 差距未单项闭合 |
| `grouped_gemm` | 2.496 | 5.568 | 明确 shared execution-policy gap：source 为 single-worker persistent traversal，当前 shared pass 对 ragged contraction 不选择 persistent；不能在 cuTile leaf 特判 |
| `absorbed_mla_decode`, `splitk_mla_decode` | 2.197, 13.484 | 26.288, 129.145 | provider 缺 16-head packing、TMA/MMA operand form、partial layout 与 occupancy 联合 realization；不是 terminal 字符串问题 |
| MHC/Gemma：`mhc_gemm_rms_scale`, `mhc_apply_residual`, `mhc_sinkhorn`, `gemma_prefill`, `gemma_decode` | 2.797, 1.090, 1.113, 0.434, 1.241 | 2.033, 1.364, 1.766, 4.872, 3.962 | 共享 Plan 可表达，但这些 entry 尚未逐项 A/B；不能宣称原因完全关闭 |
| `chunk_gated_delta` | 6.439 | 21.470 | correctness 路径本轮首次闭合；性能尚未审计，是当前最大的新增 provider-performance gap之一 |
| 小型 pointwise/normalization：`swiglu`, `chunked_softmax`, `rope_qk`, `gelu`, `rms_norm` | 均 <=1.05 | 1.110, 1.207, 1.181, 1.076, 1.087 | 只在 H100 超标，尚无共享规则证据；保留为 target/provider tuning residual |
| `official_fmha` | 0.198 | 1.342 | 5090 source 自身约 25 ms，不能把 generated 的比值写成优势；H100 residual 未单项闭合 |
| `moe_expert_projection` | 0.776 | 1.494 | H100 差距未单项闭合，不能套用 grouped-GEMM persistent 结论 |

这里必须明确：cuTile 36/69 个跨设备 pass 行超过 1.05。虽然主要结构族的责任层已经可见，但并非每一行都有独立 A/B，因而性能验收尚未完成。

### 4.4 TileLang

| family / entries | 5090 ratio | H100 ratio | 责任与证据 |
|---|---|---|---|
| `dense_flash_attention` | 1.812 | 1.341 | provider-native attention pipeline/transfer/ownership quality gap |
| `varlen_gqa_prefill` | 2.737 | 2.732 | ordered-ragged attention provider form 未追平 source |
| `block_causal_attention`, `varlen_block_causal_attention` | 3.914, 3.709 | 4.933, 6.946 | 现在数值通过；provider pipeline/physical tiling 仍明显落后 source |
| `gqa_attention_backward` | 4.514 | 3.161 | 作者多 kernel/atomic 分解存在，TileLang provider realization 尚未对齐 source |
| `grouped_gemm` | 2.200 | 2.643 | 直接 region index 已能正确 replay；性能 form 未闭合，不是 structured-indirect capability 失败 |
| `per_token_fp8` | 1.381 | 2.502 | 本轮从失败变为正确投影；FP8 provider form 的性能尚未闭合 |
| `mhc_post` | 1.674 | 2.971 | 本轮从 JIT 失败变为 pass；未做独立性能 A/B |
| `online_softmax`, `mamba_chunk_scan` | 0.980, 0.934 | 1.360, 1.088 | 仅 H100 超标；尚无稳定 shared/provider 归因 |

TileLang 的 18 个超标 pass 行中，大多数属于 attention/GEMM provider program 质量，而不是 TileLang 语言没有原语；但 `online_softmax`、`mhc_post` 等尚未逐项实测定位。因此同样不能宣称 1.05 门槛已达成。

## 五、所有 non-pass 的真实边界

### 5.1 Intent 语言/KIR 实现缺口

- cuTile 两机 `nvfp4_quantize`：源格式需要独立的 scale encoding、group/rounding 与输出 shape 合同；当前没有对应 DSL 实现。
- TileLang 两机 `block_fp4_quant`：缺 typed E2M1 conversion 与 packed-nibble output ABI。
- TileLang 两机 `deepseek_topk_selector`：缺 typed float-bit reinterpret、shared histogram 与 barrier 合同。
- TileLang 两机 `persistent_mla_decode`：缺与 source cooperative persistent algorithm 一致的作者程序；不能由 compiler-private stage 伪造。

### 5.2 shared Physical Program

当前六表没有 `physical_program_failed`。这不是说 shared policy 已经性能最优，而是说明当前 registry 中所有可表达算法都能构造合法、可验证的 Physical Program；上一轮 `chunk_gated_delta` 的 runtime-domain 缺口已经关闭。

### 5.3 Intent provider program/form 缺口

TileLang：

- `block_sparse_gqa_decode`, `gqa_decode`, `varlen_gqa_decode_logits`, `paged_mla_decode`, `native_sparse_attention_forward/decode`：缺 subwarp SIMT GEMV form。ref 已证明 TileLang 本身有二维 thread binding、vectorized K load、thread all-reduce 与唯一 writer；缺的是 Intent provider program，不是 target 语言。
- `mamba_chunk_state`：缺 structured-indirect reduction replay form。
- `sparse_mla_backward` H100：进入 provider program 后仍无合法 target form。

### 5.4 provider JIT / 下层编译成本

- TileLang `linear_attention_forward/backward`, `retention_forward`：provider source 已生成，但所有候选未能完成 JIT/initial launch；不能升级为语言不支持。
- TileLang `mhc_pre`：5090 worker timeout，H100 provider JIT/initial-launch 失败。
- TileLang `w4a8_gemm`, `dequant_bf16_fp4`：两机 worker 超过 300 秒。这个状态只证明首次编译成本超预算，不等于 target capability 缺失。
- TileLang `bitnet_int2_decode`：当前 provider 无 subwarp SIMT GEMV form，明确拒绝旧的 3--4x 慢路径。
- TileLang `fp8_lighting_indexer`：5090 pass；H100 停在 provider JIT，说明 FP8 基础存在、设备/provider 组合尚未闭合。
- cuTile `sparse_mla_prefill`：两机 provider JIT/initial-launch 失败，尚不能写成 shared 或语言失败。

### 5.5 设备资源或 target capability

- cuTile H100 `block_scaled_gemm`：当前 source/provider form 需要 SM100 E8M0 scaled MMA；H100 SM90 不具备该能力。
- Triton 5090 `modern_flash_attention_forward`、`flash_attention_backward`：source kernel 自身需要 163840 B shared memory，设备可用 101376 B；失败发生在 source adapter/initial launch，不是 generated compiler。

### 5.6 source/adapter

- Triton H100 `flash_attention_backward`：source backward 在当前 Triton dtype/toolchain 路径 JIT 失败。
- Triton 两机 `legacy_flash_attention_bias`：vendored source 在当前 Triton 上数值错误，保留 `source_compatibility_gap`。
- TileLang 5090 `sparse_mla_backward`：source adapter 无法把动态 shared-memory 上限设到 231424 B；H100 同 entry 已越过 adapter，随后在 provider program 失败。

## 六、当前编译空间的清晰边界

### 6.1 shared pass 做什么

shared pass 只做下层无法从 target source 自己恢复、且跨 GPU provider 成立的决定：

- program ownership、worker/fold 与 program-space dimensionality；
- ownership/traversal/reduction/lane/access 等 range purpose 与 tile role；
- persistent traversal 是否成立；
- private scalar/vector/workspace residency；
- stream/ragged/partition binding；
- boundary validity、padding identity 与 consumer neutralization；
- unique-write 在 Cartesian grid 上的 ownership 合法性；
- structured contract/reduce/scan 的 physical skeleton。

依据是 typed axes、def-use、producer/consumer、reuse/lifetime、effects、validity、device capability 与当前 Physical Program，不是 kernel 名或 shape 常数。

本轮之前的 A/B 已证明一个仍未解决的 shared policy：全标量 program axes 一维折叠使 paged MLA 在 5090/H100 快约 11.3%/3.7%，却使 H100 paged GQA 慢约 58.4%。因此没有把“一律折叠”提交成规则；当前 facts 尚不足以写出无魔法阈值的统一判据。

### 6.2 tuner 搜什么

tuner 只搜索不改变算法/physical skeleton 的参数：

- tile role 的具体数值；
- Triton `num_warps`、`num_stages`；
- cuTile occupancy/CTA 与联合 tile profile；
- TileLang threads/stages/warp policy；
- provider-local 等价 form/spelling 候选，例如 Triton pointer/descriptor。

合法范围不是完全无约束笛卡尔积：

- static extent 按上界与 power-of-two 约束裁剪；
- runtime extent 由 early prune 裁剪；
- Triton descriptor 按 rank、alignment、stride、contiguity 与具体 block axes 裁剪；
- cuTile exact-store 按 transfer axes 与 runtime divisibility 裁剪；
- provider capability 不成立的 form 在进入 tuner 前拒绝。

provider lowering 自己产生的隐式 shared memory、register allocation、PTXAS/TileIR 资源失败仍由下层真实编译淘汰。Intent 不复制 Triton/TileLang 的隐式资源模型。

### 6.3 provider pass 与 terminal translation

provider pass 可以读取 Physical Program 中的 `exec_*` SSA、relation、effect 与 selected range，派生 target-native form；它不能重新选择 shared ownership、blocking、stream end 或 residency。

- Triton provider 选择 pointer/descriptor、linear/strided transfer、native scaled primitive等 form；
- cuTile provider 选择 prefix boundary、exact-store、program/stream 联合参数 surface；
- TileLang provider 应选择 native MMA/SIMT GEMV、copy/buffer/pipeline form；当前 subwarp、structured replay 与部分 attention form 仍未实现。

terminal translator 只消费这些已选 form 并拼写目标 API。三个 leaf 中仍有较厚的表达式/索引打印，但全量未发现它们重新决定 shared physical structure。

## 七、验证与代码卫生

- 在统一的真实路径 `/mnt/hdd/tmp/intentdsl-final-check-build-20260824` 新建 clean build，`intent-compile` 完成 67/67 个构建步骤；避免了旧 build tree 因 `/tmp` symlink 与 depfile 路径不一致造成的假增量状态。
- 两台机器实际执行六张全量表；没有数值失败。
- 对 5090 padded RoPE 做 current/旧二进制交替 A/B，并逐字比较 Plan/source。
- 对 H100 dense GEMM 定向复测：`1.319984/1.322368 = 0.998x`，排除全量 `1.078x` 为 codegen 回归。
- 对 5090 两个 embedding baseline 定向复测，确认约 `1.091x/1.080x` residual 稳定存在。
- 用 clean-build compiler 再执行一次 padded RoPE 的 emit、JIT、GPU 数值对照与计时，结果继续 pass。
- `git diff --check` 在 CSV 统一为 LF 后通过。
- 自查未发现 kernel-name、具体 shape literal、op-count/whole-region matcher、silent fallback 或第二条 executable path。

手动复现入口仍是唯一 runner，例如：

```bash
PYTHONPATH=python:examples \
  /home/kingdom/.venvs/intentdsl-mlir20/bin/python \
  -m repro.v2.runner triton \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --output /tmp/intentdsl-repro.csv \
  --kernel padded_rope_cache_update
```

## 八、可作为下一阶段起点的真实状态

### 已经做完的核心

- Kernel IR 到 Physical Program 的 runtime-domain provenance、range、validity、padding 与 selected physical extent 能跨三家传递，不再依赖符号名或逻辑 shape 回猜。
- Triton 主路径已经站住：103 个跨设备 pass 行中 87 个在 1.05 内，核心 GEMM/normalization/attention/paged decode 均数值正确；大多数 residual 已有实测责任层。
- 当前 registry 中不再有 shared `physical_program_failed`；失败能落到 Intent 实现、provider program、provider JIT、worker timeout、source/adapter或设备 capability。
- compiler-private multi-launch stage 没有回来；作者需要多 kernel 时由多个 `@intent.kernel` 与外层编排表达。

### 尚未关闭，不能伪装成完成

1. cuTile `chunk_gated_delta`、MLA/split-K/MHC/Gemma 等大量 provider form 性能缺口；H100 的极端 ratio 不能用“下层质量”一句带过。
2. cuTile ragged grouped contraction 的 persistent traversal 是 shared execution-policy 缺口；现有 typed facts 还不足以产生经跨设备验证的统一规则。
3. TileLang subwarp SIMT GEMV、structured-indirect replay、attention pipeline 与 FP8 form 尚未闭合；当前 37 个跨设备 pass 行只有 19 个在 1.05 内。
4. Core 仍缺 Mamba3 所需的 pair join/interleave value-shape op；Physical Program 仍缺高质量 strip-mined generic reduction tree。
5. 70 个超标 pass 行中仍有一部分只有 family-level 责任证据，没有逐 entry A/B；因此本轮的 1.05 性能验收目标没有诚实完成。

这里的“未关闭”已经不是混杂的 `compile_failed`：每项都指向可继续处理的具体层。下一阶段若继续推进，应以这些未闭合层为输入，而不是重新从生成源码逐格添加 leaf 特例。
