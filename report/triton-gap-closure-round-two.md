# Triton 第二轮：算法对齐与剩余差距闭合

## 1. 本轮范围与结论

本轮从上一轮固定的 Physical Program / pass 责任边界继续向下推进，只处理 Triton：

- 对算法分解与 upstream 不同的 entry，修改作者侧 DSL 和调用编排，使算法、调用次数、workspace 与计时 scope 对齐；
- 对 1.05 附近的小差距，并排检查 generated/source，实际试验替代 physical/provider 选择；
- 对两机方向反转或绝对差只有几微秒的 entry，只复测确认，不把设备差异固化成架构分支；
- 对 source/adapter 侧 non-pass，确认失败发生位置，不用参考实现或慢路径冒充 baseline。

结果不是“所有 ratio 都小于 1.05”。更准确的状态是：

1. paged MLA、paged GQA、QKV、roll 已闭合，generated 已不慢于 source；
2. RoPE、BatchNorm、Mamba3 的算法已经对齐，且差距显著收窄；残余分别落在 effect/address 投影、generic Welford physical tree、pair interleave/store form，已经用替代实现实测定位；
3. Flash Attention 在 5090 上已快于 source，H100 仍约 10%；补候选与替换 `range`/`tl.range` 都实测过，未形成可保留的通用收益；
4. 两机反转项与 source 自身失败项维持原定性，没有加入设备分支、kernel matcher 或假 baseline。

本轮是定向验证轮，不是全量轮，因此没有用这些局部结果覆盖六张固定 baseline CSV。所有当前数字集中记录在本报告。

## 2. 作者级算法对齐

### 2.1 paged GQA decode

旧 DSL 是单 kernel/单 head 的在线流，和 vLLM upstream 的 split-K 两阶段、KV-head ownership 与 query-head group 不一致。

本轮改成作者显式编排的两个 kernel：

1. `paged_gqa_decode_partials` 按 `(batch, kv_head, split)` 拥有工作，组内同时处理对应 query heads；
2. stage 1 写 bf16 partial output 与 f32 partial LSE；
3. `splitk_attention_bf16_to_f16_reduce` 做第二阶段 merge，最终输出 f16；
4. adapter 在计时前预分配 workspace，计时区内只保留两次 launch；source 同样为两阶段。

这不是 compiler-private stage，也没有让编译器替作者拆 launch。算法阶段、workspace dtype 与调用次数都由作者侧定义。

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.351968 | 0.371360 | 0.947781 |
| H100 | 0.292848 | 0.304224 | 0.962606 |

同一个 partial kernel 也用于 `splitk_paged_attention`，该 entry 从原来的设备方向反转变成两机都更快：

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.353696 | 0.420832 | 0.840468 |
| H100 | 0.292736 | 0.817584 | 0.358050 |

### 2.2 paged MLA decode

旧 DSL 是单次 launch 的 page/token 双层流；vLLM source 是固定 8 个 token split、stage-1 f32 workspace、stage-2 merge，而且每个 stage-1 program 同时处理 16 个 query heads。

本轮新增 `paged_mla_decode_partials` 并显式接入现有 f32-to-f16 reducer：

- split 数、workspace 和第二次 launch 都在作者编排中；
- stage 1 按 source 的 token split 计算 page-table 地址；
- latent K 同时作为 V，rope 维单独参与 score；
- partial output 与 LSE 均为 f32，stage 2 输出 f16；
- query-head region 为 source 算法中可见的 16-head contraction body，不是 leaf matcher。

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.257616 | 0.356016 | 0.723608 |
| H100 | 0.252160 | 0.365648 | 0.689625 |

原来的约 `14.0×/16.3×` 不是一个难以调平的 provider 常数问题，而是比较了不同调用分解。对齐后该差距消失。

### 2.3 QKV projection pipeline

旧 adapter 把 generated 做成三次独立 GEMM launch，而 xFormers source 是一次 launch 内覆盖三组投影。

本轮 DSL 把 projection 作为显式独立轴，权重 ABI 为 `(P,K,N)`，一个 kernel 内只有一条统一 contract body，输出 `(P,M,N)`；adapter 的 weight stack 在计时外完成。当前 baseline 的 Q/K/V 宽度都为 4096，因此这一 ABI 与当前 source case 对齐；它没有声称覆盖三种不同输出宽度的通用 QKV 接口。

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 1.783936 | 1.810400 | 0.985382 |
| H100 | 1.071840 | 1.258752 | 0.851510 |

### 2.4 BatchNorm training

旧 DSL 用 S-axis scalar state merge，不是 FlagGems source 的 Welford 数值路径。第一版对齐探针曾把 source 的 `BLOCK_N=512` 和 lane arrays 直接写进 DSL；独立审计后否决了该版本，因为 `512` 是 physical tile，不是作者算法语义。

最终 DSL 表达为：

- 作者选择 Welford typed combiner；
- S 轴用 `state_stream` 表达 source 中真实存在的分段/递推；
- 每个 chunk 对 `(B, current-S-region)` 做 multi-axis generic Welford reduce；
- chunk 间用同一 Welford combiner 合并；
- output 是第二次有序 stream，与 source 的统计/写回两遍结构一致；
- physical `S_TILE` 仍由 compiler/tuner 决定，没有把 `512` 固化进编程模型。

这个真实 kernel 同时暴露并修掉三处通用 compiler 缺口：

1. shared reduction legality 原先人为只接受单轴；现在 Kernel IR 的多轴语义可以进入 Physical Program；
2. Triton 原生 `tl.reduce(..., axis=None, combine_fn=...)` 本来就支持 all-axis generic combiner，本轮 provider pass 把“覆盖全部输入轴”机械投影成 `axis=None`；未支持该 form 的其它 provider 仍在自己的 capability pass 拒绝；
3. physical `exec_make_record` 到 `exec_extract` 的 field provenance 以前只识别 KIR op 名，现改为读取 `intent_plan.source_op` 的统一 semantic identity，不再把 physical op 当成陌生 record source。

实测过三种作者/physical 形态：

| 5090 形态 | generated ms | source ms | ratio | 结论 |
|---|---:|---:|---:|---|
| 全 `(B,S)` generic Welford，`axis=None` | 0.061408 | 0.039296 | 1.562703 | full fragment 太大 |
| chunked state-stream + generic Welford | 0.052736 | 0.040272 | 1.309495 | 最终保留 |
| source-visible 固定 512 lane arrays 探针 | 约 0.053 | 约 0.039 | 约 1.35 | 层次错误，已删除 |

H100 最终为 `0.049360/0.040784 ms = 1.210278×`。

因此 residual 已不再是 DSL 写了另一个算法，也不是 generic combine 没投影。剩余缺口是 Physical Program 尚不能把 generic Welford reduction 实现成 upstream 的 strip-mined lane-array physical tree；当前只有 full all-axis collective 与作者可见 chunk recurrence 两种 form。这个结论由三种形态的真实数值/性能 A/B 支撑。

### 2.5 padded RoPE cache update

旧 DSL 有 Q/K/V 三个 effectful branch，source 是先选择输入/输出地址，再执行一份 rotation body。

本轮改为：

- packed Q/K/V 输入；
- 根据 head kind 选择一个输出 row；
- 一份 rotation body；
- value head 通过 select 保留原值；
- 两次统一 store；
- packed input 与 output storage 的 ABI adapter 在计时外完成。

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.042816 | 0.034864 | 1.228086 |
| H100 | 0.036352 | 0.029632 | 1.226782 |

相比约 `1.80×/1.68×` 已明显收窄。`valid_head` mask 在逻辑上恒真，但当前 boundary-neutralization 需要它为 rounded physical lane 提供 fill；删除后两机都会在 Physical Program 阶段明确拒绝，因此保留的是 validity 合同，不是算子特判。

剩余差距为 5--8 微秒。Triton tuner 已实际比较 1/4-warp 等候选，生成侧选择的不是写死值；没有发现另一个稳定更好的物理参数。

### 2.6 Mamba3 SISO forward

本轮按 source 做了以下作者级对齐：

- Q/K 各完整读取一次，再把最后的 pair dimension 分成 even/odd；
- qk accumulation 改为 source 同形的 bf16 contract-with-ones；
- 去掉重复 `dt`/residual-scale 读取；
- source-visible pair rotation 保持四个逻辑结果。

这暴露出一个以前被 rank-preserving gather 误报遮住的 provider form：rank-3 fragment 最后一维为 2，静态读取 component 0/1。最终实现没有把 `static_projection` 变成 shared provider role：

- Common 只保留有界、result-shape 一致的静态投影事实；
- Triton provider pass 进一步要求“最后一维静态长度 2、index 为 0/1”，选择 `tl.split` form；
- terminal emitter 只打印已选 `tl.split`；
- cuTile/TileLang 没有被迫接受 Triton spelling，未实现时在 provider capability 层拒绝。

| 设备 | generated ms | source ms | ratio |
|---|---:|---:|---:|
| 5090 | 0.283760 | 0.170704 | 1.662293 |
| H100 | 0.309248 | 0.156352 | 1.977896 |

为定位 residual，实测了以下替代选择：

- 5090 强制 descriptor/TMA：约 `0.313 ms`，比最终 `0.284 ms` 更慢；
- 只保留 stage-1 pipeline：约 `0.318 ms`，更慢；
- `maxnreg=128`：约 `0.361 ms`，更慢；`maxnreg=256` 与当前约 `0.284 ms` 持平；
- 把 upstream 的 approximate sin/cos 换成与 generated 相同的精确 sin/cos，source 约从 `0.169` 变为 `0.186 ms`，只能解释约 0.017 ms，不能解释剩余约 0.10--0.15 ms。

并排源码后仍存在一项真实算法表面差异：source 用 `tl.join` 把 pair components 重新交错，并用两个完整 descriptor store；当前 Core 只有 split、显式 even/odd scatter，因此 generated 是四个 half-store。现有 Core 无法用另一种自然写法表达 join/interleave。关闭这项需要一个跨 target 有明确数值/shape 合同的新 Core value-shape op，而不是在 Triton leaf pattern-match 四个 store；本轮没有用 target 特例伪造该能力。

## 3. 小差距逐项结果

### 3.1 已关闭

| entry | 5090 ratio | H100 ratio | 处理 |
|---|---:|---:|---|
| `flaggems_roll` | 1.003272 | 0.996159 | DSL 改为 upstream 的 flat 1D iteration；shared pass 为“root store、单轴、无内层 lane 的纯逐元素实例”建立 `pointwise_1d` ownership，tile 进入 tuner；强制 1024 lane 的 A/B 为 `0.088320/0.088064 ms`，证明旧的每实例一 program 是 policy 缺口 |
| `qkv_projection_pipeline` | 0.985382 | 0.851510 | 三 launch 改为作者单 kernel projection axis，见 2.3 |
| `flash_attention_forward` 5090 | 0.975294 | — | 已不慢于 source |

`pointwise_1d` 只决定 pure pointwise scalar instances 的物理 lane packing；带 contract/reduce/scan 的 body 不触发该角色，因此没有恢复曾被否决的自动张量化。

### 3.2 几微秒或两机方向不同

| entry | 5090 | H100 | 结论 |
|---|---:|---:|---|
| `mamba3_siso_step` | `0.098384/0.094304 = 1.043×` | `0.097264/0.091792 = 1.060×` | H100 绝对差 5.47 us；5090 已在门槛内 |
| `flaggems_softmax_backward` | `0.126496/0.159744 = 0.792×` | `0.092624/0.083696 = 1.107×` | 两机 winner 反转，H100 绝对差 8.93 us |
| `flaggems_triangular_solve` | `0.031744/0.037888 = 0.838×` | `0.041360/0.037984 = 1.089×` | 两机 winner 反转，H100 绝对差 3.38 us |

这些项没有产生设备分支，也没有改共享规则去交换两台机器的赢家。

### 3.3 Flash Attention H100 的验证

最终 H100 为 `1.961200/1.783056 ms = 1.099909×`。为确认它不是漏掉一个简单参数或原生 loop spelling，本轮做了两组 A/B：

1. 当前 `(M=128,C=128,TMA=1,8 warps,2 stages)` 为约 `1.965 ms`；新增的 `8 warps,3 stages` 可到约 `1.913 ms`，但仍约 `1.097×`，且 5090 需要 131104 B shared memory、超过 101376 B；把该候选并入完整 tuner 后没有形成稳定的最终收益，已撤回；
2. generated stream loop 从 Python `range` 改成 source 使用的 `tl.range`：5090 基本不变，H100 从约 `1.96 ms` 退到约 `2.05 ms`，已撤回。

因此 H100 residual 不是“缺一个 warp/stage 值”，也不是“没有调用 `tl.range`”。当前 generated/source 都使用 128×128 contraction、descriptor/TMA 候选和原生 `tl.dot`；剩余约 10% 尚未由一个可保留的 shared/provider 选择解释。报告保留该事实，不把一次不稳定的候选写进永久搜索空间。

## 4. 两机方向反转项

| entry | 5090 | H100 | 状态 |
|---|---:|---:|---|
| `scaled_fp8_splitk_gemm` | `4.293120/2.500592 = 1.717×` | `3.161760/3.415136 = 0.926×` | 明确反转，不追 |
| `mamba_chunk_state` | `0.020000/0.014336 = 1.395×` | `0.015424/0.014912 = 1.034×` | 5090 绝对差 5.66 us，H100 接近持平；不写设备规则 |
| `splitk_paged_attention` | `0.840×` | `0.358×` | source-aligned partial 重写后不再反转，已闭合 |

## 5. 三个 source/adapter non-pass

### `modern_flash_attention_forward`

- 5090：source kernel 自身需要 163840 B shared memory，设备可用 101376 B；失败在 source provider initial launch，不是 generated compiler；
- H100：同一 source 可运行，定向结果 generated 约 `0.155712 ms`、source 约 `0.267776 ms`。

adapter 没有额外复制或错误 scope 可修，5090 保持设备资源边界。

### `flash_attention_backward`

- 5090：同样先撞 source kernel 163840 B shared-memory 需求；
- H100：source 在 `_bwd_kernel_dk_dv` 的当前 Triton dtype/toolchain 路径 JIT 失败。

这两种都不通过替换 source 或 PyTorch reference 伪造 baseline。

### `legacy_flash_attention_bias`

vendored source 在当前 Triton 3.6 上能编译运行，但 source 自己的数值对照错误；性质仍为 source compatibility gap，不改写 upstream 算法来凑 pass。

## 6. 本轮新增的通用 compiler 能力

1. **多轴 reduction legality**：shared KIR/Physical Program 接受唯一、非重复、rank 内的轴集合；padding identity 同时作用于所有 reduction axes。
2. **Triton all-axis generic combine**：provider pass 只在轴集合覆盖全部 input rank 时选择 `axis=None`；terminal emitter 机械打印，不理解或改写 typed combiner。
3. **physical record provenance**：record-field resolver 使用 semantic operation identity，同时适用于 KIR `intent.make_record` 与 Physical Program `exec_make_record`，删除按底层 op 名猜来源的错误路径。
4. **有界 static fragment projection**：Common 只确认 source/result rank、静态 index 边界与 projected shape；Triton provider pass 再选择仅自己支持的 final-dim-two-component `tl.split` form。
5. **纯逐元素单轴 ownership packing**：shared `pointwise_1d` role 只在 body 不含区域级结构时成立，具体 lane 数交给 tuner。

这些规则都不读取 kernel 名称，也没有把 Triton API 字段抬到 shared Plan。

## 7. 验证范围

两台机器均实际执行了 DSL → KIR → Physical Program → Triton provider pass → terminal source → Triton JIT → GPU numerical comparison。最终定向覆盖：

- `flash_attention_forward`
- `paged_gqa_decode`
- `splitk_paged_attention`
- `paged_mla_decode`
- `qkv_projection_pipeline`
- `mamba3_siso_forward`
- `padded_rope_cache_update`
- `flaggems_batch_norm_training`
- `flaggems_roll`

另对 `mamba3_siso_step`、softmax backward、triangular solve、FP8 split-K、Mamba chunk state 与三个 non-pass 做了定向复测/A-B。本轮没有跑全量，也没有建立测试脚手架。

可重复的入口形式为：

```bash
PYTHONPATH=python:examples \
  /home/kingdom/.venvs/intentdsl-mlir20/bin/python \
  -m repro.v2.runner triton \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --output /tmp/triton-round-two.csv \
  --kernel paged_mla_decode
```

## 8. 尚未关闭但已定位的状态

| entry | 当前残余 | 已验证的责任层 |
|---|---:|---|
| `mamba3_siso_forward` | 1.66× / 1.98× | Core 缺 pair join/interleave，导致四个 half scatter/store；TMA、pipeline stage、maxnreg、trig A/B 均不能解释/关闭 |
| `flaggems_batch_norm_training` | 1.31× / 1.21× | 算法已是 Welford；Plan 缺 upstream lane-array/strip-mined generic reduction physical tree；三种 decomposition 已实测 |
| `padded_rope_cache_update` | 1.23× / 1.23× | effect schema 已统一；仅剩 5--8 us，tuner 候选已覆盖，未发现稳定替代 form |
| `flash_attention_forward` H100 | 1.10× | tile、descriptor、dot 已对齐；新增 stage 候选与 `tl.range` 均实测不闭合，尚无可保留的通用决定 |

这些不是写在 doc 中的永久“设计事实”，而是本轮代码基线上的实测状态；后续状态只应更新在 report 或下一次全量表中。
