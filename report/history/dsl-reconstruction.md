# IntentDSL DSL 重构

## 最终 public DSL

作者表面现在只承载算法、数据关系、数值与可观察 effect：

- logical domain、runtime subregion、absolute source coordinate 与 endpoint；
- 普通控制流，以及作为迭代独立语义的 `I.parallel(domain)`；
- generic `reduce`/`scan` 与机械 shorthand `reduce.sum/max/any/all`、`arg_reduce.max`；
- typed `region_fold`/`region_scan`，其 source slices、summary schema、combine、identity、state action 与 emit 都由作者显式给出；
- ordinary、scaled 与 sparse contract；scaled format 和 sparse metadata 使用 closed typed schema；
- pure `histogram`；logical buffer、gather/scatter 与完整 atomic family；
- 固定 Philox4x32-10 语义的 `random.bits/uniform`；
- 64-bit logical `index`、tuple/record、标准整数/浮点 tensor element types，以及 carrier 上的显式 packed decode。

`I.auto`、所有 `I.partition`、`I.state_stream`、public `ordered`、旧 `atomic_add/atomic_cas/random`、ordinary `i4/u4/fp4` tensor type，以及作者可见的 atomic scope、layout/alignment/contiguity/vector-width 已从 public 定义与导出中删除。External-view strides、alias 与 noalias 仍是数据解释或调用前置条件，不是 physical mapping hint。

## 语料迁移

完整扫描了 `examples/kernels/` 的 93 个文件和 216 个 `@intent.kernel`。51 个文件需要修改；另外 42 个文件已经只使用最终表面，因此没有制造无意义 diff。规范示例和真实 kernel 使用同一套名称与语义。

HEAD 中 79 处旧 `state_stream` 均逐点分类后迁移：

- 18 处 attention/MLA/paged/variant 路径成为 `region_fold`；作者保留 chunk-local contract、mask、normalization 与 summary merge，compiler-selected segment extent 不可观察。
- linear-attention forward、backward-Q 和 reverse-backward-KV 共 3 处成为 `region_scan`；slice summary、incoming state action、slice-local contraction 与输出 relation 都显式存在。
- online softmax、cross entropy 与 Welford 落到 typed generic reduce；普通前缀与 retention transition 落到 scan。
- Mamba、gated-delta 等非结合递推保留普通有序 loop carry；selective-scan 中原本仅服务 blocking 的 stream 被展开为完整逻辑矩阵 contraction。
- split-K、decode partial 与 host-visible partial ABI 使用显式 part domain、boundary arithmetic、source subregion 和原有多-kernel 编排；没有把可观察 part identity 交给 compiler 重新选择。
- 其余仅需最终完整结果的路径落到完整域 reduce/contract，不保留 physical tile extent。

迁移没有新增或合并 `@intent.kernel`，也没有改变 host-visible pipeline 数量。Causal/window/varlen 的 logical predicates 和 source coordinates仍由作者表达；原 `stop` 中只依赖 compiler blocking 的部分不再作为作者 surface 存在。

## 机械迁移与需数值复核的重写

以下 17 个文件主要是机械 surface/schema spelling 迁移：

```text
backward/embedding.py                 backward/softmax.py
clustering/kmeans.py                  compaction/nonzero.py
compaction/unique_consecutive.py      optimization/adafactor.py
pointwise/batched_affine.py           position/rope_cache.py
reduction/boolean.py                  routing/moe_align.py
routing/mqa_logits.py                 sampling/nucleus.py
sparse/csr_spmm.py                    streaming/ordered_prefix.py
streaming/splitk_reduce.py            synchronization/compare_exchange.py
variants/decomposition.py
```

它们只涉及 namespace、atomic result schema/order、nested parallel、已定义 shape symbol 或删除废弃参数/physical constraint。

以下 34 个文件包含手写算法关系、summary algebra、位级 decode、tail/subregion、数值路径、effect 或 ABI 重写，第三轮第一次真实 GPU 数值验证应优先覆盖：

- summary 与 structured recurrence：`backward/{attention,layer_norm,sparse_mla}.py`、`loss/cross_entropy.py`、`normalization/{batch_norm,softmax}.py`、`ragged/jagged_mean.py`、`routing/mhc.py`、`streaming/{attention,attention_specialized,block_sparse_attention,gated_delta,linear_attention,mamba,mla,online_softmax,paged_attention,selective_scan}.py`、`variants/{normalization,streaming}.py`；
- packed/scaled/sparse：`contraction/{block_scaled,block_sparse,sparse_2to4,weight_only_int4}.py`；
- explicit part/tail/access validity：`backward/causal_conv.py`、`convolution/{direct,varlen}.py`、`reduction/two_pass.py`、`variants/contraction.py`、`vision/{max_pool,max_pool_with_indices}.py`；
- RNG/effect/ABI：`normalization/dropout_residual_rms_norm.py`、`simulation/monte_carlo.py`、`statistics/histogram.py`。

其中 packed INT4 使用 low-nibble-first carrier decode 和显式 sign extension；W4A8 另有 packed tail validity。BitNet source 的 2-bit values 是公开实现中的 unsigned `0/1` 编码，因此保留 unsigned decode而不错误加入 signed two-bit extension。Scaled contract 的 scale tensor是 `u8` carrier，E8M0 scale encoding由 scaled schema定义；sparse 2:4把 external packed metadata显式解为 canonical logical positions。Histogram 按最终规格成为 pure value-producing op，故其旧 `InOut<f32> + atomic_add` ABI 有意收敛为 `Out<f32>` 上写入 `u32` canonical counts 的结果；这不是把旧实现伪装成机械改名。

## Summary validity / presence

38 个 kernel 显式消费 bool-valid summary：

- `streaming/attention.py` 10 个、`attention_specialized.py` 8 个、`mla.py` 7 个、`paged_attention.py` 2 个；
- `variants/streaming.py` 4 个；
- `loss/cross_entropy.py` 3 个；
- `normalization/softmax.py` 1 个、`streaming/online_softmax.py` 2 个、`variants/normalization.py` 1 个。

这些形态集中在 causal、ragged、paged、sparse 或 empty-domain 可出现的 online summary。Attention family 使用 `{valid, maximum, denominator, accumulator}`；softmax/cross-entropy family 使用 `{valid, maximum, denominator[, predicted]}`。Invalid summary 的 maximum 统一规范化为有限 `0.0`，指数 scale 在进入 arithmetic 前由 validity 置零，因此 identity 在左右两侧、empty source 与 physical tail 上都不会形成 `-inf - -inf` 的 NaN。BatchNorm 另有 1 个 kernel 使用 Welford `count=0` 作为 typed presence；其接口前置条件是 `B*S>1`，以保证 unbiased running variance有定义。

## 全量裁决结果

全量扫描中的分叉均已依据最终规格和真实 source 落定，没有留下待用户选择的位置：

- linear attention 的不可观察分段是 `region_scan`；retention 的 element transition 是 ordinary scan；Mamba 与 gated-delta 的 strict recurrence 或可观察 chunk pipeline 不是 region scan。
- split/page partial 的 part identity 与 partial tensor ABI 可观察，使用显式 part domain，不恢复 `partition(count=P)`。
- attention/MLA 的 QK、PV、scaled/sparse contraction 均由作者显式 structured op 提供，不要求 compiler 从算术图猜 whole operator。
- source-specific explicit integer RNG/decode 算法继续作为普通作者代码存在；只有 public `I.random` 迁移到 canonical Philox family。
- provider API 中已经内含的 scale/format含义不反向制造 ordinary narrow tensor dtype，也不把 target spelling写进 DSL。

不存在阻止本轮源码迁移的规格矛盾。

## 横向与收尾自查修复

两次独立静态审计实际修掉了以下问题：

- linear attention 一度被改成逐元素 f32 scan，丢失了作者的 chunk-local contraction 和 f16/f32 round points；最终改为 typed `region_scan`，forward/reverse causal relation均使用 absolute coordinates。
- selective scan 曾把 source row 与 scan column 误写成同一 axis 的对角读取；现恢复独立 row/scan axes，并用 runtime tail extent限制最后 chunk。
- convolution、causal-conv backward、max-pool、W4A8、Mamba、gated-delta 与 varlen convolution 中，padding或ceil-tail曾在 mask 前形成越界地址；现由 source-derived动态 subregion或 safe index + explicit validity 在读取前闭合。
- max-pool-with-indices 在有效值恰为 `-inf` 时，padding可能赢得 tie并返回非法 index；现对 invalid winner选择窗口内第一个合法 logical index。
- biased attention 的 empty slice maximum 曾保留 `-inf`；现与其它 summary 一样规范化 invalid representation。
- 清除了 tuple/cartesian `I.parallel`、tensor predicate上的 Python `and/or`、domain/subregion 误作shape、generic reduce/scan上的旧 `acc_dtype`、stale `HEAD_TILE` constexpr和cuTile source wrapper吞掉的旧 `scale` kwarg。

最终静态状态为：93 个文件全部可由 Python AST 解析，仍为 216 个 kernels；作者源码与 public exports中旧 surface调用为零；public import/export一一对应；registry、三家 provider `CASES`、source runtime路径、required constexpr和output arity未发现断链；diff未触及 frontend、KIR、GPU IR、provider compiler、baseline CSV或 source算法。

## 第二轮必须闭合的 canonical KIR 缺口

本轮按边界没有修改 frontend/KIR。下一轮必须删除旧 executable path，并让最终 surface形成唯一 canonical KIR：

- 删除 frontend/semantics/AST lowering 与 KIR 中的 `auto/partition/state_stream`、旧 atomic、旧单体 random 和 ordinary narrow packed dtype路径；
- 为 generic reduce/scan typed combine、record/tuple accumulator、`region_fold/region_scan` helper regions与source-axis relation建立canonical ops和verifier；
- 接入 `reduce.any/all`、generic sparse contract、histogram、完整 atomic family及CAS record result、Philox `random.bits/uniform`；
- 让 ordinary contract 不再接受假的 `multiply/combine`，让 scaled contract消费正式 format/group/scale schema，不再要求旧 `f8e8m0fnu` tensor dtype；
- 让 external-view metadata只读取最终 strides/alias/noalias，并删除对已移除 layout/alignment字段的访问；
- 让 dynamic subregion、absolute coordinate provenance、active validity、safe indexed access与runtime extents在KIR中保持为typed facts，而不是从shape或邻接结构反推。

第二轮的验收因此是全部93个文件形成并通过canonical KIR；本轮没有保留旧surface alias、fallback或临时wrapper来伪造这项能力。
