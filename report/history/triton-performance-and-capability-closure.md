# Triton 性能与能力闭合报告

## 1. 本轮范围与结论

本轮只修改和验证 Triton 路径，没有修改 cuTile、TileLang、public DSL、Kernel IR、shared GPU pass，也没有扩 registry。

本轮解决了两处确定的 Triton 生成质量问题：

- `flash_attention_forward`：5090 从 `2.306392x` 收到 `0.976313x`；H100 从 `2.305503x` 收到 `1.124511x`。5090 已超过 source，H100 的结构性大差距已经消失，仍剩约 12.5%。
- `flaggems_max_pool2d_with_indices`：5090 从 `1.576563x` 收到 `0.776563x`；H100 从 `2.604853x` 收到 `0.945614x`。

同时，`flaggems_histogram` 重新测得 5090 `1.000729x`、H100 `0.999741x`，此前的 `1.21x/1.18x` 不是当前稳定差距，没有为它添加任何新路径。

Triton 当前仍不能诚实地说“所有可比项都在 1.05x 内”。剩余项中有三种不同性质：

1. baseline 与 generated 的算法分解或调用数不同，不能作为同结构 kernel 差距；
2. 已定位到通用 provider form 或 shared blocking 缺口，但本轮没有一个足够通用且安全的实现；
3. 两台设备方向反转，或绝对差只有数微秒，当前证据不支持改一条共享规则。

## 2. 实现改动

### 2.1 逻辑读取终点读取精确 Plan range

`intent.region_end` 以前在 Triton materializer 中直接返回逻辑轴总长度。作者在 attention 中写下的是“当前 query region 的末端”，但目标源码得到的是整个 `Q`，因果 K-stream 因而没有按 query block 收紧。

现在 emission 读取 `selectedRegionValueRange` 和该 region 的精确 physical start：

- ownership/traversal region 的末端统一为 `min(selected_start + selected_tile, logical_end)`；
- ragged axis 使用 `sequence_end` 作为 logical end，再与当前 selected range 相交；
- 找不到精确 selected start 时直接诊断，不从轴名、逻辑 shape 或 program 顺序反猜。

这属于唯一正确语义的投影修复，不是 attention 特判。

### 2.2 Triton transfer form：pointer 与 tensor descriptor 成为 provider-local 候选

Triton provider pass 现在给满足严格矩形条件的 transfer 写入：

```text
intent_plan.triton.transfer_form = pointer | pointer_or_descriptor
```

当前 descriptor form 的合法子集是：

- `view_load`/`view_store`；
- rank 至少为 2；
- 最内层是完整连续 slice；
- 只有一个非标量 block axis；
- 其它维度是静态索引或精确标量索引；
- fill 只能是 none/zero；
- runtime view 连续、16-byte 对齐且末维字节数满足 descriptor 对齐。

provider pass 记录 form，terminal materializer 只机械发出 pointer/`tl.make_tensor_descriptor` 两种 target 语法；`USE_TMA` 作为 Triton-local 候选交给下层实测，不按 GPU 架构名分支。

实测 winner 正好说明这种组织是必要的：

- 5090 上 FlashAttention 的 descriptor 形态更慢，tuner 选择 pointer；
- H100 上 descriptor 明显更快，tuner 选择 TMA。

同一份算法与 shared Plan 因设备不同选择不同 target form，没有把设备经验常数写进 shared 层。

### 2.3 因果 stream 的 prefix/boundary form

上游 FlashAttention 把 K-stream 拆成完全可见的 off-band prefix 和需要因果谓词的 boundary 部分。此前 generated 在所有 K-block 上都发逐元素因果 mask。

Triton provider pass 现在通过 typed use-def 条件选择 `prefix_boundary` form：

- stream 的 stop 是 ownership region 的 `region_end`；
- compare 两侧的唯一 provenance 分别是该 ownership axis 和 stream traversal axis；
- compare 结果只通过明确的 `intent.mask` 消费。

Plan 上记录 boundary axis 与可中和的 mask node。materializer 对 prefix loop 只中和这些已记录 mask，再机械 replay 同一个 stream body 生成 boundary loop。没有 kernel 名、attention 名或 op 数量判断，也没有改写 Kernel IR。

### 2.4 Triton 参数候选补齐

Triton tuner 增加了两类此前缺失的合法参数点：

- 纯 pointwise lane 的 32/64-lane、1-warp 候选；
- `program_m + program_n` 的 8/16/32/64 小二维 tile 候选；
- `program_m + stream_contract` 的成组候选。

MaxPool 的旧 winner 被迫使用过大的通用二维 tile；加入小二维候选后，两机都收敛到 source 以内。候选按 physical role 集合声明，不按 kernel 名选择。

### 2.5 删除无意义的恒真 gather 包装

`gather` 的 validity 已经是编译期 `True` 时，不再发出 `tl.where(True, value, fill)`。这只是 target 源码化简，不改变索引、validity 或填充值语义。

## 3. 超标项逐项归因

| entry | 5090 当前 | H100 当前 | 结论与责任层 | 本轮处理 |
|---|---:|---:|---|---|
| `flash_attention_forward` | `0.976313x` | `1.124511x` | 原缺口同时位于 region-end 投影、Triton descriptor form 和 causal stream form | 三处均已实现；H100 仍剩约 12.5% |
| `paged_mla_decode` | `14.007038x` | `16.271123x` | generated 是作者写下的单 kernel 双层 state-stream；source 是 split-K partial + reducer 的多 launch 算法。是同结果的端到端比较，不是同结构 kernel 比较 | 不把 compiler-private multi-launch stage 加回来；该 ratio 不作为单 kernel 生成质量结论 |
| `paged_gqa_decode` | `1.273124x` | `2.563656x` | 同样是单 kernel online reduction 对 source split-K pipeline | 保留端到端数字，撤销“单一 provider form 缺口”的错误归因 |
| `mamba3_siso_forward` | `2.006445x` | `2.382852x` | 算法一致；source 使用 sequence×feature 的二维带步长 descriptor。当前 provider form 只支持一个 block axis + 连续末维的矩形 transfer | 已钉死为 Triton provider-program 表示缺口；未用 materializer matcher 强行放行 |
| `flaggems_max_pool2d_with_indices` | `0.776563x` | `0.945614x` | Triton 参数空间缺少小二维 output tile | 已修复 |
| `padded_rope_cache_update` | `1.802877x` | `1.681939x` | source 先选择 q/k/v pointer，再执行一份 rotation body；DSL 明确写了三路控制流，generated 保留三路 branch。缺的是带 effects 的通用 Triton if-conversion form | 未按 kernel 名折叠；保持真实缺口 |
| `scaled_fp8_splitk_gemm` | `1.709973x` | `0.935723x` | 两边都是 4-way split-K、64×64×256、8 warps、3 stages、FP8 `tl.dot`、最终 atomic。generated 的 winner 与 source 配置相同；差距在两机方向反转 | 无证据支持修改 shared policy；记录为 SM120 上 Triton rank/address/atomic lowering 的设备敏感差距 |
| `flaggems_batch_norm_training` | `1.453921x` | `1.315288x` | source 的 body 看见 B×S tile；DSL 的算法结构是 C ownership + S state-stream，当前 Plan 没有独立 B blocking。要改变需 shared physical mapping，不是 Triton 拼写 | 本轮禁止只为 Triton 修改 shared 决定，未动 |
| `mamba_chunk_state` | `1.338330x` | `0.956740x` | 5090 绝对差约 5 微秒，H100 generated 更快；两边原语/配置已对齐 | 不建立跨设备方向相反的规则 |
| `flaggems_histogram` | `1.000729x` | `0.999741x` | 旧超标不能稳定复现 | 无代码特判，CSV 更新为当前实测 |
| `splitk_paged_attention` | `1.132925x` | `0.862326x` | 两边均为 two-stage，5090 约 56 微秒差；H100 generated 更快。cache ABI、partial layout 与 reducer form 不完全相同 | 未找到跨设备成立的 provider/shared 修法 |
| `flaggems_triangular_solve` | `0.811153x` | `1.087838x` | H100 绝对差 3.3 微秒，5090 generated 更快 | 不为微秒级方向反转添加 ordered-loop 特化 |
| `flaggems_softmax_backward` | `0.795272x` | `1.108896x` | H100 绝对差 9.1 微秒，5090 generated 更快 | 不为微秒级方向反转添加 reduction 特化 |

`mamba3_siso_forward` 在一次 5090 进程中 source 出现过 `0.299 ms` 的离群值；独立重跑恢复为 `0.168816 ms`，表中采用独立复测结果，没有用离群值掩盖差距。

## 4. provider-native 能力探测

### 4.1 TMA / tensor descriptor

检查了 Triton reference 中 descriptor-based FlashAttention，并对 generated source 做了 pointer/TMA A/B。TMA 对 H100 有收益，对 5090 没有收益，因此最终实现保留为 provider-local autotune candidate，而不是静态打开。

当前没有把 Mamba3 强行塞进同一个 form。Mamba3 需要的不是放宽一个布尔 capability，而是显式 provider facts：

- 原始 N-D view 到 2D descriptor subview 的 axis projection；
- 两个 descriptor 维度的 logical extent、block extent 和 origin；
- ABI 的真实 stride；
- tensor-valued broadcast index 与 descriptor coordinate 的对应；
- 对齐和 innermost-contiguity legality。

这些事实尚未存在。直接在 materializer 里从 `value_index` 周边结构重建会重新制造隐藏编译器，所以本轮没有这样做。

### 4.2 warp specialization

把同一 generated loop 改成 `tl.range(..., warp_specialize=True)` 做过一次性探测。5090 的 Triton 在 automatic warp-specialization pass 中失败，说明当前 emitted loop 形态不能把它当作合法通用候选；该探针已删除，代码中没有保留开关或 fallback。

### 4.3 `tl.range`

单独把两个 state-stream loop 从 Python `range` 改成 `tl.range`：

- 5090：`2.687456 ms` → `2.689024 ms`，无收益；
- H100：`1.964080 ms` → `2.050528 ms`，且 tuner 从 TMA winner 退回 pointer。

因此没有保留这条写法。

## 5. 四个 non-pass 的最终状态

| entry | 5090 | H100 | 定性 |
|---|---|---|---|
| `block_sparse_gqa_decode` | `133120 B > 101376 B`，provider first launch 失败 | `0.900851x`，pass | 逻辑 sparse block/page 固定为 128；直接改 64 会改变页映射算法。合法修复需要 provider physical sub-tiling/replay，不是换一个现有 tuner 常数 |
| `modern_flash_attention_forward` | source preparation 需要 `163840 B > 101376 B` | `0.524335x`，pass | 5090 是 source kernel 资源边界，generated 不是失败方 |
| `flash_attention_backward` | source preparation 同样需要 `163840 B` | source backward JIT 在当前 Triton 的 dot operand dtype 合同处失败 | generated 三阶段在 H100 已运行；两机都属于 source/device/toolchain 边界 |
| `legacy_flash_attention_bias` | source compatibility gap | source compatibility gap | vendored legacy kernel 在 Triton 3.6 下 biased/unbiased 均数值错误；不能绕过 adapter 或放宽 tolerance 冒充 baseline |

5090 仍是固定表中的 4 个 non-pass；H100 仍是 2 个 non-pass。没有一个是 Intent Triton compiler 新增的 compile/numerical failure。

## 6. 验证范围

本轮执行了：

- 本地与 H100 独立 checkout 的 C++ build；
- 5090 与 H100 并行定向 runner；
- 用户列出的全部超标 entry；
- 四个 non-pass；
- TMA、causal split、warp specialization、`tl.range` 的一次性 A/B；
- 每次 runner 都包含 generated/source 数值对照和实际 GPU timing。

最终固定数字写入：

- `report/baseline-new/triton-5090.csv`
- `report/baseline-new/triton-h100.csv`

额外尝试用旧语料的 `varlen_attention` 检查 ragged region-end 组合，但它在进入 Triton provider form 前就停在 shared Physical Program 的 broadcast provenance 冲突，因此不能拿它证明或否定本轮 Triton form；该失败没有被归到 Triton，也没有为它修改 shared 层。

## 7. 真实剩余边界

本轮之后，Triton 的两处数量级/近数量级投影塌陷已经被修掉。尚未关闭、且有具体代码结构证据的 Triton compiler 缺口是：

1. Mamba3 所需的二维带步长 descriptor provider representation；
2. padded RoPE 所需的、对 side-effecting branch 有完整 legality 的通用 if-conversion form；
3. 5090 block-sparse 128-token logical page 内的 provider physical sub-tiling。

BatchNorm 的 B blocking 属于 shared physical mapping，不应在“只做 Triton”的轮次里用 Triton 私有机制补。paged GQA/MLA 的大 ratio 是不同作者算法/launch decomposition 的端到端差异，不应通过恢复 compiler-private multi-launch 来追平。

因此，Triton 的主路径与本轮确认的两处严重投影问题已经站住；但若把目标严格定义成“每一个同算法 baseline 都不超过 1.05x”，当前状态仍未完全达到，不能把剩余项写成已完成。
