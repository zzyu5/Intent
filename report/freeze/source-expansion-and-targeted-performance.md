# 真实语料扩充与定向性能收敛

## 结论

这一轮没有新增语言构造，没有改写作者算法，也没有做全量重跑。结果分成三部分：

1. 从已经 vendor 进 `source/` 的实现中新增两个真实算法族、四个固定表 entry：state-cache causal convolution update，以及 Q/K in-place RoPE 的 full、partial、inverse 三种形态。
2. 新 RoPE 暴露并修掉 ordinary `view_store` 丢失 tensor-indexing 事实的问题；cuTile 现在从 Plan 读取这项事实，并对动态张量下标使用原生 `ct.scatter`。
3. 追性能时发现 embedding 的 feature lane 一直被固定成“单个 program 处理整行”。共享 realizer 现在只对作者已经显式 partition、主体拿到区域、且没有区域级语义依赖的 pointwise lane 增加独立 program ownership。RTX 5090 三个 target 分别改善 71.4%、38.2%、33.2%；H100 分别改善 68.1%、63.1%、80.5%。

5090 上 TileLang 的三个 RoPE entry 都生成了完整 TileLang 和 CUDA 源码，但 CUDA 12.8 的 `nvcc` 在编译 `sm_120a` 时崩溃。相同 Kernel IR、Physical Plan 和 TileLang 投影在 H100 上全部通过，因此这些格子记录为 `failed`，不误写成 target capability `unsupported`。

## source 审视与选择

本轮先审视了 CUDA、cuTile、Triton、TileLang 四类 vendor 源，再决定新增项；没有按“容易编”反向选题。

| vendor 线索 | 判断 | 本轮处理 |
|---|---|---|
| `source/cuda/causal-conv1d/.../causal_conv1d.cpp` 的 `causal_conv1d_update` | 与已有整段 causal conv forward/backward 不同；它更新持久 state cache，并立刻计算当前 token 输出 | 新增 one-token update entry |
| `source/cutile/tilegym/position/rope/rope.py` 的 `_rope_kernel` | 同一 kernel 原地更新 Q/K，Q/K 头数不同，支持 full/partial，反向复用 inverse rotation | 新增 full、partial、inverse 三个 entry，并接入公平 cuTile source 对照 |
| vLLM Triton combined top-k/top-p Qrita kernel | 单 kernel、算法可表达，但与现有 insertion top-k 和 nucleus cutoff 属同一 sampling 族；其主体规模远大于本轮另外两个结构，未拿它替代更有区分度的 stateful update 与多张量原地 partial transform | 已审视，未伪装成已覆盖 |
| TileLang DeepSeek segmented radix top-k | 真实单 kernel，但算法依赖 block shared histogram、shared atomic 和 `T.sync_threads` | 当前冻结 Core 没有跨 target 的线程块同步合同，不造一条 target-only DSL 假能力 |
| TileLang persistent MLA decode | 依赖 `T.sync_grid()` 在同一 kernel 内跨 CTA 合并 | 超出当前单-kernel 可移植同步合同，不搬成普通 attention 冒充同算法 |

这轮没有向 `source/` 再复制代码：选中的两份上游实现已经按既有目录存在。未选项保留为真实边界，不通过改算法、拆调用或拼 PyTorch reference 来制造 baseline。

## 新增算法

### State-cache causal convolution update

新增 `causal_depthwise_conv1d_update`，固定模型级运行形状为：

- batch 64；
- channel 4096；
- state width 4；
- f16 输入、state、weight、bias、output；
- f32 累加；
- SiLU 在同一 kernel 内完成。

一次调用完成两件不可拆的算法动作：把每个 `(batch, channel)` 的 state 左移一格并写入当前输入，然后用更新后的四个 state 与权重做 depthwise 点积。runtime 同时校验 in-place state 与输出，计时区间只含已经准备好输出和 state 的 kernel launch；每次计时前恢复 state，恢复不进入 CUDA event。

vendor wrapper 还允许 chunked sequence、`cache_seqlens` 和 `conv_state_indices`。本 entry 明确是最常见的 one-token contiguous-state 特化，不声称覆盖那些不同接口形态。vendor 目录只有 C++ dispatch，缺实际 CUDA kernel 和可独立运行的同范围 runtime，因此 source 时间留空。

### Q/K in-place RoPE

新增两个 DSL kernel 和三个测量 entry：

- `rotary_qk_inplace`：full RoPE；
- `rotary_qk_partial_inplace`：只旋转前 64 个 head dimensions，其余 64 个保持原值；
- inverse entry：复用 full kernel，输入负 sine，与上游 backward 的 inverse rotation 一致。

固定运行形状为 `B=4, S=2048, QH=32, KH=8, D=128`。Q/K 头数不同，cos/sin 沿 head 轴广播，Q/K 都原地写回；乘加在 f32 中完成，结果转回 f16。`rotate_pair` 是正常的 `@intent.fn`，没有在三个 entry 中复制公式。

cuTile upstream 使用同一调用次数、相同 Q/K/cos/sin ABI 和相同计时范围。Q/K 的输入恢复位于 event 外，生成侧和 source 侧都只计一次 kernel launch。causal update 没有这样可拆出的 source kernel，因此不填假的 source 数字。

## 新语料逼出的 compiler 修复

### Tensor-index store 事实闭合

RoPE 第二半写回的地址是 `phase + half_dimension`，作者表达式已经完整进入 Kernel IR。ordinary `view_store` 原来只分析 boundary，没有像 load/scatter 路径一样记录 tensor-indexing；cuTile leaf 因而把动态张量下标误当普通 block store。

修复分两层：

- `KernelFacts` 的 ordinary `view_store` handler 同时执行 `classifyTensorIndices`；
- cuTile store handler 只读 Plan 的 `tensor_indexing`，非 `none` 时机械投影为 `ct.scatter`。

这不是 RoPE 分支。受这项事实影响的旧 `shifted_row_copy` 与 `reshape_and_cache` 都做了定向数值回归；两台机器均通过。

### Pointwise lane 的 program ownership

`embedding_forward_lookup` 的 DSL 已经明确写出 token partition，主体一次看见一块 token region；feature domain 只承担 pointwise load/store lane。旧 Plan 把动态 feature extent 固定成一个完整 row-vector，所以每个 program 同时处理 token tile 与全部 1021 个 feature，生成侧没有机会得到上游的二维 `(token block, feature block)` grid。

共享决策现在满足以下条件时，给该 lane 同时分配 `parallel` role、program order 和可调 ownership tile：

- lane 的直接用途只有 load/store/gather；
- 所有用途属于同一个 parallel owner；
- owner 的输入是作者显式写下的 partition；
- 同一个 owner 下没有另一个 vector domain；
- 该轴不承担 reduction、scan、contraction 或区域级 mask 语义。

Kernel IR 没有变化。Plan 只是为一个已有的、纯 pointwise 的逻辑轴选择二维物理分配，tile 大小仍交给各 target tuner。

第一次实现时缺少“owner 必须来自显式 partition”这一条，导致 `shifted_row_copy` 的标量 row 外层也被拆成二维 grid；H100 cuTile 同环境 A/B 从 0.0163 ms 退到 0.0245 ms。补齐判据后恢复到 0.0164 ms，而 embedding 的二维 grid 与全部收益保留。这个 A/B 同时证明最终规则区分的是作者主体看见 region 还是 scalar，而不是 kernel 名字或 shape 阈值。

### `I.indices` 的执行域诊断

开发 RoPE 时还发现：若 `I.indices(domain)` 写在任何 execution region 之外，前端过去仍会生成 tensor SSA；但此处没有“当前选中的 physical range”，leaf 最终只能报 operand 没有 target value。

RoPE 上游本来就在 kernel 行主体内构造 dim index，所以 DSL 按原算法把配对索引放在 token parallel 内。Kernel IR verifier 同时补上唯一语义合同：kernel 顶层的 `intent.indices` 必须诊断“缺少选择 indexed range 的 enclosing execution region”，不再把名义上存在、实际不可投影的写法放到深层失败。

## 定向性能结果

### RTX 5090：新增 entry

单位为 ms；`source` 只在同算法、同调用数、同 launch-only scope 时填写。

| kernel | Triton generated | cuTile generated | cuTile source | TileLang generated |
|---|---:|---:|---:|---:|
| causal conv1d update | 0.0771 | 0.0277 | — | 0.0153 |
| Q/K RoPE full | 0.0728 | 0.0696 | 0.0696 | failed |
| Q/K RoPE partial | 0.0241 | 0.0266 | 0.0266 | failed |
| Q/K RoPE inverse | 0.0710 | 0.0696 | 0.0655 | failed |

通过的 generated RoPE 与 reference 最大误差为 0；cuTile source 与同一 reference 最大误差不超过 `4.89e-4`。causal update 的 state 误差为 0，输出最大误差为 `1.22e-4`。

cuTile full/partial 与上游基本持平；inverse 慢约 6.3%。差异没有被拿来改 DSL 算法或固定 target 参数。

### H100：新增 entry

| kernel | Triton generated | cuTile generated | cuTile source | TileLang generated |
|---|---:|---:|---:|---:|
| causal conv1d update | 0.0413 | 0.0259 | — | 0.0138 |
| Q/K RoPE full | 0.0617 | 0.0639 | 0.0600 | 0.0609 |
| Q/K RoPE partial | 0.0321 | 0.0393 | 0.0333 | 0.0321 |
| Q/K RoPE inverse | 0.0617 | 0.0641 | 0.0599 | 0.0611 |

H100 上 12 个 generated 格子全部数值通过。generated RoPE 与 reference 最大误差为 0；causal update state 误差为 0，输出最大误差为 `3.05e-5`。

### Embedding 的同表前后对比

| device | target | 固定表旧 p50 | 本轮 p50 | 改善 | 与 Triton upstream |
|---|---|---:|---:|---:|---:|
| RTX 5090 | Triton | 0.1543 | 0.0442 | 71.4% | 1.028x |
| RTX 5090 | cuTile | 0.1085 | 0.0670 | 38.2% | — |
| RTX 5090 | TileLang | 0.0668 | 0.0446 | 33.2% | — |
| H100 | Triton | 0.1029 | 0.0328 | 68.1% | 0.959x（固定 source 0.0342） |
| H100 | cuTile | 0.1437 | 0.0530 | 63.1% | — |
| H100 | TileLang | 0.1472 | 0.0287 | 80.5% | — |

同一条共享 Plan 修复在两台机器、三个 target 都改善，且 H100/TileLang 收益最大。Triton 在 5090 上把与上游的差距从 3.59x 收到 1.028x；H100 上从 3.01x 落后变为约 4% 领先。现有 source 列的测试逻辑没有变化，因此固定 source 数字没有被重测值覆盖。

### 上一轮 emission 改动的实际收益

上一轮把 cuTile 间接读取从 `ct.gather` 后再接逐元素 `where` 收成单个 masked gather，当时没有更新固定表。本轮只复验直接受影响的两格：

| RTX 5090 repro | 固定表旧 p50 | 本轮 p50 | 改善 | 数值 |
|---|---:|---:|---:|---|
| cuTile paged attention | 0.3436 | 0.3252 | 5.4% | pass；D=80 fully-masked row 同样 pass |
| cuTile MoE | 10.2724 | 10.1040 | 1.6% | pass |

H100 的 cuTile Python/Torch/CUDA Tile 环境在固定表之后被更新，绝对数不能直接归因给代码。同机、同环境、上一提交对本轮代码的 A/B 为：paged attention 0.8104 对 0.8117 ms（无可辨识变化），MoE 10.1647 对 9.9704 ms（约 1.9% 改善）。因此本轮只更新 5090 的这两行，不用混杂环境的 H100 数字覆盖旧固定值。

## 下层失败与环境归因

5090 的三个 TileLang RoPE 不是能力检查拒绝：

1. canonical Kernel IR、Physical Plan、TileLang source 和下层 CUDA source 都生成完成；
2. CUDA 12.8 `nvcc -arch=sm_120a` 在编译生成 CUDA 时 segmentation fault；
3. 同一份代码在 H100 的 CUDA 12.2 `nvcc -arch=sm_90a` 上全部编译、运行、数值通过。

因此固定表用 `failed`，不写成 `unsupported`，也没有为 TileLang 在共享层加入绕行机制。

H100 最初的 TileLang 定向项全部失败，原因是远端默认选择 `/usr/bin/nvcc` 11.5，它不认识 `sm_90a`。显式使用机器已经安装的 CUDA 12.2 后，先用未改动的 upstream online-softmax runtime 验证环境，再运行本轮条目；这属于测试接线，不进入项目代码。

## 实际验证范围

没有跑全量。执行范围只覆盖新增和真正读取本轮共享判据的 entry：

- 5090：三个 target 的 causal update、full/partial/inverse RoPE、embedding；
- 5090：cuTile `shifted_row_copy`、`reshape_and_cache`、paged attention、MoE；
- H100：上述同范围 entry；
- H100：额外用 upstream TileLang online softmax 区分环境失败与 compiler 失败；
- H100：对上一提交和本轮代码做 cuTile embedding、shifted-row、reshape-cache、paged-attention、MoE 的同机同环境 A/B。

H100 验证使用独立的 `/tmp` Git bundle 工作副本和独立 build 目录；活动仓库原有的 dirty/untracked 文件没有被覆盖或修改。

两份固定 CSV 保持相同列结构和相同 kernel/case 顺序；只新增四行，并更新 embedding 与 5090 上已经确认改善的 paged-attention/MoE 数字。表中没有加入版本、阈值或检查逻辑。

