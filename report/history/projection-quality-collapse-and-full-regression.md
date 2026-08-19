# 投影质量塌陷排查与双机全量回归

## 1. 本轮结论

本轮没有把 LayerNorm backward 的现象继续归结为“跨目标不存在共同的 bf16 正确答案”，而是把问题拆成了两个独立合同：

1. 作者选择 partial buffer 的 dtype，这是算法表示的一部分，编译器不能为了照顾某个目标改回 f32。
2. 每个目标必须判断自己能否把该 dtype 下的 many-to-one reduction 投影成可用的目标原语。语义上能生成一条指令，不等于具备可接受的高吞吐实现。

最终定性如下：

- Triton 能把 bf16 partial reduction 投影到可用的原生路径，DSL 保持 bf16。
- cuTile 的公开类型检查允许 SM90+ 上的 bf16 atomic，生成代码也确实是 `ct.atomic_add`，但当前 tile atomic surface 在这一 many-to-one 冲突形态下退化到 0.7228 ms；同一算法的 f32 路径是 0.0928 ms。这里不是漏写了一个更好的 cuTile API，而是“语义原生、吞吐能力不足”。编译器现在只对 bf16 `intent.scatter_reduce` 提前报告 unsupported，不再让这条 7.8 倍慢路径冒充可用支持；普通 bf16 atomic 没有被一刀切禁掉。
- TileLang 0.1.13 具备原生 bf16 scalar/x2/x4/TMA atomic surface。当前 scalar 投影数值正确，未走 CAS 或软件循环；x4/TMA 在这份分阶段 scatter 的布局上不能机械闭合，因此没有保留实验性替代路径。它仍然支持该程序，但当前端到端延迟为 5090 0.1904 ms、H100 0.1795 ms，明显慢于 Triton。这是已测出的目标实现质量差距，不伪装成已解决。

同类排查还修掉了 cuTile 二维 tile 候选、TileLang compact transfer、ordered stream tile 选择、TileLang 尾块读取和结构组合候选等共享或 target-local 问题。两台机器随后并行完成了 113 个 kernel × 3 个 provider 的全量运行；两份固定 CSV 已用本轮实测值更新。

## 2. bf16 partial reduction 的证据链

### 2.1 DSL 与数值

LayerNorm backward 的两个 partial buffer 现在由作者源码明确声明为 bf16。三个目标消费的是同一份 Kernel IR 和 Physical Plan，没有在 shared realizer 中按目标改 dtype。

定向复测结果：

| 设备 | provider | 状态 | p50 | 数值结果 |
|---|---|---:|---:|---|
| 5090 | Triton | pass | 0.0705 ms | `dx/dw/db = 0.0009766 / 0.0809 / 0.0856` |
| H100 | Triton | pass | 0.0713 ms | `dx/dw/db` 均在现有容差内 |
| 5090 | cuTile | unsupported | — | emission 前在 bf16 many-to-one scatter 处拒绝 |
| H100 | cuTile | unsupported | — | 同上 |
| 5090 | TileLang | pass | 0.1904 ms | `0.0009766 / 0.0930 / 0.0862` |
| H100 | TileLang | pass | 0.1795 ms | `0.0019531 / 0.0975 / 0.0768` |

Triton 的定向 upstream p50 为 0.0666 ms，generated/upstream 约 1.06×。固定 CSV 中 upstream 列仍保留原先固定值；本轮没有因为重新计时 generated 而擅自改动 source 数字。

### 2.2 为什么 cuTile 是能力边界而不是拼写 bug

排查同时验证了三件事：

- 生成代码使用的是 cuTile 原生 `ct.atomic_add`，不是我们手写的 CAS 循环。
- 当前安装包的 dtype capability 检查明确允许 SM90+ 的 bf16 atomic，因此不能笼统地说“cuTile 没有 bf16 atomic”。
- 在真实 LayerNorm 冲突模式下，bf16 tile atomic 的实测吞吐比 f32 partial 路径慢 7.8 倍；没有发现另一个能机械替换、并保持相同所有权和冲突语义的 cuTile 原语。

因此拒绝条件没有写成“cuTile + bf16”，而是精确到 `intent.scatter_reduce`、bf16 destination、many-to-one 高吞吐投影这一组合。诊断落在最早拥有完整 op、dtype 和 target capability 信息的 emission 层。

### 2.3 为什么 TileLang 没有被一起拒绝

TileLang 发出的仍是原生 atomic；1.6 倍左右的退化不能直接证明它是假支持。尝试把 scalar surface 换成 x4/TMA 后，下层 layout/TMA inference 不能为当前 scatter 结构建立合法投影。因为没有一条已验证的更好机械拼写，本轮删除了实验路径，保留正确的原生 scalar 支持，并把性能差距如实记录。

## 3. 同类投影塌陷排查

排查输入不是算子名字，而是两份固定表中同一行三个 generated p50 的离散度。对离散度大的格子并排读生成源码，依次问：

1. leaf 是否漏用了目标已有的整块搬运、归约、矩阵或索引原语；
2. leaf 是否因为 Plan 少事实而自己重建结构；
3. 目标是否已经使用原生 surface，只是下层 layout、编译成本或执行模型不同。

### 3.1 找到并修复的问题

| 问题 | 根因 | 修法 | 结果 |
|---|---|---|---|
| cuTile `max_pool2d` | 通用二维 program tile 候选只覆盖 64/128，真实 2D 小窗口没有合适候选 | 扩充 target-local `program_m/program_n` 合法候选和联合 profile，仍交给 cuTile tuner 实测 | 5090 `0.1060→0.0200` ms；H100 `0.0495→0.0257` ms |
| TileLang W4A16 | compact quotient-index load 无条件先物化为 shared；但 Plan 选择的是 private fragment，导致 layout 无解 | cooperative compact transfer 只在 `result_space=shared` 时启用；fragment 保留精确间接 element projection | 5090 `0.1899→0.1034` ms；H100 `0.1656→0.1086` ms，数值通过 |
| indexed ragged relation | offset 物理范围与 indices 映射的职责混在一起，三个 leaf 曾把成员 offset 当成最终逻辑 member | 三个 leaf 都遍历 Plan 给出的 offset range，再机械应用 Kernel IR 的 indices mapping | 反向索引探针三个目标一致；原有 ragged 全量项继续通过 |
| cuTile load padding | validity 已经完成相同 fill 后，leaf 又物化一次等价 padding | 仅在 Plan 的 validity domains 与 padding domains 精确一致时消除重复 padding | 不改变算法语义，受影响 load 数值通过 |
| ordered reduction | 只要 ordered axis 同时有 reduction role 就使用 `stream_contract`，把普通 stream reduction 的候选限制到 128 | 只有真正存在于 `contractionDomains` 的轴才使用 `stream_contract`；普通 ordered reduction 使用 `stream` 候选 | cross-entropy 恢复：5090 Triton 0.3414 ms，H100 0.1918 ms |
| TileLang guarded bulk read | 整块 fast path 的尾块 `else` 清空整个 tile；另外 consumer-neutralized 曾过宽地允许 ordered/ragged/reduction 动态范围进入 bulk path | full tile 才整块复制并同步；尾块逐元素按精确边界读取；不再把 consumer-neutralized 当成任意动态轴可 bulk 的证明 | MLA prefill 两机误差 0.0009766，5090 0.0405 ms、H100 0.0273 ms |
| TileLang Mamba candidate | target tuner 把 layout 不合法但能编译的结构候选与合法候选混在一个 CUDA context，`skip_check` 让错误候选可被选中 | 按 Plan 的 `program_m + query + stream_contract + reduction` 角色组合声明当前 TileLang 可兑现的结构子集，policy 仍由 tuner 选择 | 5090 0.0246 ms、H100 0.0276 ms，误差 6.10e-5 |

这些修改没有一个按 kernel 名分支。shared realizer 只修正了“什么轴才是 contraction stream”这一结构事实；cuTile/TileLang 的限制和候选都留在各自 target-local capability/tuning 层。

### 3.2 读过源码后没有误判成 leaf 拼写问题的格子

| 格子 | 当前最大离散 | 源码结论 | 定性 |
|---|---:|---|---|
| H100 `block_sparse_attention` | 14.63× | TileLang 已使用原生 `T.gemm`；两台机器生成结构和候选一致，H100 单独慢 | TileLang/H100 下层 layout/GEMM 质量边界，不是漏用原语 |
| 5090 `absorbed_mla_prefill` | 10.07× | cuTile 已使用三次原生 `ct.mma`，没有手写 contraction | 尚未闭合的物理/下层质量差距，不是一个可机械替换的 leaf 拼写 |
| `viterbi_decode` / `smith_waterman` | 4.2–4.9× | 三家都是直接标量递推/动态规划投影，没有某家漏用已知同语义原语 | target 执行模型差异 |
| `barrier_option` | 4.8–5.4× | 差异来自控制与同步模型，不是同一个 canonical op 被展开成错误循环 | target 执行模型差异 |
| `causal_conv1d` | 4.2–5.0× | cuTile 已使用其原生 indexed load/gather surface | 下层 indexed access 质量差距 |
| H100 cuTile attention family | 约 1.6–2.5× | 本轮前后的同机日志生成源码、候选和 winner 一致；定向复测稳定复现 | 不是本轮回归，仍是既有物理/下层性能缺口 |

因此，本轮没有为了抹平表格而在 shared Plan 中加入目标专用 layout 或算法替换，也没有把这些仍可正确运行的目标一律改成 unsupported。

## 4. 全量回归

### 4.1 执行范围

- 5090 与 H100 同时启动，互不等待。
- 每台机器运行 113 个 kernel × Triton/cuTile/TileLang，共 339 个 repro。
- CSV 中因部分 kernel 含多个 case，共 122 行记录。
- runner 累计用时：5090 3235 秒，H100 3453 秒；并行后的墙钟由较慢的 H100 路径主导，约 58 分钟。
- 每一格都执行真实 DSL → Kernel IR/Plan → target source → 下层编译/运行 → 数值对照；没有新增测试目录、fixture 或检查逻辑。

原始 runner 状态：

| 设备 | Triton | cuTile | TileLang |
|---|---|---|---|
| 5090 | 111 pass / 1 unsupported / 1 compile failure | 110 pass / 2 unsupported / 1 fail | 98 pass / 10 unsupported / 5 fail |
| H100 | 111 pass / 1 unsupported / 1 compile failure | 108 pass / 3 unsupported / 2 fail | 102 pass / 10 unsupported / 1 fail |

CSV 将失败进一步按真实性质保留为 `unsupported`、`compile_failed`、`compile_timeout` 或 `failed`，没有把 runner 的统一非零退出码全部压成一种状态。

### 4.2 失败与边界

- Triton `token_sparse_mla_prefill`：两机都在下层编译时超过 Triton 单 tensor 1,048,576 元素上限；记为 `compile_failed`。
- cuTile `fp8_mqa_logits`：runtime-sized lane 不能作为当前 matrix-M 轴，保留 target capability unsupported。
- TileLang `token_sparse_mla_prefill`：当前没有 mechanical batched GEMM projection，提前 unsupported。
- 5090 TileLang 三个 RoPE case：`nvcc -arch=sm_120a` 崩溃；H100 同源码全部通过，保留为 5090 target-toolchain failure，不上移到 shared capability。
- 5090 TileLang `absorbed_mla_prefill`：所有候选在该设备的 layout/validation 阶段失败；H100 通过，保留 `compile_failed`。
- H100 cuTile `token_sparse_mla_prefill`：tileiras 单候选编译超过 10 秒，保留 `compile_timeout`；不是算法或 Plan correctness 失败。
- cuTile LayerNorm backward：本轮新增的精确 bf16 many-to-one scatter unsupported，属于有意关闭慢伪支持。

没有发现新出现的跨三目标共同数值错误。全量中暴露出的 MLA 尾块和 Mamba 候选问题已在本轮内修复并重跑受影响项。

### 4.3 关键性能变化

改善最明确的项目：

- cuTile `max_pool2d`：5090 改善 81.1%，H100 改善 48.1%。
- Triton `layer_norm_backward`：两机均改善约 14%。
- Triton `mamba_chunk_scan`：5090 改善 27.2%，H100 改善 24.7%。
- TileLang W4A16：5090 改善约 45.5%，H100 改善约 32.4%。
- H100 TileLang bf16 GEMM：改善约 22%。
- 5090 三个 provider 的 `block_sparse_attention` 均有 1.3%–7.0% 改善。

需要如实保留的变化：

- TileLang LayerNorm backward 因作者改为 bf16 partial 后，从 5090 0.1145 增至 0.1904 ms、H100 0.1151 增至 0.1795 ms；这是原生 scalar bf16 atomic 的目标质量，不是旧 f32 路径的回归比较。
- H100 cuTile attention family 与固定表相比显著变慢，但本轮开始前的同机全量日志已经是同一慢值；本轮前后源码和候选没有变化，不能归因给本轮 patch。
- W4A16 的 Triton/cuTile 固定表值在当前环境未复现；本轮没有修改这两个 leaf 的 W4A16 投影，因此 CSV 记录当前全量值，不虚构“保持旧值”。

当前赢家分布：

| 设备 | Triton | cuTile | TileLang | 无可比较 winner |
|---|---:|---:|---:|---:|
| 5090 | 59 | 37 | 42 | 0 |
| H100 | 61 | 29 | 36 | 1 |

存在并列，所以三列合计可大于 122。两台机器的赢家分布仍然不同，三种 target surface 的价值没有退化成“只保留一家即可”。

## 5. 代码边界复核

本轮最终代码满足以下边界：

- 没有 kernel-name dispatch；`repro.sh` 中的名字只负责选择真实 upstream baseline 文件，不参与 lowering。
- bf16 scatter capability 在 cuTile leaf 判断，没有为了 cuTile 往共享 Plan 增加字段。
- indexed ragged 的 offset range 来自 Plan，indices expression 来自 Kernel IR；leaf 不再从 extent 或角色名猜 member identity。
- `stream` 与 `stream_contract` 的选择只依赖共享 contraction fact，不依赖附近 op 名称。
- TileLang compact/shared、bulk/tail 和 structural candidate 限制均消费已有 Plan 字段或角色集合，没有重写 Kernel IR。
- 被实验否决的 TileLang x4/TMA atomic 和 fragment-as-shared compact 路径没有残留。

## 6. 当前没有关闭的事项

本轮仍不能把以下差距称为“已解决”：

- TileLang bf16 many-to-one atomic 虽为原生语义，吞吐仍明显弱于 Triton。
- H100 TileLang block-sparse GEMM、cuTile absorbed MLA、cuTile causal convolution 和部分动态规划结构仍有较大 provider 离散。
- token-sparse MLA 在三个下层分别撞到 tensor-size、编译时间或 batched-GEMM capability 边界。
- H100 cuTile attention family 的当前慢值稳定、且不是本轮引入，但仍未达到固定表曾记录的水平。

这些项没有被写进 `doc/`，也没有被包装成编程模型事实；它们只作为当前实现状态保留在本报告和两份固定表中。
