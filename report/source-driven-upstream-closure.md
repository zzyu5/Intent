# Source 上游实现闭环报告

## 结论

这一轮没有自行设计新算法，而是从已经纳入 `source/` 的外部实现反向补齐 DSL 入口和可比接线。最终新增 12 个编译记录（11 个 runner 名，其中 FP8 分为 E4M3/E5M2 两个独立 case），全部经过同一份 Kernel IR、同一套 GPU Physical Plan 和三个 target emitter；H100 上 35/36 个 provider-case 数值通过，唯一不支持项是 cuTile 在 SM90 上明确拒绝 `float8_e8m0fnu`。RTX 5090 上已经实测的 32/36 格全部数值通过，剩余 4 格因同机外部进程占用约 28.5 GiB 显存而未完成测量，CSV 明确记为 `not_measured`，没有借用旧数字或 H100 数字。

本轮真正修掉的不是 12 条按名字分裂的路径，而是三处共享缺口：

1. structured 与 data-dependent tensor index 在事实和 Plan 中不再混成一个 `tensor_indirect` 布尔值；
2. state stream 内的收缩轴可以同时承担 ordered 与 reduction，流的全局 K 偏移不再被内层 reduction 投影覆盖为零；
3. pointwise 结果轴、stream 内层收缩轴和 staged feature tile 进入共享 Plan，target 叶子不再从逻辑 shape 猜这些物理事实。

两份固定全量表已更新为 101 个 case：RTX 5090 位于 `report/baseline/kernel-performance.csv`，H100 位于 `report/baseline/kernel-performance-h100.csv`。既有 89 行及其 source 数字没有改动；本轮只追加 12 行。

## 一、清单口径

`source/` 当前有 159 个 Git-tracked 文件。排除 runtime、test、cache、support、`*_runtime.py` 和 `test_*.py` 后，有 89 个上游实现文件：CUDA 20、cuTile 17、TileLang 23、Triton 29。

89 是实现文件数，不是 89 个算法。一个算法往往在多个 provider 各有一份实现；一个文件也可能包含多个低层 kernel、调度 helper 和 launch stage。因此报告不把文件名 basename 匹配当成算法覆盖率，也不声称“89 减 12”就是已有算子数。

本轮对 source 做了语义级筛选，得到三类结果：

| 处置 | 数量 | 含义 |
|---|---:|---|
| 新增 DSL 编译记录 | 12 | 上游算法和现有记录不同，能够作为单次 kernel 调用陈述 |
| 新接公平上游数字 | 9 个 provider-case | 算法、调用次数和计时范围能够对齐；FP8 两种格式分别计数 |
| 不接数字 | 其余新增格 | 只有 dispatch wrapper、算法/接口不同、或无法拆出同 scope 内层 kernel |

重复 provider 实现、support/helper、同一算法的不同 target 拼写，不另建 DSL entry。明确没有硬凑的例子包括：

- causal-conv checkout 当前只有 C++ dispatch wrapper，不能拿 wrapper 时间冒充内层 kernel；
- TileLang GQA decode 是 split/no-split 调度，和本轮 continuous GQA 单 kernel 不是同一算法记录；
- TileLang Mamba source 的运行封装不能在当前接线中拆成同调用次数、同计时范围的内层 baseline；
- fused linear cross entropy、block-sparse attention split/combine 等是作者编排的多调用 pipeline，不应被塞成一条单-kernel记录；
- source 中 FMHA backward、causal-conv backward/update、真正的 sparse 2:4 primitive 等仍是后续可独立审视的上游候选，本轮没有把“尚未处理”写成“确认无需处理”。

## 二、补入的 12 个记录

| 记录 | 上游结构 | 本轮压出的能力或边界 | 公平 source 数字 |
|---|---|---|---|
| `causal_conv1d` | causal depthwise Conv1D + bias + SiLU | 左侧负偏移、非对称边界、depthwise 收缩 | 无：checkout 只有 dispatch wrapper |
| `continuous_gqa_decode` | continuous KV 上的 GQA decode | ordered stream + 多对一 head 映射 + stream 内收缩 | 无：现有 TileLang source 是另一种 split 算法 |
| `mla_prefill` | content/position 双 score 的 causal MLA | 两个收缩共享同一 stream、GQA 映射、在线归一化 | cuTile |
| `mamba_chunk_scan` | chunk selective scan forward | 高元融合、组广播、causal scan 与前态项 | 无公平内层接线 |
| `w4a8_packed` | int8 activation × packed signed int4 weight | 位运算、符号扩展、stream 内 unpack、整数收缩 | TileLang |
| `embedding_forward_lookup` | embedding row gather | data-dependent tensor index 与 in-bounds 前置条件 | Triton |
| `block_scaled_matmul` | 每 32 个 K 元素共享 E8M0 scale | E4M3/E8M0 类型、stream 与独立 inner reduction 轴组合 | cuTile（仅 RTX 5090；SM90 不支持 E8M0） |
| `splitk_attention_reduce` | split-K partial output/LSE 合并 | max + exp2 权重 + 加权归约 | cuTile |
| `fp8_gemm/e4m3` | FP8 GEMM | E4M3 输入输出与 FP32 accumulate | TileLang |
| `fp8_gemm/e5m2` | FP8 GEMM | E5M2 输入输出与 FP32 accumulate | TileLang |
| `index_select_rows` | row index-select | 一维 data-dependent gather | Triton |
| `scaled_index_add` | unique indexed scaled update | data-dependent unique scatter、原位更新 | Triton |

算法忠实性方面有两点明确边界：`embedding_forward_lookup` 通过作者前置条件声明合法 index，而 upstream 还为无效 index 发 mask；当前 benchmark 只生成合法 index，因此是同一 workload。`block_scaled_matmul` 覆盖 upstream 的逻辑二维 scale 路径，不覆盖它专为特定矩阵单元准备的 swizzled scale layout；两者数学算法相同，存储布局能力不同。

## 三、编译器具体修改

### 1. 索引事实不再丢失结构差异

旧路径把所有 tensor index 都归成一个“间接索引”。这使 emitter 不知道索引是作者给出的规则表达式，还是从外部张量读取的数据。

当前共享事实和 `intent_plan.transfer.tensor_indexing` 明确区分：

- `none`：没有 tensor index；
- `structured`：index tensor 能沿 SSA/use-def 回到逻辑轴和受限整数表达式；
- `data_dependent`：index 来自外部数据，只能按真正 gather/scatter 投影。

cuTile 和 TileLang 的 load/store 现在只消费 Plan 中的分类。structured index 保留作者已经写出的表达式；data-dependent index 使用各自真正的 gather/scatter 原语。没有新增任何算子名判断。

### 2. 同一逻辑轴的 ordered + reduction 组合

W4A8 首轮暴露了一个静默错编：stream loop 已经生成全局 `stream_axis_index`，但同一轴同时作为 contract reduction 时又被 inner-reduction 初始化覆盖成零。于是每个 K tile 都重复读取第一个 tile，三个 target 一起产生巨大误差。

Physical Plan 现在用 `intent_plan.stream_axis` 明确记录 stream 与内层 reduction 轴的关系：

- stream 轴和 reduction 轴是同一节点时，保留 stream 的全局块偏移；
- 两者是不同节点时（block-scaled 的外层 `KB` 与内层固定 32），才为独立 inner axis 建立本地 range；
- 某个 stream 内 contract 没有显式逻辑 reduction domain 时，该关系为空，而不是拒绝整个 attention 算法。

修复后 W4A8 三后端 `generated/reference=0`，continuous GQA、MLA、分页 attention 和变长 attention也重新通过。这是轴角色组合的修复，不是量化 GEMM 或 attention 特判。

### 3. 收缩范围与 pointwise 结果轴进入 Plan

block-scaled 将外层 `KB` 流和固定 32 的内层收缩分开后，realizer 会从 Kernel IR 的收缩轴事实生成固定 inner range；三个 emitter 只负责将同一 range 投影成各自的向量/切片表达。

pointwise 结果的逐轴 provenance 也进入 `intent_plan.pointwise.axis_nodes`。这让 reshape、broadcast 和 structured index 在 target 上读取同一份结果轴绑定，不从结果 shape 反猜逻辑轴。Plan verifier 会拒绝引用不存在物理轴的绑定。

### 4. staged scatter 读取 stage feature tile

全量受影响读者复验发现 cuTile grouped GEMM 曾把 staged scatter 的 feature index 展开成完整 `N=4096`，而值仍是 `(128, 64)` tile。根因是叶子从外部 view shape 重建索引，没有消费 stage plan 中的 `offs_feature` 和 `feature_mask`。

当前 staged cuTile scatter直接读取 stage 已生成的 member routes、`offs_feature` 与 `feature_mask`。Triton/TileLang 各自保留必要的目标语法路径，但使用的是同一份 stage member/feature 决策。三后端 grouped GEMM 的 base、K/N/member tail 和 empty-group case 均重新数值通过。

### 5. 类型和边界

语言、MLIR type、ABI 和三个 target 补齐了 `f8E8M0FNU`、E4M3/E5M2、u8/i8 与 i64 接线。Triton 没有 E8M0 tensor dtype，因此 ABI 机械映射为 u8 storage，cast op 根据自身输入类型兑现 `2^(x-127)`；它不再沿 operand producer 链猜测来源。

cuTile 在 H100 SM90 对 E8M0 明确报 `TileUnsupportedFeatureError`。这被记录为 `unsupported`，没有降级成另一个算法，也没有增加架构型号分支。

## 四、新接上的上游比较

以下是本轮第一次接入并固定的公平对照，均为预分配输出、计时区间内只保留同次数 launch 的组合。

### RTX 5090，p50 ms

| 记录 / provider | generated | source | 比值 |
|---|---:|---:|---:|
| W4A8 / TileLang | 0.1163 | 0.0895 | 1.299x |
| block-scaled / cuTile | 0.1775 | 0.1393 | 1.274x |
| split-K reduce / cuTile | 0.0056 | 0.0061 | 0.918x |
| FP8 E4M3 / TileLang | 0.0124 | 0.0184 | 0.674x |
| FP8 E5M2 / TileLang | 0.0138 | 0.0184 | 0.750x |
| index-select / Triton | 0.3702 | 0.3666 | 1.010x |
| scaled index-add / Triton | 0.5222 | 0.5268 | 0.991x |

embedding 的本机 Triton 测量被外部显存占用阻断，因此没有固定旧数字。

### H100，p50 ms

| 记录 / provider | generated | source | 比值 |
|---|---:|---:|---:|
| embedding gather / Triton | 0.1028 | 0.0342 | 3.006x |
| MLA prefill / cuTile | 0.0249 | 0.0193 | 1.290x |
| split-K reduce / cuTile | 0.0064 | 0.0066 | 0.970x |
| W4A8 / TileLang | 0.1791 | 0.1580 | 1.134x |
| FP8 E4M3 / TileLang | 0.0141 | 0.0135 | 1.044x |
| FP8 E5M2 / TileLang | 0.0143 | 0.0135 | 1.059x |
| index-select / Triton | 0.1891 | 0.2198 | 0.860x |
| scaled index-add / Triton | 0.2698 | 0.2979 | 0.906x |

H100 的 block-scaled cuTile source 与 generated 都受同一 SM90 E8M0 能力边界阻断，因此没有数字。continuous GQA 和 causal Conv1D 没有可比 upstream kernel，不填参考时间。

## 五、验证与未决判断

本轮实际执行的是每条 DSL emit 后真实运行并对 reference 的 repro，没有新增 test 目录、pytest 或校验设施。

- H100：新增矩阵中 35 个 provider-case PASS，cuTile block-scaled 明确 unsupported；同时复验 grouped GEMM、paged attention、varlen attention 等共享判定读者。
- RTX 5090：新增矩阵 32 格实测 PASS；Triton embedding、cuTile continuous GQA、cuTile MLA、TileLang continuous GQA 因外部 ComfyUI 占用约 28.5 GiB 导致 OOM/tuner 无合法运行配置，记 `not_measured`，不是 compiler unsupported。
- stable softmax 三后端再次通过，p50 分别为 0.3645、0.3686、0.3535 ms，和固定表同一量级。

有两项取舍当时无法从单个 target 推断，最终由跨 target 结果决定：

1. staged unique-store 不能机械复用普通 store 的通用 pointer/mask；stage 的 member/feature 二维映射是 Physical Plan 已决定后的必要目标投影。强行合并曾使 grouped GEMM 数值错误，因此保留独立语法路径，但删除了先计算普通路径再丢弃的死逻辑。
2. H100 上 cuTile grouped GEMM 最终 generated 已能编译并完成 tuning，但同进程 upstream autotuner 因外部 VLLM 占用约 77.7 GiB 显存而 OOM；这不改变既有固定 source 数字，也不作为编译器失败。

当前 compiler diff 中没有按新增 kernel 名称分支；新增分叉只依据 op、索引类别、轴角色、stage relation 和 target capability。
