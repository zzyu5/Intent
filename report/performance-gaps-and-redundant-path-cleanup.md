# 性能缺口修复与冗余路径清理

## 1. 本轮范围与结论

本轮只处理三个已经定位过的性能缺口，并审计它们周围是否留下双份实现；没有新增算子，也没有做全量回归。

最终结果是：

- W4A8 的 `k // 2` 不再退化成普通 data-dependent indexing，而是成为共享层可证明、Plan 可携带、TileLang 可直接消费的 compact quasi-affine coverage；两台机器上均数值通过，TileLang 分别达到 upstream 的 `0.7718x` 和 `0.6052x`。
- ordered-ragged GQA prefill 的读取物化与消费者中和事实进入共享 Plan，TileLang 不再逐元素兑现本可整块搬运的读取；5090 上达到 upstream 的 `0.9274x`，H100 上三者最快的 cuTile 为 `4.5847 ms`，快于 upstream 的 `5.2314 ms`。
- LayerNorm backward 的实验推翻了“算法与 adapter 已完全对齐、只剩共享 collision realization”这一前提。把 upstream 的 bf16 partial + serialized lock 搬进 Plan 后反而更慢，并且同一 bf16 partial 算法会让 cuTile、TileLang 大幅退化。最终没有保留错误的 collision Plan；保留 provider-neutral 的 f32 partial ABI、单次批量清零、归约轴角色和静态 stride 消费。H100 Triton 已达到 upstream 的 `0.9744x`；5090 Triton 仍为约 `1.21x`，原因被重新定位为跨两个作者 kernel 暴露出来的 partial-storage 算法/ABI差异，而不是一个可以由当前单-kernel Plan 静默改写的物理参数。

本轮没有发现 U/S/P 判据无法归类的编译器决定。LayerNorm 暴露的是一个更外层的边界：跨作者 kernel 的公开中间张量不属于单-kernel Physical Plan 的决策空间。

## 2. W4A8：compact quasi-affine coverage

### 2.1 根因

W4A8 的 packed weight 访问使用 `k // 2`。此前分析只知道它是一个动态 tensor index，随后把它压成 `data_dependent`；这丢掉了三个已有事实：

- 索引只有一个来源轴；
- 除数是正的编译期常量；
- 一段连续 logical-K 对应一段更短但仍连续的 physical packed span。

TileLang 因而只能按 logical-K 逐元素加载，无法一次搬入唯一的 packed 覆盖区间。

### 2.2 修法与边界

共享分析新增受限的 compact quotient 事实，只接受：

```text
floor((d + c) / p)
```

其中：

- `d` 是唯一的非负逻辑域来源；
- `d` 的系数必须为 `1`；
- `c >= 0`；
- `p > 1` 且是编译期常量；
- 不能包含第二个 domain、tensor gather、变量除数或 opaque scalar 来源。

这个范围不是为 W4A8 定制的；它描述的是仍可机械计算连续物理覆盖区间的最大必要子集。共享事实把 `divisor/offset` 写入 transfer-relative `intent_plan.range`，Plan verifier 校验其合法性，TileLang leaf 只做两件事：按 Plan 计算 compact base/extent，并把 compact buffer 机械展开回 logical lane。Triton 与 cuTile 原本已能直接消费索引表达式，无需另加决策路径。

### 2.3 U/S/P 归类

- **U（唯一合法事实）**：索引表达式、唯一来源轴、常数除数与 logical-to-physical span。
- **S（结构性选择）**：TileLang 将这一 span 物化为一次 compact cooperative load，而不是逐 logical element 读取。
- **P（参数性选择）**：logical tile 的具体大小，继续交给既有 tuner。

### 2.4 定向结果

| 机器 | Triton | cuTile | TileLang | upstream | 最好 generated / upstream |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 0.1036 ms | 0.1163 ms | 0.0696 ms | 0.0901 ms | 0.7718x |
| H100 | 0.1541 ms | 0.2173 ms | 0.0963 ms | 0.1591 ms | 0.6052x |

六个 generated 结果均与整数参考精确一致。

## 3. ordered-ragged attention：联合读取决定

### 3.1 根因

varlen/GQA causal prefill 同时包含：

- ragged member ownership；
- ordered key stream；
- GQA 多对一 head 映射；
- causal visible prefix；
- contraction 前的 K/V 搬运。

此前 leaf 能看到这些局部结构，却没有一份共享事实说明某个越界通道是否会在可观察结果前被 mask/neutral element 消除。TileLang 因而保守地把读取展开成逐元素谓词、逐元素写片上 buffer，并额外同步；这不是 TileLang 语法必然要求，而是 Plan 欠定后 leaf 只能重新猜。

### 3.2 修法

本轮把两个相互独立的事实写进共享层：

1. transfer 的物化方式是 `direct` 还是 `deferred_to_contract`；
2. 无效通道是否已被消费者以归约恒等元中和，即 `consumer_neutralized`。

中和证明从 Kernel IR 的 use-def、padding identity 和最终 consumer 推出一次。TileLang 只有在 Plan 明确证明以下条件时才使用 f16 cooperative bulk copy：

- 物化方式允许直接搬运；
- 无效通道会在 contraction 前被中和；
- 物理 extent 是精确可投影的二次幂块；
- target 的片上 buffer 形状与 source span 一致。

不满足时仍走原有精确边界路径；没有 attention 名字分支，也没有在 TileLang leaf 重新推导 visible prefix。

### 3.3 U/S/P 归类

- **U**：ragged/ordered 关系、逻辑读取终点、GQA head 映射，以及“无效值是否在可观察结果前被中和”的合法性事实。
- **S**：在证明成立后选择 direct cooperative materialization；否则保留精确边界物化。
- **P**：query/K tile、线程数、流水级数和 provider 的具体候选，仍由下层 tuner 实测。

### 3.4 定向结果

| 机器 | Triton | cuTile | TileLang | upstream | 最好 generated / upstream |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 5.1986 ms | 5.6909 ms | 5.9283 ms | 6.3928 ms | 0.8132x（Triton） |
| H100 | 5.0472 ms | 4.5847 ms | 5.5611 ms | 5.2314 ms | 0.8764x（cuTile） |

六个 generated 结果均通过参考对照，最大误差为 `2.44140625e-4`。TileLang 自身在 5090 上为 upstream 的 `0.9274x`；H100 上为 `1.0630x`，但同一 Plan 的 cuTile 投影在该机上成为赢家。

## 4. LayerNorm backward：实测推翻原归因

### 4.1 初始状态

5090 上原始 generated 使用：

- 两个 f32 partial buffer；
- 每行两个 `atomic_add`；
- 每次调用分别清零两个 partial；
- 第二个作者 kernel 归约 partial。

同机 A/B 为 generated `0.0836 ms`、upstream `0.0626 ms`，即 `1.3352x`。

upstream 使用 bf16 partial、每 group 一个 lock/count，并在临界区进行 first-write 或 load/add/store。上一轮据此把差距归为共享 collision/partial-storage realization。

### 4.2 被否决的 Plan 方案

本轮曾实现一个不按算子名匹配的 `first_write_serialized` collision Plan：共享层识别 `owner % slots` 的 rank-two scatter collision，Triton/cuTile 从同一 Plan 生成锁、first-write 和 partial merge，TileLang 在没有 forward-progress-safe primitive 时提前拒绝。

实测否决了这条方案：

| 5090 Triton 方案 | generated | upstream | 比值 |
|---|---:|---:|---:|
| 原 f32 atomic + 两次清零 | 0.0836 ms | 0.0626 ms | 1.3352x |
| bf16 partial + serialized lock | 0.0989 ms | 0.0665 ms | 1.4869x |
| 锁 spelling 改为 upstream 默认 atomic 语义 | 0.1063 ms | 0.0667 ms | 1.5930x |

因此该 Plan op、三个 leaf 的 lock/workspace 分支、相关 verifier、诊断和 wrapper 接线全部删除，没有留下“不再走但还存在”的 collision 子系统。

### 4.3 为什么不能把 bf16 partial 当成共享 Plan 字段

这里有一个之前被忽略的边界：LayerNorm backward 在当前语料里是两个独立的作者 kernel。partial buffer 是第一个 kernel 的公开输出、第二个 kernel 的公开输入，其 dtype 和 shape 已经是作者 ABI，而不是 compiler-private workspace。

把 partial 改成 bf16 后，5090 Triton 配合单次批量清零可达到 `0.0700–0.0717 ms`，约为 upstream 的 `1.0522–1.0767x`；但同一份 DSL 会造成：

- cuTile：`0.0928 ms -> 0.7228 ms`；
- TileLang：`0.1145 ms -> 0.1890 ms`。

这证明不存在一个跨三 provider 共享的 bf16 partial 正确性能答案。若让每个 provider 选择不同 partial dtype，就会改变两个作者 kernel 之间的 ABI；单-kernel Plan 没有这项权限。要把 partial dtype 重新变成编译器私有决定，前提是作者把整个两阶段过程交成一次可编译调用，由编译器拥有 intermediate 的生命周期；当前项目明确不做这种跨调用编排。

### 4.4 最终保留的改动

- DSL 保持 provider-neutral 的 f32 partial ABI；第一维明确为静态 `PARTIAL_GROUPS=128`，与实际算法一致，不再把固定 group 数伪装成动态 `G`。
- 作者的 stride 前置条件进入 ABI metadata；Triton 通过一个共享 `staticViewStride` 查询只发射真正动态的 stride 参数，不在多处重新判断。
- `intent.reduce` 的来源 domain 明确记录 `reduction` 角色。归约轴和 lane 轴重合时，ownership 不会再把归约域误当成并行 program 轴；静态 reduction/lane 使用同一已选 physical extent，不生成一个无人消费的伪搜索参数。
- 两个确实必须清零的 partial 使用一次 `torch._foreach_zero_`，计时区间仍包含这次真实 GPU clear；没有把必要工作移出计时范围。
- Triton validity 与已有 mask 相同时不再重复合取。

### 4.5 最终定向结果

| 机器 | Triton | cuTile | TileLang | Triton upstream | Triton / upstream |
|---|---:|---:|---:|---:|---:|
| RTX 5090 | 0.0754–0.0764 ms | 0.0979 ms | 0.1181 ms | 0.0626 ms | 1.2059–1.2202x |
| H100 | 0.0836 ms | 0.1265 ms | 0.1151 ms | 0.0858 ms | 0.9744x |

所有结果均数值通过；f32 partial 的 dW/dB 最大误差处于 `2.4e-6` 量级。

5090 的剩余差距不再登记成“已知共享 collision Plan 没实现”。公平描述是：当前 generated 与 upstream 使用不同的跨调用 partial-storage 算法；在不扩大编译器职责到跨作者 kernel 编排的前提下，不能通过 leaf 特化或 Plan 字段合法抹平。

## 5. 冗余路径审计与清理

### 5.1 已删除

- **失败的旧路径**：完整删除实验性的 `CollisionOp`、shared collision facts、Plan verifier、三个 target 的 lock/ticket/workspace 接线和重复诊断。它既没有带来性能收益，又会让 TileLang 出现一个伪支持边界。
- **重复判断**：TileLang load 已在取得 `physicalFill` 后统一检查失败，删除后面的第二次相同检查。
- **重复谓词**：Triton load 的 validity 与原 mask 完全相同时不再生成 `mask & mask`。
- **无效候选入口**：cuTile 不再因为任意 `gather`/`members` 就创建两种 gather spelling 候选；只有 staged indirect gather 或 ordered indexed members 真实需要时才把该 target-local 拼写交给 tuner。

### 5.2 审计后确认应保留

- `scaled_contract` 之前的 DSL 模拟路径已经没有调用点；当前 `block_scaled_matmul` 直接构造 canonical `intent.scaled_contract`。
- Triton 的 native `tl.dot_scaled` 与 fallback、cuTile 的 `ct.mma_scaled`、TileLang 的 `T.scaled_gemm_fallback` 是各自 capability/projection，不是共享层的重复算法。Triton 两种 spelling 作为明确候选真实参与选择，不能当死代码删除。
- E8M0 cast/decode 仍被一般 cast 路径使用，不是旧 block-scaled 模拟残留。
- 旧的固定 warp/stage 阶梯没有残留；Triton launch 参数来自 tuner 配置，cuTile occupancy/gather spelling 也由 target-local tuner 选择。
- 三个 target 各自的 view/store/contract 叶子代码虽然形态相似，但承担的是不同 API、buffer model 与能力子集的机械投影；没有把合法的 target leaf 当成冗余删掉。
- frontend、shared realizer 与 leaf 的不支持诊断分别覆盖语言合同、共享物理合法性和目标能力子集；没有发现同一条件被多层重复或互相矛盾地拒绝。

## 6. 验证范围

本轮没有全量回归，也没有新增测试设施。实际执行的验证仍只有项目规定的真实 repro：

```bash
./examples/run/repro.sh triton w4a8_packed
./examples/run/repro.sh cutile w4a8_packed
./examples/run/repro.sh tilelang w4a8_packed

./examples/run/repro.sh triton varlen_gqa_prefill
./examples/run/repro.sh cutile varlen_gqa_prefill
./examples/run/repro.sh tilelang varlen_gqa_prefill

./examples/run/repro.sh triton layer_norm_backward
./examples/run/repro.sh cutile layer_norm_backward
./examples/run/repro.sh tilelang layer_norm_backward
```

同一组命令也在 H100 上执行。H100 的 TileLang 使用已经安装的 CUDA 12.2 `nvcc`；系统默认 CUDA 11.5 不识别 `sm_90a`，这是环境选择，不是 compiler capability。

此外，归约轴角色的共享改动用 softmax 的三 target 既有 repro 做过定向复验，未引入数值回归；没有因此扩大为全量测试。

## 7. 提交

- `700e766 fix: preserve ragged member extent provenance`
- `2487d0b realize compact and neutralized transfers`
- `8334784 refine reduction and ABI realization`

报告本身单独提交，不修改 `doc/`，也不改两份固定 baseline CSV。
