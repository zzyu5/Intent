# H100 全量测评与定点修复

本轮完成了冻结版本的全量测评，随后修复已定位的问题并做定点验证。**高性能库中，同一个 Intent 程序的两个 target 在整体上已经接近；新生成的 TritonBench 100 题，正确率优于现有 agent Triton 提交，但整体性能仍未达到 reference 或 agent Triton。** 不能把高性能库的结果套到新生成的 100 题上。

当前表包含全量结果及已完成的开发复测；没有再次全量运行最新编译器，也没有重新生成或修改这 100 份候选程序。冻结的全量结果分别保存在 Git 提交 `1917bbc9`（生产库）和 `5fab2401`（agent 实验）中。

## 结果入口与统计口径

只维护以下三个输出：

- [Triton 来源高性能库：source、Intent→Triton、Intent→cuTile](results/triton-h100.csv)。
- [cuTile 来源高性能库：source、Intent→Triton、Intent→cuTile](results/cutile-h100.csv)。
- [新生成 100 题：reference、agent Triton、Intent→Triton、Intent→cuTile](../agent_tritonbench/results/h100-high-refresh-20260923/results.csv)。

时间单位均为 **ms，完整算子耗时**；JIT、编译、调优不计入性能列。加速比定义为“对照耗时 / Intent 耗时”，大于 1 才是加速。只在数值通过、有有效时间的配对上计算几何均值；不同配对数不混作同一批样本。

生产库沿用原 baseline-v2 source 时间。为判断是否共同环境退化，只额外诊断了两个 source：varlen conv 为旧 0.050504、新 0.048496；layer norm 为旧 0.057824、新 0.061336。这不支持“当前所有 Triton 回退都由设备共同变慢导致”。诊断时间没有覆盖原 source 列。

## 高性能库

| 来源 | Intent→Triton 通过 | Intent→cuTile 通过 | Triton 相对 source 加速 | cuTile 相对 source 加速 | 同 DSL cuTile/Triton 耗时比 |
|---|---:|---:|---:|---:|---:|
| Triton 库 | 54/54 | 53/54 | 1.124×，52 对 | 1.106×，51 对 | 0.997×，53 对 |
| cuTile 库 | 35/35 | 35/35 | 2.599×，35 对 | 2.435×，35 对 | 1.067×，35 对 |

cuTile 库另有 2 条 H100/source 不支持的记录：block-scaled GEMM 的 SM100/E8M0 路径、NVFP4 packing 路径。保留状态，不强行运行，也不计入上表可运行分母。Triton 库有两条 source 时间不可用，但生成程序本次通过，不能把它们的 pass 清空。

**整体耗时比已在 1.2 以内，不等于逐项达标。** Triton 来源中仍有 7 项 cuTile 比同 DSL Triton 慢超过 20%，其中 3 项超过 2 倍；cuTile 来源分别为 6 项和 3 项。

| 主要长尾 | cuTile/Triton 耗时比 |
|---|---:|
| FP8 split-K GEMM | 3.557× |
| batch norm training | 2.085× |
| max pooling with indices | 2.038× |
| attention sink decode | 5.060× |
| sparse MLA prefill | 3.200× |
| absorbed MLA decode | 2.179× |

生产库仅剩 `softmax_backward` 的 cuTile 数值检查失败。当前证据是 f32 归约/融合后的误差超过原容差，尚未证明实现违反 Intent 数值合同；没有放宽容差或擅自改成 f64 累加。

cuTile 来源的 2.435× 是相对这批固定 source 配置的结果，不能解释成“生成代码普遍比 cuTile 库快 2.4 倍”。例如 sparse MLA source 固定 `TILE_H=1, TILE_N=64`，本例产生 131072 个 blocks；生成程序采用已有的持久工作分配。原 source 308.889 ms 是这个具体入口的观测，不是最佳 cuTile 实现的证明。配置入口见 [attention.py](providers/cutile/attention.py)。

## 新生成的 TritonBench 100 题

| 语言组 | 冻结全量通过 | 定点修复后通过 |
|---|---:|---:|
| reference | 100 | 100 |
| 原 agent Triton 提交 | 73 | 73 |
| 同一批 Intent→Triton | 90 | 94 |
| 同一批 Intent→cuTile | 77 | 93 |

agent Triton 另有 7 题无提交、11 题程序错误、9 题数值失败。没有重新生成 Triton 对照组。右列是原提交在修复后的编译器/定点配置下的开发结果，不能替代左列的冻结成绩。

cuTile 的 `solve` 仍使用[定向静态配置](../agent_tritonbench/results/h100-high-refresh-20260923/solve/cutile-tuning.json)和 60 秒 native 编译预算。`normalize_pairwise_distance` 已在默认配置下通过，其定向配置已删除。因此 **93/100 仍不表示默认配置已全量验证到 93/100**。

| 当前可比较结果 | 几何加速比 | 配对数 |
|---|---:|---:|
| Intent→Triton 对比 reference | 0.932× | 92 |
| Intent→cuTile 对比 reference | 0.704× | 91 |
| Intent→Triton 对比 agent Triton | 0.754× | 68 |
| Intent→cuTile 对比 agent Triton | 0.584× | 68 |
| Intent→cuTile 对比同 DSL Triton | 0.753× | 91 |

最后一行等价于 cuTile 耗时约为 Triton 的 **1.329 倍**。这些数值不能支持“这轮 Intent 整体性能已超过 agent Triton”。

性能聚合排除 `sum_std` 的 CPU 常量结果和 `fused_svd_reconstruct` 的复制捷径；正确率仍以 100 题为分母。agent Triton 的 `tensordot_rsqrt` 使用 TF32，不进入同精度 agent 配对。

cuTile 从 77 增加到 93 个 pass 后，几何加速比反而降低，主要因为新增了原来不能计时的慢题；不能据此认定修复让原来通过的 75 个有效性能样本回退。它们并未在这次定点修复中全部重测。

## 已修复的具体问题

| 问题 | 修复与实际结果 |
|---|---|
| Triton Mamba 单步回退 | launch 参数绑定反复处理维度、stride、alias 信息；复用 view span、打包 metadata/overlap，并用原生 heuristics 延迟构造 descriptor。0.248720 → **0.053720 ms**，原容差通过。完整算子计时没有改为只测 kernel。 |
| varlen conv 配置缺档 | 共享 pointwise/reduction profile 缺中间 64 分块。0.131360 → **0.076976 ms**；仍慢于 source 0.050504 ms，未宣称完全追回。 |
| cuTile chunked softmax 超时 | 原先只有 Triton 使用的动态归约资源约束上提并供两个 leaf 使用。剔除明显超出候选预算的组合后，**0.819568 ms、pass**。 |
| matmul 结果被完整物化 | 合法的“matmul 后按 parallel 输出坐标取值”先形成 FULL_M×FULL_N，再 gather。扩展已有 contraction projection，沿 free axes 传递规则分块，保留 K、累加、有效性及轴置换。原 tensordot 现在 Triton **0.062720 ms**、cuTile **0.115952 ms**，均通过。 |
| loop carry 覆盖 | Triton 多结果 yield 的顺序赋值破坏旧状态；改为同时赋值。Chebyshev 原提交通过。 |
| retained read 与类型关系 | 提前物化需要保留的读，避免再修改新建的 chunk 坐标；保留 StoreOp 身份，重建失效的分析事实；修复 tuple reduction 分量的共同物理 shape。determinant 两个后端通过。 |
| reverse suffix | 支持合法隐式 domain end 的反向后缀证明；least-squares QR 两个后端通过。 |
| cuTile 操作缺口 | 补动态均匀行提取、重复 Cartesian 坐标的显式投影、i64 常量类型，以及 erfc/i0 lowering。log-softmax-linear、masked-select、erfc、i0 原用例通过。 |
| subregion 容量过大 | 容量分析保留常量与循环下界，避免把 k+1 到 257 的最多 256 个成员补齐到 512；solve 尾部从 256×512 收紧为 256×256。静态归约候选也按容量去重。 |
| 写回分块与默认候选过大 | 将已有归约范围约束复用于逐点写回；全覆盖处理保留 `min(parameter, shape)` 的既有分块。cuTile 依据已导出的维度绑定，在合法候选中保留填充最少的 pointwise ownership 组合。normalize 默认配置 **Triton 0.009136、cuTile 0.009336 ms，均 pass**，reference 0.016984 ms；cuTile 比原定向结果 0.063112 ms 快 6.76 倍。 |
| 资源估算与索引证明丢失 | 广播、splat、reshape 不再独立按展开后的 tile 计寄存器；参考 Triton 的 [view lowering](../../../ref/triton/lib/Conversion/TritonGPUToLLVM/ViewOpToLLVM.cpp)。cuTile 保留复合最小值和 atomic 活跃坐标边界，使已有运行时范围检查可以选择 32 位内部数组索引；不改变外部 ABI、逻辑 index 类型或数值精度。 |

这些改动没有自动增加 kernel，没有修改候选算法，也没有新增另一套 form/plan。主要实现位于 [GPU shared passes](../../lib/Dialect/GPU/Transforms/)、[Triton leaf](../../lib/Target/Triton/)、[cuTile leaf](../../lib/Target/CuTile/)。

数学 helper 采用成熟算法：erfc 使用 [OpenLibm 分段算法](https://raw.githubusercontent.com/JuliaMath/openlibm/master/src/s_erf.c)，i0 使用 PyTorch/ATen 的 Cephes Chebyshev 形式。此次验证范围是原生产用例及其既定容差；未声称做过全定义域 ULP 验证，未实现的 f64 路径仍明确拒绝。

## 超时究竟是什么

静态配置表固定的是候选集合，不是每个候选的 native 编译时间或运行时间。此次至少有三种不同问题：

1. **无意义的大物理形状。** tensordot 的整矩阵物化已修复；solve 的动态尾部容量已收紧。这是 compiler 工作。
2. **合法复杂 kernel 的预算不足。** solve 选择一个 reduction=256、access=3、occupancy=1 的配置，native 准备实际约 26 秒；15 秒限制会误判失败。采用 60 秒编译预算后通过，完整算子 **12.480344 ms**。编译耗时没有放进性能列，但算子本身仍慢。
3. **调优候选运行过慢或不返回。** attention sink 的诊断栈停在 cuTile warmup，GPU 持续工作。后续原生调优记录既有约 130 ms 的慢候选，也有触发 3 秒运行限制的候选，不能全部称为 JIT 卡住或死锁。

实验入口现在使用 cuTile 自带的 `single_run_timeout_sec` 隔离候选运行，没有自建调优框架。attention sink 保留原有访问方式/occupancy 选择后得到 **3.999440 ms、pass**，source 为 4.336416 ms；仍明显慢于同 DSL Triton。

只删除 gather 或 occupancy=2 来缩短准备是错误取舍：FP8 曾因此从约 7.91 ms 变成 10.44 ms。已恢复原选择；当前 **7.301776 ms**，winner 是 gather、occupancy=2、64×64 输出块。不能把该收益简单归因于新增 K=256 profile 行；最新程序已是每个 IT iteration 执行完整 256-K MMA。

准备与 native 编译在不同工作进程间并发；实际 GPU 初始化、调优和计时共用锁，释放锁前同步 CUDA。不是用外层 flock 把整个实验串行化。

normalize 最终定点准备约 65.5 秒，两个 kernel 分别保留 24、45 个候选；后者有 3 个 native 候选超过 15 秒，其他候选及最终程序通过。这里解决的是整题超时，未声称所有 native 配置均可编译，也未以增加全局超时来取得 pass。

## 剩余失败的边界

两个 Intent target 共同剩下 6 题：

- `min`：提交将 min 的 identity 写成 -inf，运行得到非有限结果。
- `combined_activation`：作者 reshape/broadcast 关系不合法。
- `fused_cross_entropy_log_softmax`：把 domain 和 tensor index 混合后产生额外 Cartesian 轴，结果不能存入一维 losses。[现有索引合同](../../doc/dsl/authoring.md)明确了这一区别。
- `fused_cross_entropy_softmax_layernorm`：数据产生的索引缺少当前合同要求的有效边界声明。
- `solve_symmetric_ldl`：作者只实现 1×1 pivot，未覆盖 reference 的 2×2 pivot 行为。
- `matrix_power_eig`：直接乘法与 reference 的 eig/reconstruction 路径在本例 f32 下超出既定容差。

**cuTile 另有 solve_multiple_lu 的数值错误，尚未解决。** 同一 canonical input 的两个 shared IR 在排除 target/config 属性后相同。原 benchmark 的 workspace 观察进一步确认：首行和前两次 pivot 一致，但第一次 panel 更新已错；第二行数值恰好符合消元系数再次除以主元的结果。cuTile frontend IR 中只有一次 load/div，随后将同一值用于 store 和 broadcast/FMA，token 链完整；尚未证明是哪条 native 变换改变了该值。64 分块、不同 access、native O0、32 位数组索引，以及外部隔离的 CUDA 13.4.92 控制均未解决。正式环境仍保持 cuTile 1.5 / CUDA 13.3，没有改候选、容差或补猜测性的 barrier。

不能因为 Triton 有 debug_barrier 就给 cuTile 增加一套 CTA barrier。cuTile 的 block 内通信由下层处理，Tile IR 的 token order 能建立操作之间的顺序；这里已有同 workspace 的 alias/token 关系。[cuTile 执行模型](https://docs.nvidia.com/cuda/cutile-python/execution.html)、[Tile IR 内存模型](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html)支持这一职责边界，不能无证据归因为“缺同步”。

## 性能结论与后续优先级

目前可解释的结论是：

- 高性能库同 DSL 的整体差距已经收敛，但 FP8 split-K、跨行统计和不规则 KV 访问仍有长尾；不能概括为某个后端天然更好。
- 新 100 题中，一部分差距来自作者组织。例如全局归约/扫描只有一个 kernel，而对照采用显式分阶段程序。compiler 不会替作者自动拆成多个 kernel。
- 一部分确实来自 lowering：整矩阵保留、分块参数遗漏、host 绑定成本与数学 primitive 缺失，本轮已有真实修复收益。
- `num_worker_warps=1` 在本实现中表示交给下层推断；cuTile 没有同名 `num_stages` 也不意味着没有 pipeline。Triton 源码存在 descriptor 分支不证明 winner 使用了 TMA。后续归因必须结合实际配置和 native 产物。

FP8 当前为 Triton **2.048984 ms**、cuTile **7.288080 ms**；BN cuTile **0.092016 ms**，与同机修复前 0.092672 ms 接近。本次边界/资源估算修复没有显著改善这两个长尾。FP8 winner 的 Nsight Compute 显示 L1/TEX 吞吐 66.99%、DRAM 吞吐 3.99%、Tensor pipe 活跃 4.07%；170 registers/thread、约 106.5 KB shared/block 将驻留限制在 2 blocks，achieved occupancy 12.45%。当前证据支持片上访存压力和低占用，不支持“缺少 MMA”或“DRAM 带宽耗尽”；原子活动 23.06% 也不足以单独认定原子是主因。Profiler 时间不写入 benchmark 表。

下一步保持定点推进：继续定位 cuTile LU，解决 solve 对定向配置的依赖，并检查静态 K 的物理分块与上述性能长尾。新生成程序的算法组织问题另行归类，不改写本轮提交来追分。尚不能宣称全部正确性问题和性能目标已完成。

## 执行入口

生产库复用 `python -m experiments.gpu`，分别选择 provider=triton/cutile 和 target=triton/cutile；`--source-results` 与 `--output` 指向对应既有宽表，使用 `--jobs` 并发准备、`--gpu-lock` 串行 GPU 窗口。

已有 agent 提交复测使用 `python -m experiments.agent_tritonbench.benchmark`，传入原 `--program`、`--reference-ms`、`--timing` 和同一 `suite-100.json`。不调用包含生成步骤的主入口。solve 的 cuTile 定点命令额外传 `--cutile-compiler-timeout 60 --tuning-config .../solve/cutile-tuning.json`；normalize 不再传定向配置。其余编译缓存、中间 IR、native 源码和诊断观测均在仓库外。
