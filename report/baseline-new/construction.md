# baseline-v2 construction contract

## 1. 目标与边界

baseline-v2 回答的是：Intent 针对某一门 target language 发射的代码，和这门语言当前公开的高性能手写实现相比如何。它不再把“编译器能跑的全部语料”与“公平的性能对照”混在一张表里。

旧矩阵已经冻结在 `report/baseline/`。本目录下一轮只生成六张结果表：

- `triton-5090.csv`
- `triton-h100.csv`
- `cutile-5090.csv`
- `cutile-h100.csv`
- `tilelang-5090.csv`
- `tilelang-h100.csv`

每张表至少 30 个非 variant entry；当前 source 清单已提供 Triton 38、cuTile 34、TileLang 30 个候选。三门语言尽量使用同一功能族，但不会为了对齐而伪造 source 或改变 upstream 算法。30 只是最终表的下限，不是停止扩充现代模型核心路径的上限。

## 2. 什么可以成为一行

一行的单位是 runtime-visible algorithm entry：有明确输入输出、可独立调用、可独立计时。以下均不单独计数：

- `@jit` 文件里的内部 helper；
- autotune config 或同算法不同 tile 参数；
- 只为 reference/packing/import 服务的 support；
- `variant_*` 等等价 DSL 拼写；
- PyTorch composition、临时参考实现、手写 CUDA；
- 只有源码、没有可执行 runtime 的快照。

forward/backward、prefill/decode、dense/sparse、单 kernel/多 kernel pipeline 只有在算法合同和调用边界确实不同的时候才分行。

## 3. source 合同

V2 source 必须同时满足：

1. 公开、以性能为目标维护的实现；
2. 和表所属 provider 相同：Triton 对 Triton、cuTile 对 cuTile、TileLang 对 TileLang；
3. 算法结构、输入输出语义、调用次数和计时范围能与 Intent DSL 对齐；
4. 本地保留原始 source，算法结构不改；ABI 适配只放在相邻 runtime；
5. 能在目标设备的当前环境真实编译和运行。

FlagGems 只冻结在 V1，不进入 V2。CUDA reference 仍可帮助理解算法，但不进入这六张表。没有合格 source 的算法不拿低质量实现凑数。

每个 provider 的唯一 source 清单分别是：

- `source/triton/README.md`
- `source/cutile/README.md`
- `source/tilelang/README.md`

README 同时记录 entry、上游文件、模型级 case 和直接 runtime 命令。仓库中的非 CUDA kernel 文件若不在对应 README 中，也不是 README 明列的 necessary support，就不应存在。

## 4. 输入规模

输入按真实模型层级选择，不用小 tile 代替 workload：

- GEMM/FFN：典型 hidden 4096/7168、intermediate 11008/14336，token 2048/4096/8192；
- attention prefill：sequence 2048/4096，head dim 128，真实 batch/head 数；
- attention decode：query 可以是 1，但 KV cache 必须是 4096/8192/16384 级；
- routing/MoE：真实 expert 数、top-k 与 token 数；
- normalization/pointwise：至少覆盖完整 model token×hidden 张量；
- quantization：对完整权重/activation 矩阵计时，不只量一个量化 block。

同一 entry 在 5090D 和 H100 默认使用同一 case。若某个 case 在一台设备上因显存无法成立，不能静默缩小后仍写成同一行：应换成两台都成立的模型级 case，或让该 entry 不计入两台共同的 30 行。

## 5. 计时合同

相邻 `*_runtime.py` 的职责是证明 vendored source 可被调用并打印真实形状、结果摘要和一次耗时；它不是最终 CSV 的测量器。下一轮的统一 runner 才负责表格测量，并遵守：

- JIT 编译、autotune、输入生成、输出/workspace 分配在计时外；
- timed region 只含用户为了得到该结果必须执行的 GPU launch；
- generated 与 source 的算法、launch 数和 scope 一致；
- sub-ms kernel 两边都用 CUDA graph；短核重放之间使用按设备 L2 容量确定的冲刷；
- adapter copy、ABI packing、路由 metadata 若不是算法本身的一部分，放在计时外；若算法本身需要，双方必须同侧计入；
- 多 kernel pipeline 只和相同 pipeline 比，整条 pipeline 一起计时；不拿内部 stage 和完整 wrapper 相比；
- 数值对照先通过，再记录性能。

source p50 在 case、runtime scope、provider/toolchain 与设备均未改变时保持固定；日常编译器修改只更新 generated 数字。改变上述任一条件时才重新采 source。

## 6. CSV 形状

六张表都使用同一份精简列：

```csv
kernel,case,generated_p50_ms,source_p50_ms,ratio,status
```

- provider 与 device 已编码在文件名中，不重复成列；
- `case` 对应 source README 中固定的 shape/dtype 合同；
- `ratio = generated_p50_ms / source_p50_ms`；
- `status` 只记录 `pass`、`unsupported`、`compile_failed`、`numerical_failed`；
- 不再保留 `triton_scope`。旧表需要它，是因为一张表混入了不同 provider、不同 callable scope 和 generated-only 行；V2 每张表的 provider 与计时合同已经固定，该列只会重复信息。

相近功能的 entry 在表中相邻排列，但不再维护 `algorithm_group` 列。相似而不等价的算法各占一行；例如 online softmax 与单 tile softmax、MoE routed projection 与完整 MoE FFN 都不会合并。

## 7. 下一轮的完成定义

下一轮才把这份 source 清单兑现为 DSL/example 和六张 CSV。某个 entry 只有同时满足以下条件才进入最终表：

1. DSL 照 source 的算法结构书写，不为编译器改算法；
2. 三层 lowering 走完整，生成对应 provider 源码；
3. generated 与 source 在同一 case 上数值一致；
4. source 和 generated 的 timed callable 按第五节对齐；
5. 两台设备的结果分别写入各自表，不互相复制。

未满足的候选可以保留在 source README 中，但不能为了凑 30 写入 CSV；应从同一 provider 的其它合格 source 补足。
