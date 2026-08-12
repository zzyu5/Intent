# H100 跨设备物理决策报告

## 结论

同一套 Python DSL、canonical Intent Kernel MLIR、共享 GPU Physical Plan 和三个 target emitter 已在 RTX 5090 D 与 H100 80GB HBM3 上分别闭合。两份全量表均包含 89 个 case、267 个 provider-case：264 个数值 PASS，3 个 TileLang 明确不支持，没有残留 downstream failure。

这轮得到的核心证据不是 H100 更快，而是同一算法在两台机器上确实得到不同的物理兑现：

- 编译器从实际 CUDA device 查询资源事实，不再依赖某个 PyTorch 版本是否暴露同名属性；
- 同一搜索空间由下层 tuner 在两台机器上选出不同的 tile、occupancy 与 cluster hint；
- persistent/row launch 的实际 program 数按设备 SM 数和编译后资源占用计算；
- L2 冲刷缓冲按实际 L2 容量分配，不再隐含一台机器的缓存大小。

没有出现“同一条结构规则无法同时服务两台机器”的证据，因此没有新增源码形态搜索，也没有任何 H100、5090D 或架构代号分支。

## 一、实验对象与口径

两台机器的实测属性如下：

| 设备事实 | RTX 5090 D | H100 80GB HBM3 |
|---|---:|---:|
| SM 数 | 170 | 132 |
| 每 SM shared memory | 102400 B | 233472 B |
| 每 SM registers | 65536 | 65536 |
| L2 | 100663296 B | 52428800 B |
| warp size | 32 | 32 |
| matrix units capability | 有 | 有 |

H100 使用三个相互独立的 Python 环境：Triton、cuTile、TileLang 0.1.13。编译器 C++ build 位于仓库外；虚拟环境、编译缓存和测量日志均未写入项目。

两份固定数字分别位于：

- 当前机器：`report/baseline/kernel-performance.csv`；
- H100：`report/baseline/kernel-performance-h100.csv`。

两份 CSV 使用同一 20 列 schema、同一 89 行 `(kernel, case)` 顺序和相同的 scope/status 缺失模式。H100 表由本轮完整日志机械抽取；当前机器表未重测、未覆盖，source 数字只由各自机器上的真实 upstream 调用产生。variant 的 original 不是 upstream，没有写入 source 列。

## 二、设备事实如何进入编译器

### 1. 统一 capability 查询

GPU target 现在通过 CUDA Driver API 查询：

- compute capability major；
- multiprocessor count；
- maximum shared memory per multiprocessor；
- maximum registers per multiprocessor。

Triton、cuTile、TileLang 共用同一个 resolver。这个改动解决了 H100 环境中旧版 PyTorch property object 不暴露 `shared_memory_per_multiprocessor` 的问题，同时避免把某个 Python wrapper 的字段集合当作设备合同。查询失败会携带 CUDA error name 直接报错，不使用默认值兜底。

### 2. 当前真正参与决定的属性

| 属性 | 当前消费位置 | 对两机的影响 |
|---|---|---|
| registers/SM | shared realizer 的 private-buffer capacity rule | 两机数值相同，因此本轮 residency 结果相同 |
| matrix units | contraction capability gate | 两机都允许 contraction |
| SM 数 | persistent 与 row launch program count | 170 与 132 产生不同 launch 规模 |
| shared memory、编译后 registers | Triton row stages/occupancy | H100 row stage 为 4，5090 D 为 2；program count继续受实际 kernel 资源限制 |
| L2 size | CUDA Graph 重放前的 cache flush | 分别分配两倍实际 L2，不使用固定字节数 |

`compute_units` 与 `shared_memory_per_unit` 已进入 compiler capability contract，但 shared realizer 当前除合法性校验外没有用它们选择轴、范围或 buffer residency；`dynamic_vector_width` 对固定 warp GPU 仍为 false。报告不把“被查询”写成“已参与 Plan”。

## 三、哪些规则在两台机器上都成立

### 1. 算法结构到 Physical Plan 的规则

89 个 case 覆盖 normalization、contraction、dense/varlen/paged attention、ragged MoE、scan、卷积、量化、动态规划、排序、原地更新和等价分解变体。两台机器上均保持：

- 轴角色、ownership、traversal、reduction、lane 与 access footprint 由算法 IR 决定；
- persistent 只由 program-space 与数据结构条件选择，不按设备型号分支；
- boundary、fill、stream stop、ragged mapping 和 private-buffer owner 不在 emitter 中重推；
- 三个 target 只投影同一 Plan，TileLang 的三个既有能力边界保持明确诊断。

全量结果相同地为 Triton 89 PASS、cuTile 89 PASS、TileLang 86 PASS + 3 unsupported。这说明共享结构规则没有因换机器产生新的算法误编译或 emitter 裂变。

### 2. 私有存储层级规则

私有缓冲的 scalarization budget 与 owner-local budget 分别由 `registers_per_unit / 1024` 和 `/ 128` 得到。两台设备均报告 65536 registers/SM，因此它们产生同一 residency 选择。

这只能证明规则在这两台“寄存器上限相同”的机器上没有冲突，不能证明两个除数设备无关。因为本轮没有提供不同的 register capacity 输入，这一维仍未被真正扰动；没有据此把存储层级加入组合搜索。

### 3. persistent mapping

persistent 的结构选择本身在两机相同，但具体驻留 program 数不同：cuTile 使用 `SM_count / num_ctas × occupancy`，Triton 与 TileLang 也以各自下层可见的 SM 数约束 launch。这里正确分开了“算法决定使用 persistent traversal”与“这台机器实际能同时驻留多少 program”。

## 四、哪些决定因机器而不同

### 1. 下层 tuner 在同一候选空间中选出不同赢家

cuTile 的完整日志能够直接观察 `best.config`。代表性对照如下：

| Kernel | RTX 5090 D winner | H100 winner |
|---|---|---|
| attention | K=32, Q=64, num_ctas=1, occupancy=2 | K=128, Q=128, num_ctas=2, occupancy=2 |
| GEMM | K=32, M=128, N=128, num_ctas=1, occupancy=2 | K=64, M=128, N=128, num_ctas=1, occupancy=4 |
| online softmax | N=1024, num_ctas=1, occupancy=2 | N=512, num_ctas=1, occupancy=4 |
| selective scan | N=512, num_ctas=1, occupancy=4 | N=1024, num_ctas=1, occupancy=2 |

这些差异发生在既有 `intent_plan.search_space` 和 provider tuner 内。共享 realizer 没有挑 tile 常数，也没有为 H100 加新候选规则。

Triton 内部能得到 `best_config`，但当前 `CompiledArtifact` 不暴露 autotuner object；TileLang 0.1.13 也没有稳定的公共 winner 字段。本轮没有为记录报告而新增一层追踪表示。三者最终性能与 cuTile 的可观测 winner 一起构成证据，不能把候选表本身误写成实际选择。

### 2. provider 赢家分布改变

以每行 generated p50 的最低值为赢家，精确相等保留并列：

| 设备 | Triton 独胜 | cuTile 独胜 | TileLang 独胜 | 有并列的行 |
|---|---:|---:|---:|---:|
| RTX 5090 D | 23 | 24 | 34 | 8 |
| H100 | 35 | 22 | 31 | 1 |

89 行中有 40 行的赢家集合发生变化。这个结果直接支持“三家逐 kernel 取最优”：赢家不是某个 surface language 的固定属性，也不是一次能跨机器照搬的配置。

## 五、H100 暴露并修掉的问题

### 1. cuTile 候选失败隔离

`attention_bias` 首次在 H100 上表现为 `No valid config found`，错误末端是一个 `num_ctas=2` 候选的 `misaligned address`。算法 IR、Physical Plan 和 bias 索引关系都完整；同一 ABI 在 Triton 正常，cuTile 的 rank-3 f32 `ct.load` 也与上游惯用写法一致。

把四个候选放进独立进程后，三个候选数值正确，只有一个进程触发地址错误。cuTile 的 exhaustive tuner 虽会捕获单候选异常，但默认在同一 CUDA context 里依次 benchmark；异步地址错误成为 sticky CUDA state，后续候选随之失败，最终伪装成整个 search space 无效。

修复没有删候选、没有改 Plan，也没有按 H100 分支。生成的 cuTile wrapper 使用下层公开的 `single_run_timeout_sec`，让每个候选的首轮执行进入隔离 worker。修复后的完整 repro 为：

- 4 succeeded, 0 failed；
- 数值最大误差 `3.0517578125e-05`；
- D=80 fully masked row PASS；
- p50/p95 = `5.0203 / 5.3904 ms`。

因此它是 target runtime tuning 接线缺口，不是新的机器决策维度。

### 2. 远端可移植接线

手动 repro 入口允许从环境指定 build root、CMake generator、MLIR/LLVM CMake package 和 Python executable。默认本机行为保持不变；H100 可使用仓库外 build 与三个独立环境，不需要把机器路径写进源码。

原来 softmax/GEMM runner 中的固定 5% 性能失败阈值已经删除。该阈值把一台机器上的 provider 关系冒充成 correctness；现在 repro 只对数值失败，性能完整报告到 CSV，由两机表对照判断。没有增加测试框架。

### 3. 不是编译器问题的环境缺口

TileLang 的两条 varlen baseline 首次失败在 upstream source import，原因是独立环境缺少 `einops`；失败发生在 DSL lowering 之前。补齐依赖后 `varlen_attention` 与 `varlen_gqa_prefill` 均完整通过。这两格没有被记成 Plan 或 H100 回归。

## 六、上游在主场之后还剩什么

H100 表中有 41 个带真实 source 时间的 provider-case。以 1% 内视为持平：

| 设备 | generated 胜 | source 胜 | 持平 |
|---|---:|---:|---:|
| RTX 5090 D | 20 | 16 | 5 |
| H100 | 18 | 20 | 3 |

上游回到数据中心卡后，generated 的总体优势确实收窄，但没有消失。H100 上仍有明显结构性优势的代表项：

- cuTile LayerNorm：`0.0965` vs source `0.4509 ms`；
- Triton LayerNorm backward end-to-end：`0.0900` vs `0.2580 ms`；
- TileLang RMSNorm：`0.0931` vs `0.2042 ms`；
- TileLang grouped GEMM：`1.0229` vs `1.5438 ms`；
- TileLang attention：`4.0630` vs `5.7362 ms`；
- TileLang GEMM：`1.6706` vs `2.1736 ms`；
- Triton paged attention：`0.2203` vs `0.2949 ms`。

同时有几处必须诚实记为上游明显更快：TileLang causal varlen attention、TileLang varlen GQA prefill、Triton cross entropy、Triton SwiGLU forward、cuTile MoE。它们说明“按当前设备重新生成”不是自动胜利，也没有用三 provider 最优掩盖逐格差距。

## 七、是否需要新的源码形态搜索

本轮没有发现第三类情况：没有哪一条共享结构规则必须在 H100 与 RTX 5090 D 之间二选一，才能让两边都成立。

- tile、warp/stage、scan tile、occupancy/cluster hint 已由既有下层 tuner 实测；
- SM 数、shared memory、registers 和 L2 能用设备属性表达；
- cuTile bias 问题由候选失败隔离解决，不是搜索空间缺失；
- 私有存储层级没有被不同 register capacity 扰动，证据不足以提升为搜索维度。

所以没有建立“发多份源码再挑最快”的新框架。当前仍保持：算法 IR 一份，Physical Plan 一份，target surface 只投影，参数候选交给下层 tuner。

## 八、未被这轮证明的地方

1. private-buffer 预算的两个除数仍只在相同 registers/SM 的两台设备上观察过；它们是明确的未决经验规则。
2. `compute_units` 与 `shared_memory_per_unit` 进入 capability contract，但除 target runtime/合法性外尚未成为共享 Plan 的选择输入；这不是遗漏的字段消费就应该强行补，而是当前没有算法结构决策需要它们。
3. Triton 和 TileLang 的实际 tuner winner 没有统一稳定的 artifact 接口；本轮拒绝为报告引入第二份选择真理。
4. 本机在最终 cuTile 单项复验时只有约 2.8 GiB 空闲显存，候选隔离 worker 报 OOM；占用来自现有 ComfyUI 进程，未被终止。H100 完整 repro 与本机此前固定全量表分别提供正确性证据，但不能把这次资源受阻写成一次新的本机性能复验。
