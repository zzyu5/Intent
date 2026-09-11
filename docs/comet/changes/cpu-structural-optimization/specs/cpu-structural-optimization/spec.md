# CPU 结构优化

## 1. 编译边界与适用范围

本能力在 canonical KIR 已构造的 CPU task/block program 上进行优化。Current program 保存完整算法、逻辑轴、source/capture relation、summary/transition/state、访问与效果、输出责任及资源。优化改变实际程序，不增加旁路 planning IR，也不回读 KIR 重建已经丢失的执行结构。

跨 CPU/GPU 复用的是优化思想、成立依据以及适合共享的分析工具；两类 execution family 分别实现自身的范围、控制、状态、存储和调度变换。不要求同一个 pass、同一 physical IR 或相同后端表示。CPU 内 Mojo/Weft 共用任务与外围组织，在选定实现及 provider lowering 处分别展开。

作者 DSL 不出现线程数量、SIMD width、implementation identity 或 provider 条件。本能力沿用现有 f32 区域程序及 bool/index/integer 辅助值，不修改 canonical 数值与可观察算法，不扩大语言类型或量化格式。

## 2. 有效遍历、谓词和状态

从当前 source/capture 的坐标、逻辑轴、半开范围及有效域推导 all-true、mixed、possible 区间。只有前提成立的单调比较和范围关系参与推导；源坐标不能被分段相对下标替代，整数边界与尾部不得溢出或改变成员关系。

All-true 区间可以去掉对应谓词；mixed 区间继续执行原条件。Possible 区间之外只有在完整 typed summary 按原数值合同等价于 identity、原 combine 的中立性可采用且无必须保留的 effects 时才允许跳过。该结论必须进入真实循环边界、source slices 和专属访问，不能只留分析字段。任一必要分量未知则不裁剪该部分。

不从 float 乘法的一个零操作数推导整体为零，不假设未知输入没有 NaN/Inf，不以 signed-zero 或 accumulator/近似语义变化换取证明。现有输入、合法转换和结构化计算可提供事实，但 benchmark 采样值不是编译期假设。Proof 或 consumer 不具备所需事实时保留原计算，不另起猜测算法。

状态特化仅在逐输出成立的首次状态事实及后续不变量下省去冗余字段维护；保留空遍历、首次原 combine、后续 part 的有效性及最终完整结果。Region scan 的 transition 无贡献性与 emit 输出责任分别判断，不能因为 state 不变而删除仍需写出的输出。

上述优化按当前 IR 的类型、关系、数值和 effects 分派，不识别 attention/causal 名字，不要求特定 source tree 或输入尺寸。局部 uniform folding 和 dead-computation removal 可与区间变换协同，但不是外部语义的新定义。

## 3. 输入供应、状态和资源复用

对相连 summarize/emit/apply/combine 和局部计算消费者，依据同一输入来源、坐标/访问对应、dtype、有效域、snapshot 以及 effects 确定复用是否成立。输入准备和循环不变量的共享范围由其所有使用点与依赖决定，不只对单个 operation 保存短期读取缓存。

在合法时形成一个明确的供应值或准备结果，绑定全部对应消费者。外围 CPU/provider passes 决定跨计算的 owner、生成位置和 lifetime；局部实现继续拥有仅供内部使用的 packing、decode 和寄存器微块。两者消费同一选定 implementation 与参数，不互相重复准备。

对 next-state 到 current-state 的物化，根据旧状态最后读取、逐元素读写依赖、alias、任务责任和数值求值关系选择合法的 forwarding、直接目标写入或其它等价状态承载。消除已证明不必要的全量复制和初始化；需要旧状态快照、交叉元素读取或多消费者时保留必要存储，不能无条件原地更新。

所有共享存储保存显式大小、初始化、owner、lifetime 和释放点；跨分支及循环的状态版本必须可追踪。Runtime 只执行这些已声明资源，不隐藏跨调用 repack、persistent cache 或调用方未见的 workspace。优化后的完整 state、所有 scan 输出和 task join 保持原义。

## 4. 计算块与目标实现展开

当前 task/data tile 中的 contraction 保留完整输入、轴、累加语义、输出范围和相邻消费者关系，直到目标实现的适用条件、输入供应与资源需求已经确定。CPU 外围 blocking、region segmentation 和 implementation microtile 仍是不同层级，使用同一候选 binding 连接。

适用的 Weft f32 implementation 接收有界二维计算块，并形成 Canonical Weft IR 中的实际计算及状态/访问连接。不得在没有当前能力或依赖理由时，先将每个输出元素变成独立 1×1 allocation/fill/contraction 再指望外部 compiler 恢复块级组织。真实 tail、稀小计算或目标局部实现仍可选择标量/小块，不把“没有 1×1”作为所有程序的机械规定。

Mojo 按自身需求形成 SIMD、register microtile 和 native ABI，不要求先采用 Weft 的结构化表示；Weft 也不强制经过 Mojo vector/scalar 展开。块内专业实现可编程，但实例化结果必须进入 current IR 并参与后续 analysis、legalization 和 verifier。

实现选择使用 typed operation、shape/tail、数值、访问和真实能力条件，不替换为完整 attention、GEMM 或 scan 模板。Serializer 只拼写合法程序，不决定 loop、scratch、packing 或候选。必要参数有真实 consumer，有限 tuning 可以选同语义程序，不能用扩大搜索掩盖缺少的结构优化。

## 5. 可观察成果与性能口径

每项优化需能定位输入事实、成立条件、实际 IR 改写和减少的执行工作；保留未覆盖条件，不把静态节点数减少直接当作毫秒收益。跨算法复用以操作/访问/状态关系为依据，不以共享文件数或 provider 个数替代。

运行证据仅来自已有生产 CPU benchmark；采用现有 registry、算法、shape、外部 dtype 和 source callable，Mojo 沿用单 NUMA 8-worker，Weft RVV 沿用 1-worker 预算。这些是当前比较条件，不定义 CPU 编程模型或性能最优核数。不直接比较不同硬件的绝对毫秒。

最终竞争力为 `generated_ms / source_ms`；优化自身收益比较同条件的前后 generated 时间。时间包含该次 native invocation 必要的输入准备、packing、内部物化、workspace、状态更新、任务派发与完成；排除编译、调优、加载、预热和外部输出分配。缓存与持久格式准备沿用对应现有调用边界，差异如实说明。

项目 CSV 保留真实 generated/source 时间、ratio、容差状态及必要失败说明，不承担历史审计。前后程序/性能证据在现有 Comet 交接和验证产物中表达，不新建进度系统；source 时间漂移、只保留快项或算术平均不得掩盖慢项。Weft linear attention 的结构归因和实际 generated 耗时改善是本轮重点；无收益、无执行证据或仍未解决的缺口不得写成已经收敛。

同一次性能运行只做一次现有约定容差的数值检查，不放大容差，不追求 bitwise 或逐操作一致。只编译、预热、运行和计时受影响项；不增加独立数值、边界、回归、组合、压力或逐 pass 全排列测试，临时脚本同样受限。发现真实错误修复后复用受影响性能项确认。未受影响的现有记录保留。

参考 Triton 的显式有效阶段遍历，以及 TileLang 中局部实现、布局/资源与 lowering 的实际连接，给出双方 file:line、具体差别和后果；不同 execution model 不要求复制控制结构或机器布局。本能力的验收只使用 brief 的 A1–A4，不从内部实现清单追加场景。

## 6. 工作区及非目标

本 change 基于 `comet/cpu-region-programs` 创建 `comet/cpu-structural-optimization` 独立 worktree，当前集成目标仍为前者。主目录、旧分支和其它 change 保持原样；最终合并、推送、PR 或 worktree 清理由 Archive 阶段按授权处理。

不接入 IME/AMX、DSA/Ascend、全量 dtype/量化格式或一般索引/InOut，不修改 ref 与论文仓库，不重写作者算法或稳定 `doc/` 语义来迁就实现。TianchenRV 仅允许本轮已单独授权的静态有界子视图/私有窗口到 RVV stream contraction 的通用 physical lowering 补齐；不增加 Intent 专用路径，保留他人改动并独立提交。其它外部能力或语义修改仍需单独授权，不把前轮授权自动扩大到本轮。

本轮不定义所有 CPU 算子的统一 1.05/1.1 性能门槛，不以这一点免除真实收益和慢项归因的交付要求。单个普通 Native change 完成上述相关变换与验收，不建立 Supervisor、统一跨 family planning IR 或额外测试框架。
