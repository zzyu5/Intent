# 目标

在不改变 DSL 与 canonical KIR 语义的前提下，闭合 shared GPU Program 到 Triton provider/runtime 的终端权威边界，消除已确认的 emitter 越权与跨 provider 事实投影差异，并将 RTX 5090D 和 H100 上当前 Triton registry 中稳定、语义与计时可比的默认 config 性能收敛到 source 的 1.05× 以内。终端结果同时证明当前 shared/provider 结构能够干净承载后续 cuTile 与 TileLang 接入。

# 范围

- 从 fresh current compiler 和生成源重新确认当前 `ratio > 1.05` 的条目；对真正可比的差距，只从 current Physical Program 的 mapping、blocking、ownership、traversal、materialization，typed config 或 provider form 修复可复用的语义类问题。
- 让 Triton tensor-descriptor 的运行时可用性、实际 shape/stride/alignment、allocator 要求及 pointer/descriptor 双形态成为 serializer 之前的 typed provider/runtime contract；dynamic 或 padded-stride view 不得被声称为 contiguous descriptor，Serializer 不再临场发明 legality、pruning 或 fallback policy。
- 保持 shared GPU IR 对 coordinate/source-axis/range/validity/effect 的唯一权威，使 Triton、cuTile 和 TileLang 按同一 current typed facts 选择 provider form；已证明的 exact tail coverage 不因 provider 局部重建能力不一致而退化，而 storage、copy、pipeline、descriptor 和 target-native tile 仍保持 provider-local。
- 核对 terminal emitter 与 `ref/triton`、`ref/tilelang` 及当前 Triton 3.6 实际 surface；执行决策必须已存在于 current provider program 或 typed launch artifact，emitter 只做确定性拼写。
- 用同一 current compiler state 和明确的 Triton 3.6 环境完整运行 RTX 5090D/H100 registry，更新两张 Triton CSV 为真实数值、性能或非可比终端状态。

# 非目标

- 不修改 `doc/` 已定义的精确数值语义来追平使用 approximate PTX/libdevice 实现的 source，不为 kernel/entry/name/source ID/单个 shape 增加 compiler 特例。
- `1.05×` 是稳定可比终端结果，不是 lowering legality gate；不通过挑选重复运行最优值、改变 source 算法/dtype/ABI/shape/计时范围或把不可比项伪造为达标来闭环。
- 不把 TTGIR distributed layout、provider storage/copy/pipeline、TMA/MMA 形式或 autotune winner 搬入 shared IR；不把 fixed tuning table、profile 组合或缺少 cost model 本身当作缺陷。
- 不重做 05c 已闭合且当前无反例的 occurrence-aware traversal、complete config tuple、resource filter、compile-device binding、sparse shorthand、TileLang scaled contract 或 cuTile post-rewrite verification。
- 不建设 test 目录、pytest、fixture、兼容层、版本管理、额外证据文档或长期性能基础设施。

# 验收示例

- A1：对 descriptor-capable 的 Triton kernel 同时传入 contiguous 与 dynamic padded-stride view 时，typed provider/runtime contract 在 serialization 前已完整表达 descriptor 资格、真实 shape/strides、allocator 和 pointer/descriptor 两路；运行时只能选择合法形态，两路都以同一语义实际 JIT/launch 并通过数值比较，Serializer 不重建 legality 或 fallback。
- A2：含显式 source-axis mapping、range tail、validity 与 structured computation 的 shared GPU Program 经 Triton/cuTile/TileLang legalization 后，每个 provider form 只消费 current typed relation/access facts；同一 exact full-coverage access 不因 cuTile 局部识别较弱而退化为 gather/scatter，provider-local storage/pipeline 仍在各自边界内，terminal program 无需 KIR side record 或 emitter 猜测即可验证与输出。
- A3：当前 Triton registry 在 RTX 5090D 与 H100 的明确 Triton 3.6 环境上运行完整 generated/source workflow 后，所有稳定、语义与计时可比的 entry 均通过数值比较且默认 config ratio 不超过 `1.05`；任何仍高于 `1.05` 的观测必须由 fresh same-code 复测和 source/algorithm 对照证明为测量不稳定或语义不可比，并记录为准确的非性能可比状态，不保留无解释的高 ratio `pass`；两张 CSV 来自同一 current compiler state。

# 约束与不变量

- `doc/` 是语言与编译器边界的最终权威；report、registry、CSV、source 与 ref 只用于定位差异和核对 provider 实际能力。
- canonical KIR 在 physical construction 后不可变；shared GPU IR 是唯一完整 executable authority；provider legalization 只添加 typed target-local form，terminal serializer 不重建 structure、mapping、access、validity、legality 或 candidate decisions。
- compiler policy 只基于 typed semantics、current def-use/coordinates/effects/reuse/lifetime、complete config 与 capabilities；unknown 不被当作 exact，也不用默认/fallback 伪造支持。
- 保持唯一 executable path；实现按语义完整节点提交，不留临时源码、cache、调试脚本或旧/新分支。

# 决策

- 使用一个普通 Native Change，不拆 Supervisor；descriptor runtime contract、provider-form parity 与 Triton 性能都收敛到同一 terminal executable path 并共用双机结果。
- 当前真正的结构缺口是：Triton descriptor 对 dynamic stride 没有可静态证明的 contiguous 契约，而 Serializer 仍生成 runtime legality/pruner/fallback；cuTile 未消费 TileLang 已能识别的 exact-range full-validity 事实。
- 当前调查不重新打开已无反例的 shared traversal/resource/config/device 问题，也不把合法的 Triton grid permutation、access-to-pointer terminal spelling、`Out` allocation 或 TileLang buffer/storage/copy/pipeline lowering 判为越权。
- Mamba step 的当前 source 明确使用 approximate tanh/sin/cos/sigmoid，而 DSL 表达精确数值语义；该差异不能驱动 approximate-math 隐式改写，最终是否可比由 current source 对照和 fresh 运行决定。

# 待解决问题

无。

# 验证预期

- 每个语义完整实现节点都用一条现有 production compile/emit/JIT/launch 路径做数值 repro 后继续，不建立测试体系。
- 终局在本机 RTX 5090D 与 `ssh h100` 的 `/home/kingdom/.venvs/intentdsl-mlir20/bin/python` 下确认 Triton 3.6，分别执行完整 registry；结果保持算法、dtype、ABI、shape、调用次数与计时范围一致。
- 修改 compiler 后废弃之前生成源与 CSV 观测；明显异常项只做一次 fresh same-code 独立复测，不挑选最优数值。
- 受影响的 shared/provider/emitter 边界对照 `ref/triton` 或 `ref/tilelang` 的当前同类实现完成自查，最终给出双方 file:line、具体差异与实际后果。
