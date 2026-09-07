# Outcome

在现有 compiler foundations 上收束 cuTile 的实际编译、JIT/调优、运行并发与遗留实现问题，完成 RTX 5090D 和 H100 上当前 cuTile registry 的全量性能运行。每个已实现且硬件支持的同算法 entry 在约定容差内通过数值检查，并产出 generated/source 算子时间和 ratio；结果随完成随写入项目现有 CSV。性能是运行工作的中心，数值仅保留同一次 benchmark 中必要的容差检查，不扩展测试体系。

# Scope

- 对齐源码、实际 compiler binary、随编译器分发的 profiles、Python provider/runtime 与生产入口，确保当前实现真正用于运行。明确默认 JSON 配置、编译调用覆盖、provider JIT 与 runtime winner 的职责，不恢复旧 runtime candidate policy。
- 在现有 production runner 中支持资源预算内的同机有界多进程并发；不同机器可同时推进。准备、编译、必要初次运行与计时合理分阶段，不能让一个 worker 从头到尾阻塞其他 entry，也不能把并发 GPU 干扰混入算子性能。
- 区分 Intent 编译、provider 首次编译/运行、autotune 与最终 benchmark。减少有证据的重复编译、重复调优和无用候选/specialization，复用已有缓存能力，不取消有语义必要性的 specialization，不把候选耗时作为算子结果。
- 修复同一次性能运行暴露的真实编译、launch 与超出容差的问题，包括 absorbed MLA 的 NaN；补齐当前 registry 的作者实现缺口，包括 NVFP4。H100 上确实不支持的 E8M0 scaled MMA 明确记录为硬件限制，不伪造通过。
- 同算法且数值在容差内即可计时；移除以 FTZ、近似数学、中间舍入、辅助输出或完整候选集合不一致为由一概跳过计时的旧门槛。ABI 表示、布局转换、辅助工作与计时范围差异如实注明。
- 已被 Triton 使用或此前已与 Triton 对齐的作者算法保持；没有 Triton 使用的 cuTile 专用算法允许对齐 cuTile baseline。明确的 chunked softmax 组织差异局部处理，不改共享 online helper 或 Triton 独立 stable softmax。
- 使用 `report/baselinev2/cutile-5090.csv` 和 `report/baselinev2/cutile-h100.csv` 发布性能，不新增历史审计表、候选耗时表或临时报告。局部修复后只重跑受影响项，最终让完整 registry 都有实际运行结论，不因每次改动从头重跑全部。
- 保持 Comet 个人记忆、项目知识、项目规范和所选 change 一致；新 change 的正文可检索，旧 change 的历史测试要求与验收门槛不自动恢复。

# Non-goals

- 不继续旧 `cutile-provider-closure-round-six` 的 Build/Verify，不伪造其完成或删除已有实现。
- 不新增单测、边界、兼容、压力、回归或组合测试；不以 `/tmp`、内联命令或跑完即删绕过限制。
- 不另建 benchmark 平台、证据体系、运行历史台账或项目外测试脚本集合。
- 不重做已完成的 DSL/compiler 大调查，不进行无具体 lowering 或运行问题的重构，不实现新的 provider。
- 不要求逐操作/bitwise 数值一致，不为通过而放大容差；不要求双方穷举相同候选集合或全表 ratio 同时达到旧阈值。

# Acceptance examples

- A1：从明确的当前工作区使用现有生产入口时，compiler、profiles 与 Python runtime 一致，当前配置可物化到可执行 cuTile 程序；不再误用缺少当前接口和配置的旧默认 binary。
- A2：在资源允许的同机多 entry 运行中，多个 worker 能并发推进而非逐个全程等待；最终算子计时不混入其他 worker 的 GPU 竞争，失败或中断不会覆盖已经完成的 CSV 结果。
- A3：当前已实现且硬件支持的 registry entry 在性能运行内完成约定容差检查并给出真实算子时间和 ratio；已知 NaN、缺失作者算法等实现问题得到解决，真实硬件限制准确标注，细微 source 差异不再统一阻断计时。
- A4：从新 change 恢复时可检索当前 brief/spec 与项目规范，并按性能为主、容差内即可、同机有界并发的要求继续；不把个人记忆当全部项目知识，也不重新引入旧独立测试任务。

# Constraints and invariants

- `doc/` 仍定义语言和 compiler 语义；性能口径与运行安排不改变 canonical KIR、shared executable authority 或 provider 边界。
- 修复依赖 current typed facts 与实际 provider capability，对照 ref/triton 或 ref/tilelang 的同类实现，不使用 kernel-name matcher、source template 或第二条 executable path。
- 默认沿用现有 entry 的 dtype/算法容差，只在明确依据支持时调整，不为掩盖实现错误放宽；NaN/Inf 与超出容差保留真实失败，不以数值未通过的结果声称有效加速。
- 编译与预热不计入稳态算子时间；ratio 为 generated/source。输出只使用实际测得的时间，不用候选调优时间代替。
- 不停止无关 GPU 服务，不假定显存小就没有执行竞争；并发度由当前资源和运行方式约束。
- 仅保留必要生产代码、项目规范、现有 CSV 与 Comet 正式产物；不维护临时测试和调试材料。

# Decisions

- 用户停止旧 change，并要求以一个新 change 统一收束编译/config/JIT、运行方案、真实数值问题和项目知识，最终完成 cuTile 全量性能运行。
- 最新数值要求是“容差内即可”，不是完全不检查，也不是独立正确性测试阶段。此前“只性能、不做任何数值检查”的解释已修正。
- 使用单个普通 Native change，不拆 Supervisor：编译入口、runner、计时与修复共享同一生产链路，拆分会增加交接而不能独立完成最终结果。
- 原拟沿用 main，但 Runtime 拒绝在已有未完成 change 的目录中再创建 current change；因此使用 `comet/cutile-execution-closure` 独立 worktree，目标分支 main。worktree 是 change 隔离，不是算子并发的前提。
- 起点已核实：旧 `runtime/tuning/` provider 脚本已删除，候选表在 shared/cuTile 的 `TuningProfiles.json`；provider JIT 仍发生在生成程序首次调用。默认本地 compiler 产物缺当前 `--tuning-config` 与 profiles，需在 Build 对齐，不能仅凭源码 HEAD 宣称已就绪。
- 起点已核实：H100 backward 的 rank-1 register-load 修复已在源码；5090 absorbed MLA 的 orientation/NaN 问题未闭合；NVFP4 缺作者实现；旧未运行条目不能当已定位的新 bug。
- 项目知识准备已完成：active brief/spec 纳入索引，冲突规范副本停用，项目原文与持久记忆已纠正。新 workspace 仍需按其实际来源刷新索引，恢复以新 change 为准。
- 并发实现参考 TileLang 的编译/benchmark 分层，同时尊重当前 cuTile 首次调用和进程内锁的实际边界；不预设不存在的公共 compile-only API。

# Open questions

- [blocking] CONFIRM: 按以上范围进入 Build：对齐实际编译入口，改善同机并发与重复 JIT/调优，修复真实实现问题，完成双机全 registry 性能及同次运行容差检查；不新增独立测试或临时测试矩阵。

# Verification expectations

只使用现有生产 benchmark 入口。每个条目在同一次性能运行中检查一次约定容差，随后测量算子；局部修复只重跑受影响的性能项，最终覆盖双机完整 registry。代码自查直接对照 ref/triton/ref/tilelang 的同类实现并说明差异，不把参考阅读变成新的测试任务。环境、配置与项目知识仅用必要的只读状态/来源核对，不新建验证脚本或证据文档。
