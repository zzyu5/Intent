# Outcome

通过通用 analyses/passes、合理的作者表达和真实有效的有限调优，收束当前 cuTile 算子的性能缺口。现有 RTX 5090D 与 H100 可比较的全部 72 个设备条目，在原有数值容差和既定算法、case、计时范围下逐项达到 generated/source GPU 执行延迟比值 G/S <= 1.05；编译成功、局部提速或增加参数本身不代表完成。

# Scope

- 接续 `compiler-cross-target-optimization` 已提交的实现和未完成问题。旧 change 保留未完成状态，不伪造通过、不在本次创建中归档；新工作在独立 worktree 中进行，不并行修改旧工作区。
- 完整目标 Spec 固定创建时 `examples/repro/v2/registry.py` 的 37 个 cuTile entry 及其 case。RTX 5090D 全部 37 项、H100 除两个已确认硬件限制外的 35 项都承担 <= 1.05 的硬目标，不仅处理上一轮的三个余项或当前最慢条目。
- 按共同执行事实横向修复：program mapping、ownership、blocking、循环状态、矩阵乘依赖与合流、访问/复用、生命周期和物化。由 analysis 判断合法性与收益，由 transformation 改写当前 IR；同类使用点及已有 target 消费者一致处理，不以 kernel 名称选择现成算法路径。
- Shared GPU passes、provider-local legalization 和下层 compiler 各自承担已有规格定义的职责；不把线程布局、机器流水与寄存器分配全部搬进 shared，也不把 shared 执行事实交给 serializer 补猜。
- 保留参数化 IR、有限合法候选、provider JIT、实测选优与缓存复用。Tuning 参数可以驱动已有 passes 改变具体布局、分块和流水，但候选数量不能替代缺失的 transformation。
- 允许用现有 DSL 改善作者中间精度和低效表达，保持外部语义、ABI 和原容差。沿用既定算法比较边界；不预设需要新增 approximate-math 接口，也不把作者 kernel 冻结为不可修改。
- 复用现有 production runner 和两张项目 CSV，修正将 NVFP4 GPU 执行时间写成“一次 kernel launch”的模糊注释。计时是算子 GPU 执行延迟；CUDA Graph 是执行组织方式，CUDA event 是设备计时工具，不把 CPU 提交耗时、JIT 或候选搜索耗时冒充算子性能。

# Non-goals

- 不新增后端、硬件、输入矩阵或语言语义，不复制另一套机器 compiler，不构建任意 IR 图结构的笛卡尔积搜索。
- 不要求 Triton/TileLang 全表达到 1.05，不新建完整双 target 对照矩阵作为当前实现前置条件；受共同改动影响的已有消费者仍需保持正确 lowering。
- 不为 H100 增加其缺失的 E8M0 scaled MMA 或原生 FP4 source 能力；两个既定限制保持可见，不将其他失败新增为豁免。
- 不通过删除条目、减小输入、放宽容差、弱化 source、变更测量范围或平均比值达标。原容差之外的真实错误必须修复。
- 不增加独立测试、pytest、fixture、临时测试脚本、候选耗时表、进度报告或额外性能工具框架；不为每次改动重跑全表。
- 不自动合并、推送、创建 PR、删除旧 worktree 或归档旧 change；不改用户级子代理配置。

# Acceptance examples

- A1：性能修复落实为符合 `doc/` 的作者表达或通用 IR analysis/transformation；适用位置及已有 target 消费者一致处理，共同事实保持唯一 authority。每项 compiler 修复能指出实际 IR 改写、适用条件以及 Triton/TileLang 同类机制的具体对照，不存在按算子名、source 模板或 serializer 分流替代优化。
- A2：受影响的 shared/provider 参数形成有限、合法、有实际编译作用的完整候选，由保留的 provider JIT 和 tuner 实测选择并复用；参数通过相应 compiler passes 或 provider 编译选项落实为具体程序，不以固定 winner、无依据收缩、无效组合或不断增加候选掩盖结构缺口。
- A3：Spec 固定的 72 个可比较设备条目均完成原容差检查和同算法、同 case 的 GPU 性能比较，每项 G/S <= 1.05，结果位于对应项目 CSV；两个既定 H100 硬件限制仍明确可见。计时不混淆 CPU launch 与 GPU 执行，完成一项即更新该项；任何其余失败、空值或超标均使本目标保持未完成。

# Constraints and invariants

- `doc/` 是语言和 compiler 设计权威；source、作者例子、registry、CSV 和历史记录用于暴露问题，不反向定义 compiler policy。
- 决策基于当前 typed semantics、def-use、坐标/访问关系、effects、alias、reuse/lifetime、physical facts 和能力约束；条件选择必须落实为 IR 变换或已声明的 provider-local form。
- 同算法、输入 shape、外部 dtype、完整 callable 范围与原容差。细微舍入、FTZ、近似数学和中间精度差异可注明后比较，不因此跳过计时；这不授权隐式改变 DSL 语义。
- 只运行得到性能所需的 emit、编译、预热、GPU 执行与同次原容差检查。准备可合理并发；provider autotuning 和最终计时都应避免其他 GPU 工作干扰，不把计时隔离扩大成所有 worker 全程串行。
- 已达标条目仍在完整验收集合中。已有结果仅在相关执行与计时路径未变化时复用，不据历史数字声称受影响的新程序已达标。

# Decisions

- 用户要求以新 change 继续解决 cuTile 性能，并确认沿用此前提出的全表 <= 1.05 目标与 pass 化优化方向。
- 原计划沿用 `main`，但 Runtime 禁止在已有 active change 的同一目录创建第二个 change；用户随后明确同意独立 worktree。分支为 `comet/cutile-pass-performance-closure`，目标分支为 `main`。
- 采用一个普通 Native change，不拆 Supervisor：各问题共同影响 shared/provider 编译链与同一性能集合，无法把结果型验收清楚拆成互不重叠的独立交付。主代理负责调查和实现，不把频繁子代理派发作为推进前提。
- 当前阶段仅建立 Shape；旧 change 的未完成验收不因新 change 创建而变成通过。新 change 的完整范围、两个硬件限制和三个验收结果仍待最终 Shape 确认。

# Open questions

- [blocking] CONFIRM: 确认按一个普通 change 推进：现有 37 个 cuTile entry 的双机范围中，72 个可比较设备条目全部 G/S <= 1.05，保留两个已确认 H100 硬件限制；通过通用 analyses/passes 与必要作者改进收束性能，保留有效 tuning/JIT 和原容差，只使用现有性能入口。确认后进入 Build。

# Verification expectations

- 仅复用 `bash examples/run/baseline-v2.sh cutile report/baselinev2/cutile-<device>.csv [entry ...]`，在各对应设备上获得所需结果；不扩展测试矩阵，不要求一次进程重跑整表。
- 核对真实 current IR/代码与环境中的 `ref/triton`、`ref/tilelang` 同类实现，指出双方 file:line、差异与实际后果。Compiler 边界、数值和性能结论分别说明，不把已存在的 pass 名或 source 差异当成收益证明。
- 逐项更新 CSV；未变化且仍适用的结果可复用，受影响条目通过同一性能入口更新。外部负载干扰、编译超时或数值失败保持可见，不通过继续重复测量或改写阈值制造完成结论。
