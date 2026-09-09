# Outcome

本 change 于 2026-09-09 按用户明确要求停止归档。已提交的横向优化及后续 cuTile 性能改进保留在 `main`；本任务移出 active change 集合，后续分析与行动另行确定范围，不自动恢复本轮。

这是未完成任务的搁置归档，不是验收通过归档。Native Archive 因未到 archive-ready 且缺少 verification.md 而不能执行；按用户要求，仅归档正式文件。`comet-state.yaml` 原样保留停止前 Runtime 的 Build/active 快照和 A1—A3 待验结论，不代表本任务仍在执行，也不把后续 CSV 的改善追溯为本轮正式验收。本 Spec 不安装为当前 capability，以下正文仅保留历史目标。

横向修复现有 Intent compiler 与作者 kernel 的共同性能问题，让相同语义和执行结构在各实际使用点及已有 target 中得到一致处理；保留能真实选择有效参数的有限 autotuning 与 provider JIT。同时，将用户确认时两张 cuTile CSV 中 G/S >= 5 的全部八个设备条目逐项收束到 G/S <= 1.1，数值保持现有容差。局部作者优化、结构重构和候选数量减少都不能单独替代这些结果。

# Scope

- 从已合入的 cuTile execution closure 和 chunk gated delta 作者精度改进继续。按共同原因组织实现，不按最慢 kernel 排队逐个打补丁；调查直接服务于同类问题的完整实现，不另开只交报告的调查轮。
- 横向覆盖四类问题：作者表达和矩阵输入/累加精度；shared program 的 blocking、ownership、循环状态、traversal、复用与 materialization；shared 与已有 Triton/cuTile/TileLang lowering 的职责；shared/provider 参数分类、候选构造、JIT、实测选优及缓存复用。按实际缺口决定改动位置，不预设所有问题属于 leaf，也不把所有 leaf 优化机械上移。
- 作者 kernel 可以直接改进显式中间精度和不合理表达；保持语言语义不等于冻结现有作者写法。算法确有变化时，保留已被 Triton 使用或已与 Triton 对齐的算法；无 Triton 使用的 cuTile 专用算法可向 baseline 对齐。
- 默认表和覆盖文件都是候选数据。有限候选必须覆盖有意义的参数变化，真实影响生成程序并由 provider 实测选优；检查无依据的单候选收缩、组合膨胀和重复工作，不预先固定统一候选数，也不以移除 JIT 或调优作为降成本手段。
- 性能硬目标只绑定 Spec 列出的八个设备、entry、case：RTX 5090D 的 chunk_gated_delta、nvfp4_quantize，以及 H100 的 grouped_flash_decode、mla_prefill、attention_sink_prefill、gemma_prefill、gemma_decode、splitk_mla_decode。集合在确认时锁定，不随 CSV 更新重新筛选。
- 复用现有 production runner 和项目 CSV。已有同 case 的 Triton/cuTile 运行可用于归因，不能仅凭两边都慢判定 shared 有错，也不把新建完整双 target 运行矩阵作为实现前置条件。

# Non-goals

- 不宣称完成无限范围的 compiler 成熟化，不新增后端、语言语义、图级结构 autotuner或整套下层机器 compiler；不为一次性能差距重建已存在的 typed carrier。
- 不要求所有 registry 条目都达到 1.1，不扩展硬件或输入矩阵，不为每次修改重跑双机全量；不增加独立正确性测试、pytest、fixture、临时测试脚本、候选耗时表或证据报告。
- 不恢复旧 runtime candidate policy，不按 kernel/source 名称、单条 shape 或历史 winner 写 compiler 特例，不把 cuTile 现有 CTAs=1、occupancy=2 的运行覆盖固化为通用最优策略。
- 不重开已归档任务、不安装停止归档中的旧 Spec，不自动 push、创建 PR 或另建 worktree。

# Acceptance examples

- A1：四类横向问题在现有实现中按共同语义和执行结构得到处理；每项实际修复覆盖所有适用位置及已有 target 消费者，共同事实由唯一 authority 提供，target-local form 留在对应边界。作者层问题在作者层改进，不以单 kernel 特例或局部性能收益冒充 compiler 通用能力。
- A2：受影响程序的可调 shared/provider 参数形成有限、合法且有意义的完整候选，真实改变相应计算或 provider 选项，由保留的 provider JIT/autotuner 编译并按实测耗时选择和复用 winner；固定维度或合法性只允许单候选时保持真实限制，无依据的硬收缩、无效组合或重复工作不能成为默认调优策略。
- A3：Spec 锁定的八个设备条目全部使用现有性能入口完成同算法、同 case 的 generated/source 比较，在原容差内通过且每项 G/S <= 1.1；结果随完成写入对应项目 CSV。任何超标或失败均保持该目标未完成，不用平均值、删除条目、放宽容差、更换输入或比较口径代替达标。

# Constraints and invariants

- `doc/` 是语言和 compiler 设计权威。Shared policy 消费 current typed semantics、def-use、坐标关系、effects、reuse/lifetime、physical facts 和能力约束；serializer 只拼写已确定的程序，不重建执行事实或偷偷改变精度。
- 对照实际 `ref/triton`、`ref/tilelang` 及 provider 实现，说明双方 file:line、具体差异和实际后果。参考用于职责和机制，不照搬 target surface 或将 baseline 变成语言规格。
- 同算法、同输入 shape、外部 dtype 和明确的完整调用/计时范围；细微舍入、近似数学和中间精度差异注明即可。双方可独立调优，候选编译与搜索耗时不算算子时间。
- 同一次受影响性能运行内做一次现有容差检查；不放宽容差。资源允许时并行准备，控制最终计时干扰，不让所有 worker 从准备到完成全程串行。
- 只保留必要代码、既有性能表和 Comet 正式产物；不维护 tmp 交付物，不做 hash/checksum 校验或未经要求的目录整理。

# Decisions

- 用户确认横向改进与性能攻坚同时承担，且确认以两张现有 cuTile 表中 G/S >= 5 的全部条目组成 <= 1.1 的硬验收集合。
- 用户确认允许减少 tuning 数量，但必须保留实际有效的 shared/provider 参数搜索、provider JIT 和实测选优，参考 Triton/TileLang 的真实机制。
- 旧 cutile-provider-closure-round-six 已按用户明确要求停止归档；其未验收快照不构成新的执行要求。cutile-execution-closure 已完成，本 change 不重开它。
- 使用当前 main 目录创建一个普通 Native change。四类问题共同涉及作者表达、shared 参数化程序和 target 消费者，性能归因与修复需共同演进，暂不拆 Supervisor 或按 kernel/target 建立独立子任务。
- 用户已明确确认完整 Shape 并要求进入 Build；三个结果型验收项与锁定的八个设备条目保持不变。
- 用户要求本 change 完成后，下一轮单独收束 cuTile 全部性能条目至 G/S <= 1.05；本轮仍按固定八项 <= 1.1 验收，不提前扩大范围或宣称完成。

# Open questions

无。

# Verification expectations

- 仅使用现有 `bash examples/run/baseline-v2.sh <target> <project-csv> <affected-entry>` 生产入口，完成 emit、provider JIT、运行、同次容差检查与算子计时；按实际修改和硬目标选择受影响条目，不增加另一套验证入口。
- 横向覆盖以真实代码/IR 关系和 reference 对照说明；性能数字负责证明运行结果，不能单凭代码归属、静态数量或候选声明证明实际收益。
- 重用未变化的已有结果，完成一项就更新项目表格。八项硬目标未全部满足时如实保持未完成；不为结束 Comet 流程伪造通过。
