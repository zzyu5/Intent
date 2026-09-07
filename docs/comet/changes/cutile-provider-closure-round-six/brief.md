# Outcome

在不改变 DSL 与 canonical KIR 语义的前提下，推进 shared GPU Program 到 cuTile provider/runtime 的唯一 executable path，修复实际可达的 lowering 与运行缺口。Generated/source 算法相同即可开展性能比较，细微数值实现差异和额外包装工作如实注明，不统一阻断计时；已完成的 RTX 5090D 与 H100 结果及时写入项目 CSV。`generated_p50_ms / source_p50_ms <= 1.05` 保留为性能改进目标，不以双机全量重跑或所有条目同时达标作为继续推进、更新表格的前提。每项改动的 shared 或 provider-local 归属由真实输入程序、语义缺口和 reference 决定，不预设必须只改 leaf，也不重新制造 shared/core 或 Serializer 权威。

# Scope

- 在已合入 compiler foundations 收尾的当前 HEAD 上，使用现有 production runner 推进受影响的 cuTile entry；仅在对应运行需要时核对环境与 checkout/build。已有 CSV 是对应运行的观察，保留未重跑标注，不能冒充当前代码已重新执行的结果。
- 对 current registry 中可达的 access、gather/scatter、reduce/scan、MMA/scaled MMA、atomic、ordered control、stream/ragged 等形态，对照登记的同语言 source 与实际安装的 cuda-tile 实现，补齐能够由现有 shared program 承载的 provider-local legalization、typed form、verifier、serialization 或 runtime binding。
- 逐项定位 opaque timeout 和 terminal failure 的真实一侧与阶段。若差距来自 current Physical Program、完整 typed config/candidate 或 provider form，修复可复用的语义类问题；若 source 本身也不能在同环境成立，保留准确的 source、hardware 或 lower-compilation-cost 状态。
- 保持 provider rewrite 后的 current cuTile program 自足且可验证：所有执行决策在 serialization 前成为 op/type/attribute/operand/result、typed parameter/config 或 launch artifact；Serializer 只作确定性 API spelling 和已声明 wrapper 绑定。
- RTX 5090D 与 H100 每完成一项运行就更新 `report/baselinev2/cutile-5090.csv` 与 `report/baselinev2/cutile-h100.csv`；未运行项保留此前观察并注明，不等待完整 registry 或同一提交全量重跑。
- 如果 fresh cuTile entry 暴露了可达的 shared correctness 或缺失 semantic carrier，修复唯一 shared authority 后再继续；不得在 cuTile leaf 从 KIR、名称、shape 或邻接结构重建该事实。
- 对每个实际修改点，先区分缺少 shared executable facts、已有 facts 缺少 provider realization、以及 external compiler 的机器 lowering。Provider pass 使用公共 GPU/scf operations 做等价展开不构成 shared 架构越界；不得为了避免修改公共文件而在 leaf 重建事实，也不得把纯 leaf 缺口上移为 shared 语义限制。
- 允许随可达 lowering 问题做有界、渐进的职责重构，删除被新 authority 替代的重复推导和旧 executable path；不按文件行数整理目录，不另开一轮无具体反例的共享架构重建。
- 性能比较先确认算法与输入规模、外部 dtype 等基本条件；舍入、FTZ、近似数学和中间精度的细微差异注明即可，不要求逐操作数值契约相同。辅助输出、布局转换、workspace 与调用计时范围如实注明，区分 kernel 与端到端时间；真实 NaN 或错误结果仍是实现缺口。
- 候选入口仅在实际暴露配置或性能缺口时调整，双方可以分别调优；不要求相同的完整搜索空间，不以穷举候选作为运行或发布结果的前提，也不按单个 winner 或旧 timing 暗中筛选。
- 真有算法差异时，保留已被 Triton 使用或此前已与 Triton 对齐的作者算法；没有 Triton 使用的 cuTile 专用算法可向 cuTile baseline 对齐。该授权只改变相应作者算法，不改变语言语义或共享 helper 的既有算法。

# Non-goals

- 不实现 TileLang 第七轮，不重开已经闭合且没有 current 反例的 shared traversal/resource bounds、compile-device binding、TileLang config、sparse shorthand、TileLang scaled form或 cuTile post-rewrite MLIR verification。
- 不为 kernel/entry/source ID、单一 shape 或某张旧 CSV 增加 matcher、adapter-local candidate filter、source template 或第二条 executable path。
- 不把 cuTile 专有 API、tile-index 限制、warp/lane/MMA layout、底层 pipeline 或 tuning winner 上移为 DSL/KIR 语义或 provider-specific shared policy。共同的 fragment、program identity 与 external typed capabilities 保持 `doc/compiler/` 规定的职责；不为追平 source 偷改共享算法、外部 dtype 或隐瞒 ABI、调用次数和计时范围差异。
- 不把 fixed tuning table、profile 组合或缺少 cost model 本身定义为缺陷；只有 current 运行证明它们真实阻塞 lowering 或性能时才修改对应 typed candidate/form。
- 不建设测试体系，不增加 pytest、fixture、兼容层、版本管理、额外证据文档或长期 benchmark 基础设施。
- `1.05` 是稳定可比结果目标，不是 lowering legality gate；不通过挑选最优重复、缩小 registry、伪造 ratio 或把真实实现缺口改写成 unsupported 达标。

# Acceptance examples

- A1：给定 current shared GPU Program 中完整覆盖、边界覆盖和间接索引的实际 registry access，cuTile legalization 只从 current typed coordinates/source axes/validity/fill/effects 选择 native tile 或 checked gather/scatter form；生成程序实际运行后，invalid read 的 fill 与 invalid write 的 no-effect 均保持并通过数值比较，Serializer 不重建 access legality。
- A2：给定 registry 中 cuTile 1.5.0 或同环境同语义 source 已能表达的 structured/control 形态，provider rewrite 后的程序在 serialization 前通过 local schema 与 closed-surface legality，并能 emit、JIT、launch 和数值通过；不能保持 Intent 语义的组合在最早拥有充分信息的 provider/capability 层精确拒绝，同语义 source 的成功不得被误报成 target unsupported。
- A3：给定任一已完成的生产运行，其结果及时进入对应项目 CSV，失败注明已知一侧和阶段，未计时与未重跑项明确区分；不等待双机完整 registry，不用历史记录冒充当前重跑结果。
- A4：给定算法相同的 generated/source，细微数值实现差异不阻断性能测量；已测项记录真实 p50 和 ratio，并注明额外工作与计时范围。高于 `1.05` 的真实差距保留并用于定位 Physical Program、typed config 或 provider form，不用“契约不同”隐去，也不要求全表同时达标才继续推进。

# Constraints and invariants

- `doc/` 是语言和编译器边界的唯一权威；report、archive、registry、CSV、provider source 与安装包只用于定位差异和核对 cuTile 的实际能力。
- canonical KIR 在 physical construction 后不可变；shared GPU IR 是唯一完整 executable authority；cuTile extension只表达真实 provider-local structure，不能复制 program mapping、value/access graph 或 structured semantics。
- compiler policy只读取 current typed program、capability 和完整 parameter/config；unknown 不提升为 exact，不以 silent fallback、数量级更慢的串行替代或 JIT timeout 冒充支持。
- 对照 ref 必须落到双方具体实现、差异和实际后果。`/home/kingdom/phdworks/ref/triton` 与 `ref/tilelang` 的真实 compiler 实现用于核对 carrier、analysis、pass、legality 与 lowering 边界；两台机器实际使用的 cuda-tile 1.5.0 安装源码是 cuTile surface 的直接依据；registry 登记 source 用于同算法运行对照，不能互相替代。
- 性能比较以同算法为前提，保持输入 shape 与外部 dtype 等条件一致；ABI 表示、辅助工作和调用计时范围差异注明，不因细微数值差异一概跳过。Ratio 方向固定为 generated 除以 source，不用候选调优时间冒充成对 p50；该比较口径不放松 compiler 对 Intent 语义的保持要求。
- 保持唯一 production path，按语义完整节点提交；不留下临时源码、cache、远端工作副本差异、调试脚本或旧/新分支。

# Decisions

- 使用一个普通 Native Change，不拆 Supervisor；correctness、provider boundary、timeout/terminal closure、双机性能共同验证同一 cuTile executable path。
- 初建时“已通过 readiness audit，因此后续主要只改 leaf”的前提不再作为改动归属依据。Foundation 收尾证明其明确覆盖的语义类，不证明所有 shared 组合已闭合；当前工作从 fresh 程序事实继续，既不重复已完成修复，也不承诺 shared 从此不可修改。
- 两台机器继续统一使用 `cuda-tile==1.5.0`、`torch==2.13.0+cu130`、`triton==3.7.1` 与现有 cuTile venv；本轮没有版本分叉或安装升级需求。
- 旧链 `35/37`、`34/37` 和当前历史 CSV 的 `14/13 pass` 均不是验收阈值。是否必须实现某个 form 由 Intent 语义、current shared facts、同语义 source 与实际 cuda-tile capability决定，而不是按旧计数补绿。
- 先用 fresh、逐 entry 的生产路径事实选择修改点；明显差距优先修 Physical Program、typed candidate/config 或可复用 provider form，禁止 entry-local 特例。
- 用户在恢复审查后同意有界 Shape 细化并要求继续 Build；随后明确纠正为推进优先、结果增量发布、同算法即比较性能。本 brief 与 Spec 按最新指令修正，双机全量、逐操作契约一致和穷举调优不再作为推进门槛；不创建新的 change 或恢复暂停 stash。
- 暂存的 `StatefulPointwise` 与 metadata `ConstInt -> int` 只是未证明试验。新候选数据遵守当前 JSON profile 边界；是否改变 metadata specialization 需证明实际成本与 ABI 宽度，不能无证据使用普通 i32 参数或把普通循环等同于静态展开。
- Source runtime 候选入口仍可按实际问题调整，但不要求为比较性能先完成完整候选集合对齐；不能通过把真实性能差距标成不可比来完成验收。
- 本次算法核对中，原先多数 source contract gap 实为相同算法的精度或 ABI/包装差异。MoE alignment 双方都是计数、前缀和与散射；MHC Sinkhorn 的 `exp2(x * log2(e))` 与 `exp(x)` 数学定义相同。明确的算法组织差异是 cuTile 专用 `chunked_softmax_bf16` 的 online summary 对 source 三遍扫描，未发现 Triton 使用该作者函数，可局部对齐；Triton 的独立 stable softmax 与共享 online helper 不改。
- 本次只核对算法、清理恢复记忆并修正现有表格与约束，不启动测试、不改 kernel。旧 runner 仍有 `comparison.status != pass` 提前跳过计时的实现，尚待实现轮纠正；本次没有新增计时，也没有宣称 cuTile change 已完成。

# Open questions

无。

# Verification expectations

- 实现过程中每个语义完整节点只使用现有 production runner 对一个受影响 entry 做一次 emit/JIT/launch/numerical repro，确认可运行后继续；不建立测试或额外门禁。
- Shared 改动说明同一语义类的真实 IR 改写与保持条件，对照 Triton/TileLang 同类实现，并选择一条受影响的生产 repro；跨 provider 边界按实际问题定位，不自动扩为全量验证或跨 provider 测试框架。
- Timeout 必须保留 worker 最近进入的 generated/source 及 compile/JIT/launch/numerical/measurement 阶段；进程退出或超时不允许覆盖已知阶段。双方尚不可比时保留具体原因，不用旧 ratio 驱动 candidate 或 shared 改写。
- 仅对实际修改的路径使用必要的手动生产 repro；需要性能调查时测量对应 entry，并立即更新项目表格，不自动扩为双机全量任务。
- 修改 compiler 后不把此前 timing 称为新代码结果，但保留其带来源/未重跑标注的历史观察；明显异常按需做一次 same-code 复核，不挑选最优重复值。不维护 tmp 作为交付物。
- 受影响的 cuTile legalizer/verifier/serializer/runtime 边界必须与安装的 cuda-tile 1.5.0 同类实现及 registry source 对照，留下双方 `file:line`、具体差异和实际后果；最终交付只保留代码、两张 CSV 与 Comet 自身产物。
