# Outcome

在不改变 DSL 与 canonical KIR 语义的前提下，闭合 shared GPU Program 到 cuTile provider/runtime 的唯一 executable path：当前 registry 中由 Intent 完整表达、且 cuTile 1.5.0 在对应硬件上能够保持同一语义的 entry，必须实际 emit、JIT、launch 并通过数值比较；RTX 5090D 与 H100 上所有稳定、语义和计时可比的结果，其 `generated_p50_ms / source_p50_ms` 不超过 `1.05`。最终两张 cuTile CSV 来自同一 current compiler state；每项改动的 shared 或 provider-local 归属由真实输入程序、语义缺口和 reference 证明，不预设必须只改 leaf，也不重新制造 shared/core 或 Serializer 权威。

# Scope

- 在已合入 compiler foundations 收尾的当前 HEAD 上，重新核对两台机器的 Python/Torch/cuda-tile 1.5.0 环境与 checkout/build，使用现有 production runner 建立 fresh cuTile 状态；旧正式 CSV、暂停期间的临时 CSV、旧链覆盖率与 timeout 只作为调查入口，不作为当前结论。
- 对 current registry 中可达的 access、gather/scatter、reduce/scan、MMA/scaled MMA、atomic、ordered control、stream/ragged 等形态，对照登记的同语言 source 与实际安装的 cuda-tile 实现，补齐能够由现有 shared program 承载的 provider-local legalization、typed form、verifier、serialization 或 runtime binding。
- 逐项定位 opaque timeout 和 terminal failure 的真实一侧与阶段。若差距来自 current Physical Program、完整 typed config/candidate 或 provider form，修复可复用的语义类问题；若 source 本身也不能在同环境成立，保留准确的 source、hardware 或 lower-compilation-cost 状态。
- 保持 provider rewrite 后的 current cuTile program 自足且可验证：所有执行决策在 serialization 前成为 op/type/attribute/operand/result、typed parameter/config 或 launch artifact；Serializer 只作确定性 API spelling 和已声明 wrapper 绑定。
- 准备 H100 的 current checkout/build，并在 RTX 5090D 与 H100 上从同一提交完整运行 cuTile registry，更新 `report/baselinev2/cutile-5090.csv` 与 `report/baselinev2/cutile-h100.csv`。
- 如果 fresh cuTile entry 暴露了可达的 shared correctness 或缺失 semantic carrier，修复唯一 shared authority 后再继续；不得在 cuTile leaf 从 KIR、名称、shape 或邻接结构重建该事实。
- 对每个实际修改点，先区分缺少 shared executable facts、已有 facts 缺少 provider realization、以及 external compiler 的机器 lowering。Provider pass 使用公共 GPU/scf operations 做等价展开不构成 shared 架构越界；不得为了避免修改公共文件而在 leaf 重建事实，也不得把纯 leaf 缺口上移为 shared 语义限制。
- 允许随可达 lowering 问题做有界、渐进的职责重构，删除被新 authority 替代的重复推导和旧 executable path；不按文件行数整理目录，不另开一轮无具体反例的共享架构重建。
- 性能修改前先核对双方完整 callable closure、输入精度、显式舍入、中间 dtype、kernel/额外 tensor 工作、candidate contract 与计时范围。先完成数值与终端归因，再解释 ratio；不把容差通过当作契约等价。

# Non-goals

- 不实现 TileLang 第七轮，不重开已经闭合且没有 current 反例的 shared traversal/resource bounds、compile-device binding、TileLang config、sparse shorthand、TileLang scaled form或 cuTile post-rewrite MLIR verification。
- 不为 kernel/entry/source ID、单一 shape 或某张旧 CSV 增加 matcher、adapter-local candidate filter、source template 或第二条 executable path。
- 不把 cuTile 专有 API、tile-index 限制、warp/lane/MMA layout、底层 pipeline 或 tuning winner 上移为 DSL/KIR 语义或 provider-specific shared policy。共同的 fragment、program identity 与 external typed capabilities 保持 `doc/compiler/` 规定的职责；也不为追平 source 改变算法、dtype、ABI、数值语义、调用次数或计时范围。
- 不把 fixed tuning table、profile 组合或缺少 cost model 本身定义为缺陷；只有 current 运行证明它们真实阻塞 lowering 或性能时才修改对应 typed candidate/form。
- 不建设测试体系，不增加 pytest、fixture、兼容层、版本管理、额外证据文档或长期 benchmark 基础设施。
- `1.05` 是稳定可比结果目标，不是 lowering legality gate；不通过挑选最优重复、缩小 registry、伪造 ratio 或把真实实现缺口改写成 unsupported 达标。

# Acceptance examples

- A1：给定 current shared GPU Program 中完整覆盖、边界覆盖和间接索引的实际 registry access，cuTile legalization 只从 current typed coordinates/source axes/validity/fill/effects 选择 native tile 或 checked gather/scatter form；生成程序实际运行后，invalid read 的 fill 与 invalid write 的 no-effect 均保持并通过数值比较，Serializer 不重建 access legality。
- A2：给定 registry 中 cuTile 1.5.0 或同环境同语义 source 已能表达的 structured/control 形态，provider rewrite 后的程序在 serialization 前通过 local schema 与 closed-surface legality，并能 emit、JIT、launch 和数值通过；不能保持 Intent 语义的组合在最早拥有充分信息的 provider/capability 层精确拒绝，同语义 source 的成功不得被误报成 target unsupported。
- A3：给定同一 current compiler state 和两台已对齐的 cuTile 环境，完整 registry workflow 在 RTX 5090D 与 H100 上结束后，每个 entry 都具有可归因到 generated/source 及 compile/JIT/launch/numerical/measurement 阶段的终端结果；不保留无法定位的宽泛 compile failure 或 opaque worker timeout，两张 CSV 准确记录同一提交的结果。
- A4：给定双机中所有 generated/source 均数值通过且语义、candidate contract 与计时范围可比的稳定 entry，fresh workflow 记录的 `generated_p50_ms / source_p50_ms` 均不超过 `1.05`；明显高 ratio 优先由 current Physical Program、typed config 或 provider form闭合，确实不稳定或不可比者记录具体原因而不是无解释的高 ratio `pass`。

# Constraints and invariants

- `doc/` 是语言和编译器边界的唯一权威；report、archive、registry、CSV、provider source 与安装包只用于定位差异和核对 cuTile 的实际能力。
- canonical KIR 在 physical construction 后不可变；shared GPU IR 是唯一完整 executable authority；cuTile extension只表达真实 provider-local structure，不能复制 program mapping、value/access graph 或 structured semantics。
- compiler policy只读取 current typed program、capability 和完整 parameter/config；unknown 不提升为 exact，不以 silent fallback、数量级更慢的串行替代或 JIT timeout 冒充支持。
- 对照 ref 必须落到双方具体实现、差异和实际后果。`/home/kingdom/phdworks/ref/triton` 与 `ref/tilelang` 的真实 compiler 实现用于核对 carrier、analysis、pass、legality 与 lowering 边界；两台机器实际使用的 cuda-tile 1.5.0 安装源码是 cuTile surface 的直接依据；registry 登记 source 用于同算法运行对照，不能互相替代。
- 性能比较保持算法、数值契约、dtype、ABI、input shape、调用次数、candidate contract 和计时范围一致；ratio 方向固定为 generated 除以 source。
- 保持唯一 production path，按语义完整节点提交；不留下临时源码、cache、远端工作副本差异、调试脚本或旧/新分支。

# Decisions

- 使用一个普通 Native Change，不拆 Supervisor；correctness、provider boundary、timeout/terminal closure、双机性能共同验证同一 cuTile executable path。
- 初建时“已通过 readiness audit，因此后续主要只改 leaf”的前提不再作为改动归属依据。Foundation 收尾证明其明确覆盖的语义类，不证明所有 shared 组合已闭合；当前工作从 fresh 程序事实继续，既不重复已完成修复，也不承诺 shared 从此不可修改。
- 两台机器继续统一使用 `cuda-tile==1.5.0`、`torch==2.13.0+cu130`、`triton==3.7.1` 与现有 cuTile venv；本轮没有版本分叉或安装升级需求。
- 旧链 `35/37`、`34/37` 和当前历史 CSV 的 `14/13 pass` 均不是验收阈值。是否必须实现某个 form 由 Intent 语义、current shared facts、同语义 source 与实际 cuda-tile capability决定，而不是按旧计数补绿。
- 先用 fresh、逐 entry 的生产路径事实选择修改点；明显差距优先修 Physical Program、typed candidate/config 或可复用 provider form，禁止 entry-local 特例。
- 用户在恢复审查后明确同意上述有界 Shape 细化并要求继续 Build。A1-A4、双机完整 registry 和稳定可比项 `1.05` 目标保持不变；不创建新的 change 或恢复暂停 stash。
- 暂存的 `StatefulPointwise` 与 metadata `ConstInt -> int` 只是未证明试验。新候选数据遵守当前 JSON profile 边界；是否改变 metadata specialization 需证明实际成本与 ABI 宽度，不能无证据使用普通 i32 参数或把普通循环等同于静态展开。

# Open questions

无。

# Verification expectations

- 实现过程中每个语义完整节点只使用现有 production runner 对一个受影响 entry 做一次 emit/JIT/launch/numerical repro，确认可运行后继续；不建立测试或额外门禁。
- Shared 改动必须给出同一语义类的真实 IR 改写、保持条件及受影响 Triton/cuTile production repro；触及 TileLang storage/copy 等交界时再核验相应路径。验证直接使用生产 emit/运行，不建立跨 provider 测试框架。
- Timeout 必须保留 worker 最近进入的 generated/source 及 compile/JIT/launch/numerical/measurement 阶段；进程退出或超时不允许覆盖已知阶段。双方尚不可比时保留具体原因，不用旧 ratio 驱动 candidate 或 shared 改写。
- 终局在本机 RTX 5090D 和 `ssh h100` 的 `/home/kingdom/.venvs/intentdsl-cutile/bin/python` 环境下，以同一 current checkout/build 分别完整运行 cuTile registry并写入两张正式 CSV。
- 修改 compiler 后废弃此前生成源码和 timing；明显异常只做 fresh same-code 复核，不挑选最优重复值。
- 受影响的 cuTile legalizer/verifier/serializer/runtime 边界必须与安装的 cuda-tile 1.5.0 同类实现及 registry source 对照，留下双方 `file:line`、具体差异和实际后果；最终交付只保留代码、两张 CSV 与 Comet 自身产物。
