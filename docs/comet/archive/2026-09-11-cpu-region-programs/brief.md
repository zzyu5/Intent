# 目标

让 CPU 从已有 task/block 与可编程局部实现，推进到能够编译完整区域算法的 structured program：保留区域、summary、typed state 与访问关系，协调相连计算的分块及数据供应，并与 GPU 实际复用带成立条件的语义分析。以真实 Mojo/Weft 程序和少量算子性能结果交付，不以接口、文件搬迁或图中箭头代替能力。

# 范围

- 按 `doc/` 既有语义贯通 CPU region fold、region scan、summary/transition records、ordered carries 及其必需的坐标、predicate、访问、局部计算与输出连接；不另造作者算法或取消共同 CPU IR。
- 让工作范围与区域分块同时约束相连的 MatMul/统计、source 切片、访问有效域、状态和输出；task、data tile、microtile 与 SIMD width 保持不同职责。
- 将 GPU 区域有效性、identity 和状态性质推导收束为 GPU/CPU 都实际调用的共享分析与局部规则；执行模型各自改写 current program，不复制 GPU topology，不从旁表恢复执行。
- 收紧共享前已发现的浮点零值推导问题，保持 dtype、NaN/Inf、signed zero、顺序、identity 与 effects。证明不成立时保留原程序，不以禁用整条优化或放宽语言语义代替修复。
- 在现有 implementation 绑定与展开机制上，补实际计算所需的输入表示、外围数据供应、共享准备与资源需求协调。只实现有消费者的需求；内部微块和局部 packing 仍由实现拥有，跨计算/任务复用的范围及 lifetime 由外围程序形成。
- Mojo/Weft 消费同一 CPU 区域程序，各自在适当层次展开；保留已有 native artifact、有效调参、数值环境及任务完成边界。必要的 native source/reference 与生产 benchmark 接线属于本轮，不新建测试框架。
- 本轮新增区域计算首先使用 f32 数值数据，配套 bool/index/integer 坐标和状态。保留现有 f32 与 Q4_K/Q8_K 能力，不以此宣称全 dtype、全格式或全设备覆盖。

## Source coverage

需求来源 `/home/kingdom/phdworks/intent-paper/compiler-figure-design.md` 已完整读取，共 903 行；下表包含各节的文字、表格、代码示意、条件与引用的用途。报告提供本轮目标和讨论背景，`doc/` 仍是语言及编译设计权威。链接中的源码是取证参考，论文、图稿与 VTA 材料不自动扩成实现交付。

| 来源单元 | 读取 | 归类与本轮处理 | Spec | 验收 | 覆盖状态 |
|---|---|---|---|---|---|
| §1，图组与叙事安排 | complete | 图的组成、篇幅和编译叙事是背景，不交付论文图稿 | — | — | background |
| §2.1–2.3，输入知识、A/B 方法与知识来源表 | complete | 算法事实驱动程序形成；区分作者、compiler、implementation 和下层工具链 | §1–4 | A1–A3 | covered |
| §3.1，三个执行模型比较表 | complete | 采用 CPU/GPU 语义共性及不同执行承载；DSA 列为未来背景 | §1–2 | A1 | covered |
| §3.2，task/data tile/microtile/SIMD | complete | 保留 CPU 分层，补区域计算能力，不将 SIMD 作为 family 定义 | §2、§4 | A1、A3 | covered |
| §3.3，DSA 代表性模型及 VTA 链接 | complete | 未来显式本地存储/引擎模型；本轮不实现 DSA 或移植 VTA | — | — | non-goal |
| §4.1–4.3，事实、约束、选择、反馈与输出 | complete | 目标和局部实现约束参与有限候选及数据/资源组织，输出真实程序 | §3–5 | A2、A3 | covered |
| §5.1–5.4，主图标题、布局和公共入口 | complete | 图面设计为背景；KIR 与执行模型边界纳入能力 | §1–2 | A1 | covered |
| §5.5，GPU 分支及绑定示意 | complete | 保持既有 GPU 程序，作为共享规则的实际消费者，不全面重写 GPU | §3 | A2 | covered |
| §5.6，CPU 分支及程序示意 | complete | 区域遍历、状态、供应和选定计算实现进入 CPU 程序 | §2、§4–5 | A1、A3 | covered |
| §5.7，DSA 分支 | complete | 未来模型，不增加本地引擎、DMA 或同步协议实现 | — | — | non-goal |
| §5.8–5.9，工具链边界与 A/B 标记 | complete | 编译层职责纳入；图面位置、颜色及标记仅为背景 | §1、§5 | A1、A3 | covered |
| §6.1–6.2，attention 工作集与尺寸表 | complete | 采用轴/使用/资源关系；32/64 等数字及 f16 容量仅为示意，不固定候选或性能门槛 | §2、§4 | A1、A3 | covered |
| §6.3–6.4，局部实现与结构优化 | complete | 局部高性能实现与外围复用、遍历和状态优化协作 | §3–5 | A2、A3 | covered |
| §7.1–7.7，机制 A、轴传播及各目标说明 | complete | 相连计算/访问/状态的一致分块；CPU/GPU 实际改写，DSA 对应仅背景 | §2–4 | A1–A3 | covered |
| §8.1–8.5，机制 B1、五区域例子及条件 | complete | 精确范围加完整 identity/effect 证明才缩短遍历，不能将区域数量比当加速比 | §3、§6 | A2、A4 | covered |
| §8.6，B1 版式 | complete | 只作解释，不交付绘图 | — | — | background |
| §9.1–9.3，机制 B2、首段与 payload carry | complete | 逐输出成立的首段非空及后续不变量支持状态精简；保留空域与原 combine | §3 | A2 | covered |
| §9.4，B2 版式 | complete | 只作解释，不交付绘图 | — | — | background |
| §10.1–10.4，复用对象、表格与措辞 | complete | 将方法复用推进为 GPU/CPU 的真实规则复用；family 改写与 provider 实现保持分开 | §1、§3–5 | A2、A3 | covered |
| §11.1–11.2，GPU/CPU 完整实例化 | complete | CPU 补当前缺失的区域程序；GPU 保留已有执行路径并接共享分析 | §2–5 | A1–A3 | covered |
| §11.3，DSA 完整实例化 | complete | 未来路线；不接 DSA backend 或完整 attention 引擎 | — | — | non-goal |
| §12，TileLang 对照与引用 | complete | 参考其真实 infer/lower、layout/buffer/resource 连接，不移植 GPU 布局或完整候选系统 | §4–5 | A3 | covered |
| §13.1–13.6，正文、caption 与辅助例子 | complete | 编译方法由上列范围覆盖；文本、图注、论文修订不在本轮 | — | — | background |
| §14.1–14.4，绘图细节与证据表 | complete | 能力证据按 A1–A4 提供；标签、配色、排版不作为编译交付 | §6 | A4 | covered |
| §15，当前事实与完整形态边界及附属报告链接 | complete | 不将设计可能性、旧报告或上轮验收当作现有 CPU/DSA 覆盖 | §1、§6–7 | A1、A4 | covered |
| 本次用户同意架构判断并要求新 worktree | complete | 新 change 独立于 TritonBench；包含共享规则语义修正，不扩 DSA/微核库 | §1、§7 | A1–A4 | covered |

# 非目标

- 不修改作者可观察算法、dtype/近似合同、logical chunk ABI、顺序或 effects；不为方便 lowering 改写 `doc/` 为现状。
- 不创建统一 GPU/CPU/DSA physical IR，不让 Weft 必经 Mojo SIMD lowering，不增加旁路 planning program 或 emitter/runtime 算法。
- 不扩全量 dtype、量化格式、AMX/IME 库或 DSA/Ascend；不把论文图稿、正文、TritonBench agent 实验并入本轮。
- 不全面重写 GPU，不整仓搬文件，不新增全局求解器或无人消费的 capability/requirement 字段。
- 不新增性能 benchmark 以外的测试，不扩展输入矩阵或全量重跑；不预设 1.05/1.1 等旧 change 性能门槛。
- 外部 `ref/` 与 intent-paper 保持只读；TianchenRV 仅允许已确认的 CPU 区域程序所需基础操作及 lowering 补齐，独立提交并保留并发修改，不扩其它外部改动。

# 验收示例

- A1：真实作者区域程序沿 canonical KIR→CPU→Mojo/Weft 形成可执行程序，支持 region fold 的 typed summary/identity/combine 以及 region scan 的 transition、initial state、apply/emit、final state；区域坐标、captures、相连 MatMul/统计、访问、尾部和任务完成语义不丢失，不由 serializer/runtime 补算法。
- A2：GPU 与 CPU 在各自当前执行范围上实际调用同一组区域/identity/状态分析规则，证明成立时产生有效遍历、谓词或 carry 的实际简化；不成立时保留原语义。浮点零值传播不会在缺乏证明时抹去 NaN/Inf 或 signed-zero 行为，不能以复制 GPU pass 或关闭整个优化冒充复用。
- A3：相连计算的分块、访问、状态与所选 implementation 需求使用同一候选 binding；外围供应、共享准备、资源 owner 与 lifetime 在 current program 中形成，并被实际 Mojo/Weft 消费。实现内部的微块与 packing 保持可编程，既有局部实现继续复用，无 kernel-name/source-template 选择器或两套并行执行路径。
- A4：通过现有生产 benchmark 入口获得代表性 CPU region-fold 与 region-scan 程序的真实 generated/source ms、G/S 和同次既定容差结果；Mojo/Weft 的 native 调用均有实际区域程序运行证据。相同案例使用同算法、shape、dtype、资源预算及完整调用范围，数据进入项目 CSV；受影响 GPU 和既有 CPU 项只复用必要性能运行，未受影响结果保留，不以生成成功代替执行或性能结果。

# 约束与不变量

- `doc/index.md` 及 `doc/compiler/cpu-program-ir.md`、`passes-and-analyses.md`、`target-lowering.md`、DSL region/state/numerics 章节是设计权威；报告是目标与证据来源，不反向改变语言。
- 共享的是当前程序事实、带前提的分析规则与合适的改写工具。GPU/CPU 分别形成自身 ownership、range、carry、存储与调度；analysis 失效后重算，不能回读 KIR 重建已丢失的执行结构。
- 本轮的横向能力是可组合的区域计算、访问和状态，不是一个 attention kernel 模板。小算子的 SIMD、矩阵微程序和量化实现继续使用既有分层。
- 工作集估计依据实际表示和 lifetime，不能把逻辑 tensor 大小直接相加当硬件存储量；tuning 参数必须有实际消费者，候选使用目标机的完整调用时间选优。
- 自查对照本项目与 `ref/triton` 或 `ref/tilelang` 的同类实现，记录具体 file:line、差异与后果；不将每条内部发现变成独立验收项。

# 决策

- 用户已接受上一轮架构判断：CPU 基础不推倒重来，优先补区域程序、共享语义分析和有实际需求的外围协调；DSA 实现与微核数量扩张不并入。
- 用户明确要求新 worktree。已创建 `comet/cpu-region-programs`，目标分支 `main`；主目录的 `agent-tritonbench-evaluation` 及未提交实验文件保持原样，不 cherry-pick 或合入本 change。
- 采用单个普通 Native change。区域 carrier、关联分块、规则适配和 implementation 需求都会修改同一当前程序的接口及依赖，拆成独立交付的集成成本较高；不创建 Supervisor 或额外子 worktree。
- 当前 CPU 仅有受限 reduce 和普通 f32 contraction，RegionFold/RegionScan 与 predicated source load 没有正式 lowering；`formTile` 已有真实展开，`legal` 已有尺寸约束，但不等于完整供应/资源协调。GPU 区域推导仍依赖 GPU 类型；部分位于 CPU 目录的变换实际只被 Mojo 调用。这些是起点，不是允许保留的最终缺口。
- 参考依据：TileLang `tilelang/tileop/gemm/__init__.py:121–139` 分开 infer_layout/lower，`src/transform/lower_tile_op.cc:1134–1151` 实际连接 layout、buffer、workspace 与 alignment；Triton `python/tutorials/06-fused-attention.py:55–110` 由作者显式组织 causal stages。本轮复用的是约束与程序连接方法，不宣称下层能自动推导 Intent 的区域语义。
- 用户已明确确认完整 change 与 Build 范围，授权按本 Shape 和 A1–A4 持续实现；普通实现选择由执行者依据 doc/ 与 ref 决定，不再就已确认范围重复询问。
- 用户在收到 Weft 外部修改请求后明确同意继续：允许在 TianchenRV 补齐逐元素 select、区分浮点 maximum 语义及相应 lowering，以同一正式 CPU 区域 benchmark 验证，完成后独立提交；不以浮点乘加替代 select，不引入 Intent 专用路径。
- 用户进一步确认动态私有状态的有界窗口读写及对应 physical IR/pass 扩展。该实现属于通用 Weft 编译能力，复用现有 descriptor、layout、访存选择与资源/lifetime 机制；保留完整状态与未写窗口的值，不以整算子模板或全量标量化替代。必要的 extent 绑定和 local-state selection 修正一并完成，外部修改仍独立提交并保留并发修改。

# 待解决问题

无未解决的需求或授权问题。

# 验证预期

Shape 不运行 benchmark 或编译检查。Build/Verify 只复用正式 `repro.v2.runner` 及现有 native runtime；新增能力缺少 registry 接线或 CPU-native source 时，在现有 provider/source 职责下补最少的生产接线及独立参考程序，不另建临时或永久测试框架。代表程序覆盖报告中的区域 MatMul/统计/摘要组合与有输出的 region scan，沿用现有作者算法，必要时增加同算法 f32 实例化。

CPU 记录完整 native invocation，包含每次调用所需准备、内部物化、任务派发与 join，排除编译、tuning、部署、加载和外部输出分配；generated/source 使用相同 shape/dtype/线程预算和计时范围。同次只做一次约定容差检查，已有容差不变，不追求逐操作或 bitwise；真实超差修复后只重跑受影响项。GPU 受影响路径沿用原 CUDA graph/kernel 计时。未测能力如实标记，不虚构性能达标，不因输入范围示意而扩测试矩阵。
