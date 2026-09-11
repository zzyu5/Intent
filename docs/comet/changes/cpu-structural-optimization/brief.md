# 目标

在已有可执行 CPU region/task 程序上，减少无贡献遍历、重复准备和不必要的状态物化，并让 Weft 在合适的计算块粒度接手。交付真实的 current-program 改写及原生产算子的性能改善，不以 pass 数量、公共接口或生成成功代替优化成果。

# 范围

- 有效遍历与状态特化：从当前坐标、访问、summary/identity、combine 和 effects 推导合法范围，把成立的结论写入循环、输入切片、谓词和状态；证明不足的部分保持原语义。
- 准备与状态复用：协调同一 source slice 的多消费者、循环不变量与已声明的输入准备；依据读写依赖、旧状态最后使用和 owner/lifetime 消除重复准备、不必要的整状态复制及初始化。
- 计算块与展开时机：完善 CPU blocking、implementation requirements/binding 和 Weft legalization 的连接，使适用的有界二维 contraction 保留到目标实现接手；不无条件按每个输出元素建立 1×1 临时计算。真正需要的尾部或局部微块继续由合法实现决定。
- 保留 Mojo SIMD/register 路径和 Weft RVV 路径；参数仍由对应 current-program consumer 消费，不把 tuning 当成缺失结构优化的替代品，不扩大无关候选集合。
- 沿用现有 CPU registry 和 generated/source 计时入口。重点处理 Weft linear attention 的明显慢项；共同改动影响其它 CPU 条目时一并修复，只测受影响项并及时更新既有 CSV。

## 来源与范围对应

本次直接需求是本轮对话中已选定的“第一步：CPU pass 相关处理”；报告及旧 change 用于解释背景和定位现状，不重启其全部范围。

| 来源 | 状态 | 本轮决定 | Spec | 验收 |
|---|---|---|---|---|
| 用户要求先做 CPU pass 处理 | complete | 有效遍历、准备/状态复用、计算块组织进入本轮 | §2–4 | A1–A3 |
| 用户纠正“优化思想通用，不要求 CPU/GPU pass 通用” | complete | 各 execution family 改写自身 IR；已有公共分析可复用，扩大代码共享不是交付目标 | §1 | A1–A3 |
| 用户询问成果量化并接受已有 Mojo/Weft 对照 | complete | 分开 generated/source 竞争力与 generated 前后收益，保留明确计时范围 | §5 | A4 |
| 用户接受当前核数预算不是语言模型约束 | complete | Mojo 8-worker、Weft 1-worker 仅沿用为现有性能比较条件，不做核数扩展实验 | §1、§5 | A4 |
| 用户同意暂缓 Intel AMX，IME 留给后续步骤 | complete | 矩阵扩展接入、格式扩张及外部微程序库不并入本轮 | §6 | 非目标 |
| 用户确认基于上一轮 CPU 分支创建 worktree | complete | 独立分支/目录，保留主目录与旧分支 | §6 | 工作区决定 |
| 用户在明确外部修改与独立提交授权问题后回复“继续啊” | complete | 补齐 TianchenRV 中静态有界子视图/私有窗口的通用 stream lowering，独立提交 | §4、§6 | A3、A4 |
| compiler-figure-design 报告 §10 | background | 采用跨执行模型的优化方法，不要求统一 physical IR 或整套 pass | §1 | 背景 |
| cpu-region-programs 的共享代码验收及历史运行结论 | superseded | 只作实现起点；不得作为本轮强制共享代码或已性能达标的依据 | §1、§5 | A1–A4 |

# 非目标

- 不接 Intent→Weft IME、Intel/Apple AMX、DSA/Ascend，不扩全 dtype、量化格式、InOut 或一般索引覆盖；这些不作为本轮前置依赖。
- 不统一 CPU/GPU physical IR，不移植 GPU topology，不以公共 pass 文件数量衡量横向思想复用。
- 不替换作者算法或 summary/scan 定义，不新增有限输入假设、全局 fast-math 或改变 NaN/Inf、signed zero、近似与累加合同来换取裁剪。
- 不按 kernel 名、source 模板或整算子库调用选择执行路径；serializer/runtime 不新建算法、循环、共享缓存或 scratch。
- 不新增测试框架、独立数值/回归/边界/压力测试、逐 pass 全排列消融或全量重跑；不恢复 ARS，不建立额外计划/进度文档。
- 主目录的其它 change、实验产物以及 ref、intent-paper 保持只读。TianchenRV 仅允许本轮已单独授权的静态有界子视图/私有窗口到 RVV stream contraction 的通用 lowering 补齐，独立提交并保留他人改动；其它外部能力仍需单独授权。

# 验收示例

- A1：CPU region transformation 根据当前坐标范围和完整的 typed summary/identity、数值及 effects 证据形成有效遍历与谓词/状态特化；可证明无贡献的区间真实改变循环边界及其专属访问，而非只记录 possibleBegin/End。无法证明的区间保留原计算；空 source、首次 combine、绝对坐标、tail、scan 输出和最终状态不被省略。不得将 mask-false 或一个浮点零操作数单独当作删除整段的依据。
- A2：当前 region-scan 的重复准备/状态物化得到实际改写：在同一 slice 的来源稳定、访问一致且生命周期合法时复用输入供应，在旧状态不再被读取时消除不必要的 next→state 全量复制或重复初始化。生成的 CPU/provider IR 能指出减少的具体工作及 owner/lifetime，所有 summarize/emit/apply 依赖和完整 final state 保持；不以全局字符串缓存代替 IR 复用。
- A3：所选 Weft f32 实现能承接合法的有界二维计算块，CPU/provider transformation 不再无条件为每个输出元素建立 1×1 allocation/fill/contraction。块、输入供应、累加与尾部由同一 binding 和当前 IR 连接，展开结果进入正式 Weft lowering；Mojo 仍按自身需求形成局部微块，不强迫两个 provider 使用相同表示，也不引入整算子模板。
- A4：通过既有性能入口获得受影响 CPU 算子的 generated/source ms、G/S 和同次原容差结果，及时写回项目 CSV；对 Weft linear attention 给出明确结构归因和优化前后 generated 时间，实际改善不能由 source 时间漂移替代。保持同算法、shape、dtype、线程预算和完整 native 调用范围；没有真实性能收益不能把结构改写宣传成性能收敛，未运行项不宣称通过。

# 约束与不变量

- `doc/` 是规格权威；CPU current program 保留 typed ABI、逻辑轴、任务责任、访问、控制、状态与资源。Construction 后不回读 KIR 重建已丢失的执行结构，analysis 在相关改写后失效/重算。
- 跨 CPU/GPU 采用相同优化思想与成立依据，各自形成自己的执行程序。现有 `lib/Analysis/` 可以继续复用；旧 change 的“必须共享实现”不是本轮目标。
- 只有完整 summary 的无贡献性和必要 effects 均得到证明，才可跳过对应区域；只证明 transition 为 identity 不允许丢失 scan 的 emit 输出。不能从 benchmark 输入碰巧有限推断语言前置条件。
- 复用必须保存 snapshot、alias、坐标、输入有效域及旧状态最后使用。资源 owner、初始化、释放及同步在 IR 内显式可见；不隐藏跨调用准备或持久缓存。
- 局部微程序是正式 lowering 的组成，不要求所有优化都在共同 CPU pass 内；按实际影响范围分配责任。共享供应与外围生命周期不能被每个局部实现重复解释。
- 核数预算与 SIMD width、task grain、输入 batch extent 相互独立。保留既有有限 tuning 及 override，参数必须实际改变合法程序。
- 只有算子性能 benchmark 及其同次容差检查；准备允许资源预算内并发，计时避开竞争。已知错误修复后只重跑受影响项，不放大容差，不创建临时或永久额外测试。

# 决策

- 新 change `cpu-structural-optimization` 在独立 worktree 中进行；分支为 `comet/cpu-structural-optimization`，基线及当前集成目标为 `comet/cpu-region-programs`。创建时该基线尚未合入 main，本轮不自动合并 main、推送或创建 PR。
- 保持单个普通 Native change。遍历、状态、供应和 implementation binding 修改同一 CPU region 程序，需共同验收；不拆 Supervisor 或子 worktree。
- 当前代码起点：`RealizeRegions.cpp:275–294` 只消费 all-true 范围；`:195–199,243–265` 为 summary/next 建 scratch 并逐段复制状态；`Weft/Implementations.cpp:20–43` 按输出元素展开；`Weft/Legalize.cpp:438–449,958–959` 的 read 复用局限于单个 operation。这些是待改写事实，不等于已经证明最终机器重复读次数或全部耗时来源。
- 数值边界：`attention_f32.py:14–19,24–35` 的零 probability 与未知 V contraction 不能保证完整 identity，invalid accumulator 也未被短路屏蔽。现有 causal attention 不承诺无条件裁剪全部 mask-false 区域；该限制不能阻止其它有证明的变换或状态/供应优化。
- Reference 对照：Triton `python/tutorials/06-fused-attention.py:54–80` 在 source 中明确限定阶段范围，而 CPU 尚未完整形成相应合法范围；TileLang `tilelang/tileop/gemm/__init__.py:121–139` 与 `src/transform/lower_tile_op.cc:1134–1151` 将选定局部实现、布局和资源连到 lowering。借鉴职责及 IR 连接，不复制数值约定或 GPU 布局。
- 性能起点来自现有表：Weft linear attention 3.423664/1.904748 ms，G/S 1.797437；Mojo 的两条区域程序为 0.278042/0.295968 与 0.200169/0.291698 ms。历史数字用于定位，不以跨时段 ratio 漂移证明优化收益。
- 本轮不默认继承 cuTile 的 1.05/1.1 门槛，也不承诺全部 CPU 条目已经接近 source；实际收益、最终 G/S 和剩余差距分别交付。
- 用户已在完整 Shape、A1–A4 和非目标说明后回复“可以，继续”，确认进入 Build；不重复询问已确定范围，新增外部能力或语义变更仍需单独授权。
- 用户在明确询问 TianchenRV 通用 lowering 补齐、保留他人改动并独立提交后回复“继续啊”，授权上述窄范围外部修改；不扩大到 IME/AMX、算法、数值语义或新测试。

# 待解决问题

无未解决的需求或授权问题。

# 验证预期

Shape 只读调查并维护本 brief/Spec，不编译、不运行 benchmark。Build 复用 `examples/repro/v2/runner.py` 和现有 Mojo/Weft native adapters，优先覆盖两类区域程序的受影响路径；普通 GEMM、归一化或 Q4_K 仅在共享改动影响时运行，不新增输入矩阵。

优化竞争力使用 `G/S = generated_ms / source_ms`；自身收益使用同条件的优化前后 generated 时间，不用 source 漂移或候选调优耗时替代。保留现有线程/缓存/调用边界，计入本次 native invocation 的内部准备、packing、状态、workspace 与 task join，排除编译、调优、加载和外部输出分配。数值只使用同次原容差检查。

IR 差异从正常编译和性能运行已有产物取证，结合 ref file:line 解释变换及后果，不另造运行检查。最终只读复核与 Comet Verifier 分别核对 A1–A4；局部结构收益或单项 benchmark 通过不自动等于整轮验收通过。

Shape 的 `native new/status/doctor` 已确认工作区绑定及状态正常；单独的 `native check` 返回 `Unsupported Native change schema comet.native.v4 for runtime protocol 3`。这是该检查入口与当前状态协议的不一致，不改写 Runtime 状态来规避，也不据此宣称正式产物或实现已经验收；按正常 `next` continuation 推进 Build。

## Build 事实与当前缺口

- CPU 已实现 typed uniform summary/identity 证明及 possible 区间消费；证明未知时保留原计算。完整逐元素 writer→copy 可直接写入私有目标，Weft 同作用域只读供应可跨消费者复用；整块 private write 与窗口 write 均保留显式 owner/update。生产编译入口 `--stop-after-shared` 显示 linear attention 两个既有候选的 `memref.copy` 从 12 个减到 4 个，删除完整块/尾块内的 next→state 整状态复制，保留外围输出/snapshot 复制。
- Reference：`ref/triton/python/tutorials/06-fused-attention.py:54–80` 用显式阶段上下界避免无贡献遍历；本轮 `RealizeRegions.cpp:274–301` 则仅在完整 identity/effects 证明后改变边界，不从浮点零掩码推断无贡献。`ref/tilelang/src/transform/storage_rewrite.cc:301–325,951–985` 依据逐元素依赖、生命周期及存储条件复用目标；本轮 `FuseStructuredComputations.cpp:283–352` 同样保留旧状态最后读取及 snapshot，不无条件原地更新。这里的 ref 位于 `/home/kingdom/phdworks/ref`，不是当前 worktree 子目录。
- 同条件 Mojo linear attention：改动前 generated/source 为 0.098131/0.162892 ms，最终实现为 0.083348/0.154081 ms，generated 耗时约下降 15.1%，G/S 0.540938。最终 attention 为 0.278868/0.279233 ms，G/S 0.998690；GEMM 为 2.183901/2.144156 ms。均通过各自同次原容差检查，已写入 `report/baselinev2/mojo-x86.csv`。没有用更早表中 0.200169 ms 的跨时段差值宣称收益。
- Weft linear attention 的本轮改动前实测为 3.437264/1.901288 ms，最终实现为 1.260196/1.906588 ms，G/S 0.660969；generated 耗时下降约 63.3%（2.73×），source 基本不变。Causal attention 最终为 3.354785/109.410910 ms，G/S 0.030662。均由原生产 benchmark 获得且同次原容差通过，已更新 `report/baselinev2/weft-rvv.csv`；Q4_K 原记录未改动。
- Weft implementation 已从无条件 1×1 改为 binding 中的有界微块，当前 panel 不超过 4；K=1 的真实尾部继续使用局部标量收缩，保持原数值路径。CPU blocking 实际消费 staticParallelExtent。直接 16×16 会形成 256 个累加器组，4×4 主体使用 16 个累加器组和各 4 个输入组，实际进入 RVV stream-load/step/finalize。
- 外部缺口已在 TianchenRV 通用 physical passes 闭合：`hasCompleteStreamAxis` 从 subview 的静态 extent 或 memory/local view 的常量有效范围证明完整 replica 轴，同时供 `MaterializeRISCVPrograms` 与 `RVVStreamLoadOp` verifier 使用；domain 分块的名义 shape 不作为尾部有效范围，原动态 point 路径保持。`PropagateRISCVLayouts` 修正 update 输入/result 的 local carrier 传播；`PlanRISCVMemory` 消费旧 implementation，保留已选 local-update leaf。Slice 省略的尾部 selector 按 canonical 规则视为 all；Reaxis 保持位置，仅改轴名。独立提交为 `59c1a50ad`、`f3402637d`，无新增 Canonical Level、Intent 专用路径或 emitter 算法。
- 共享改动的普通 Mojo 条目已通过同次原容差：row affine 1.444316/2.082646 ms、softmax 8.081696/7.888340 ms、LayerNorm 3.797713/3.678546 ms。新 forwarding 限于 private owner；向 caller 输出扩张后出现的 softmax/LayerNorm 慢项已收回，私有状态复制收益保持。保持 private owner 有利于保留后端可见的存储关系，但尚无完整 LLVM 证据将耗时变化唯一归因于 alias analysis。
- RMSNorm 最新实测 5.019436/3.447934 ms、G/S 1.455781，原容差通过，真实慢项保留在 CSV。原生产生成入口得到的新旧 Mojo 源码无差异，不能据此认定本轮 current-program 改写造成该差距；其运行、调优或外部编译层原因尚未定位，不宣称全部 CPU 性能已经收敛，也不继续重复采样掩盖它。
- 最终独立只读代码复核 `/root/cpu_structural_final_review` 已通过，覆盖 CPU proof/transform、Weft implementation/legalization 与授权外部 lowering；复核发现的隐式 trailing selector 问题已修正。Runtime 仅补跑最终改动影响的两条 Mojo 区域程序和 Weft causal attention，均通过；新的独立 Verifier `/root/cpu_structural_comet_verifier` 给出 A1–A4 全部通过的判断。没有运行额外测试，正式验收状态及用户接受结果以 Runtime 管理的产物为准。
