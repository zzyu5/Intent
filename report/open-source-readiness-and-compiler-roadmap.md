# IntentDSL 产品形态、Pass 资产与成熟化路线图

更新：2026-10-07。依据当前 main、正式规格、本地 Triton/TileLang 源码、既定产品 program 及已有运行产物自查后原位更新。已交付的入口和基础拆分不再列为新任务；本次核对旧记录不等于重新执行所有后端。本文安排实现，不替代 doc/ 的语义规格。

## 1. 当前判断

**继续使用现有编译器骨架，下一阶段集中完善物理程序质量和优化知识的消费者。** 当前不是只有统一前端：GPU、CPU、DSA 各有真实物理 IR、分析、变换、目标实现和运行入口。已有共享知识被多个消费者使用，没有证据要求重新建立 IR、JIT、调优器或优化编排框架。

主要剩余问题是：合法结构未必得到合适的遍历和物化；新类型表达式未必被所有相关消费者理解；降低 nominal payload 未必减少实际 native 工作；一般程序组合仍有覆盖边界。这些问题由对应分析、pass 与目标消费解决，不能用统一入口或目录整理代替。

产品目标保持不变：作者用 DSL 表达算法、数值、状态和显式多 kernel 编排；compiler 形成可执行且性能良好的 family 程序；pass 是可发现、可组合、可扩展的核心资产。GPU/CPU/DSA 复用语义和证明方法，各自形成适合执行模型的结构。Triton/cuTile 等下层已有的布局、机器流水线和指令分配直接使用。

v1 以明确维护的 provider、硬件能力与工具链范围为边界。主产品不依赖 agent 服务，不用 agent 修复掩盖确定性 lowering 缺失；实验型落地 agent 暂不进入本路线。

## 2. 自查：已经交付的基础

| 已交付基础 | 当前实现与边界 |
|---|---|
| 标准 pass 注册、阶段合同、前后验证、独立 IR 工具 | [GPU Passes.td](../include/Intent/Dialect/GPU/Transforms/Passes.td):7、[CPU PassSupport](../lib/Dialect/CPU/Transforms/PassSupport.h):19、[Registration](../lib/Compiler/Registration.cpp):63。继续使用 intent-opt、optimize_ir、generate_from_ir，不再造调度框架；必要 lowering 与可选优化按合同区分。 |
| GPU 清理与稳定读取移动分责 | [EliminateCommonValues](../lib/Dialect/GPU/Transforms/Value/EliminateCommonValues.cpp):220、[Passes.td](../include/Intent/Dialect/GPU/Transforms/Passes.td):238 已分离必要 cleanup 与可选 hoist。旧“CommonValues 尚需拆分”撤掉。 |
| DSA/BANG C 局部组合责任 | [Legalize](../lib/Target/BangC/Transforms/Legalize.cpp):89、[Supply](../lib/Target/BangC/Transforms/Supply.cpp):578 已有 LocalComposition、StorageReuse、SupplySynchronization、StorageBinding，局部 fixed point 由组内拥有。旧“先拆大组合函数”撤掉。 |
| 稳定版本与供数知识已有消费者 | GPU [AccessLoads](../lib/Dialect/GPU/Transforms/Access/AccessLoads.cpp):706；CPU [ProducerVersions](../include/Intent/Dialect/CPU/Transforms/Structure/ProducerVersions.h):23；BANG C [CompletedSupply](../lib/Target/BangC/Transforms/CompletedSupply.cpp):92。后续扩大适用结构和盈利，不再把抽一个公共 helper 当目标。 |
| 30 个完整产品 program 与 host 编排 | [examples/README](../examples/README.md):7、[common.py](../examples/programs/common.py):85、[use.py](../examples/use.py):109 已支持 source/native/run、资产恢复、prepared 完整调用及多 kernel。算法唯一保存在 examples/kernels。旧 M3 主体已交付。 |
| 完整经验配置和真实调优 | [ConfigurationProfiles](../lib/Dialect/GPU/Transforms/Configuration/ConfigurationProfiles.cpp):135 沿完整行投影；Triton 使用原生 Config/autotune，cuTile 使用现有运行路径。BANG C [program.py](../python/intent/runtime/bangc/program.py):113 已有完整候选、隔离 trial/reset 与 CNRT 选择；旧“固定 entry、没有 autotuner”撤掉。 |
| GPU 独立遍历证明及容量消费 | [IterationDependencies](../include/Intent/Dialect/GPU/Analysis/IterationDependencies.h):16、[Traversal](../lib/Dialect/GPU/Transforms/Control/Traversal.cpp):135、[AccessForms](../lib/Target/Triton/Transforms/Access/AccessForms.cpp):383 已贯通依赖证明、ownership 容量和 TMA 对齐消费。证明来自当前 IR；不修改作者算法或扩候选笛卡尔积。 |
| 完成值快照与循环存储复用 | [FragmentSnapshot](../include/Intent/Dialect/GPU/Transforms/Storage/FragmentSnapshot.h) 已由 Triton/cuTile 共同消费；[GatherStorage](../lib/Target/Triton/Transforms/Access/GatherStorage.cpp) 按不变 SSA、变化索引、实际物理容量及存活关系选择。[AccessGathers](../lib/Dialect/GPU/Transforms/Access/AccessGathers.cpp):218 已闭合标量 range gather。重复机械路径已删除，W1 不再是下一轮待办。 |

现有 canonical 产品记录中，Triton、H100 cuTile、Mojo、Weft 各有 30 个 executed program、35 个 artifacts；BANG C 有 30 个 generated program、35 个 artifacts。这里只核对已有记录，不宣称本次重跑或所有输入覆盖。BANG C 源码成功不能替代 CNCC、CNRT、设备数值与性能，旧“MLU 13 项、Weft 六项”不再代表当前产品覆盖。

论文 experiments 保留历史材料，产品推进不修改、回写或扩建它。后续唯一算法与 host 入口是既定 examples/programs 和 examples/use.py；产物、缓存和运行观察在仓库外，当前结果原位更新。

## 3. 自查：仍存在的实质问题

| 问题 | 当前事实 | 实现方向 |
|---|---|---|
| 普通 scan 消费循环尚不能整体流式分块 | [RealizeScanConsumers](../lib/Dialect/GPU/Transforms/Reduction/RealizeScanConsumers.cpp):108 只接已知独立循环；:146 的重放和 :438 的 consumer 绑定仍有专门前提。 | 独立性、读取版本/别名、source slice 消费一起闭合。count-prefix 唯一写入还需活跃成员、单调性与无溢出证明；不能只删门禁或把单次 scalar unique store 当跨迭代证明。 |
| 已减少物化，但逐元素消费仍串行 | W1 消除了 Nonzero 每轮完整 shared 写，仍保留作者的逐元素消费；这不是整个 compaction 性能已完成。 | W2 从依赖、活跃写入唯一性、存储版本和 source slice 一体证明组织共同遍历；保持算法及原数值合同。 |
| 资源估算与真实收益有差距 | [Resources](../lib/Dialect/GPU/Analysis/Resources.cpp):297 明确 nominal footprint 不等于 native allocation。容量约束降低 Mamba 的寄存器/shared，但完整调用尚未提速。 | 从重复工作、consumer 使用、存活区间和并行度选择结构；native layout/分配交下层，已有经验行负责调优。不用更多配置掩盖结构缺失。 |
| 全后端性能与交付未收口 | GPU 有贯通收益，也仍有长尾；CPU/Weft 的运行覆盖不等于各类 consumer 已高效；BANG C 当前覆盖不能从 source 推成设备成功。 | GPU 主线完成当前知识包；CPU/DSA 分别消费同一规律并推进真实 native、完整调用。旧失败先核当前代码与工具链，不用旧结论阻止实现。 |

共享 carrier 和明确阶段已经成立。新表示暴露的消费者缺口在所属层修通，不意味着每次性能优化都要修改 frontend、runtime 或公共 ABI。新增优化主要修改分析、pass、目标消费及必要接线；需要更大改动时给出当前 IR 缺少的具体事实。

## 4. 接下来按知识包推进

### W1：完成值、存储版本与循环消费者——已完成

结果是“原值只准备必要次数，消费者直接使用正确版本”，涵盖共享存储构造、target 采用规则、访问组合与 native 消费，不以单个 Nonzero 特判交付。

1. 两 provider 已共用不可变 fragment 快照构造与读取。保存完整物理 lanes，包括已有 fill；位置域不冒充新 logical domain，不重新执行 producer。
2. Triton 已消费循环不变源和变化标量索引。有限容量、allocation context、owner、dominance 及原 validity/fill 闭合；混合的大小域使用实际 constexpr 物理容量条件。小域保留 native gather，但当前 private workspace ABI 仍可能保留，不能宣称冷准备零新增资源。
3. 可选采用要求未替换消费者不再跨循环保留源表示。穿透值包装，region/record 转发未知时保原。这样解决了 Q4 初版双份表示的回退；不能用少几条 shared 指令替代存活与收益判断。
4. 标量 range gather 已沿原公式闭合。原 oversized 合法化与 cuTile 存储共用机械实现，被替代的重复构造已删除；未新增配置行或改变作者程序。
5. 两后端原 30 program 全部生成。实际影响 Triton 两项、cuTile 一项，均完成 native/运行及原数值规则检查。原 Q4 输入为零权重，只说明该原生产输入的正确性范围，不冒充未运行的其它数据。

真实效果：Nonzero 相同配置的 native 每轮完整 shared 写从 4 条向量 store 降为 0，shared 从 16 KiB 降到 256 bytes；新增一次 1 MiB 私有快照，原 prefix workspace 保留。20 次完整调用中位数从 479.568 降到 240.672 微秒。Q4 最终退出新增快照，Triton 21.808 对 22.176 微秒、cuTile 96.480 对 93.888 微秒，不宣称显著收益。初版 Q4 回退保留在原始日志，未以择优重测抹掉。

这一步不擅自并行化扫描后的有序 scatter。快照能减少重复物化，但保留的逐元素串行消费仍属于后续结构问题。

### W2：producer→collective→consumer 的共同遍历——下一里程碑

结果是同一组规范化、版本和依赖证明被多种 collective consumer 使用，不为每个统计公式增加模板。

- 普通 scan 消费先扩独立性证明，再接正确 source epoch/replay 和 member slice，三者由同一完整变换关闭。整数 count-prefix 可利用 typed 0/1 输入及活跃写入证明；一般浮点 scan 不借用该单调性。
- 多 consumer 归约区分完整输入、紧凑 summary 与输出遍历。按依赖和 lifetime 移动准备、缩小 carry，保留 combine、dtype、NaN/tie 与 effect 合同。
- 已有 native reduce/scan 直接映射；不重建下层归约树、线程通信或 layout。
- 关闭可选决策仍产出合法同语义程序；关系闭合、bufferization 与 legalization 不作为随意可拔的性能开关。

### W3：contraction 的供数、初始化和输出流

结果是 typed contraction 及相邻 producer/consumer 能形成有质量的物理程序，依靠共同轴、版本和存储规律，而非算子名称。

- 复用现有 free/reduction blocking、容量绑定和完整经验行；start、extent、grid、fragment、provider form 消费同一事实。
- 多 contraction/multi-consumer 明确哪份准备可以共享、在哪个版本失效、何时再次 materialize；不把全部值延长到整个循环。
- Recurrence 区分 accumulator 与下一次 contraction input；保持作者顺序，避免供数生命周期无声变成昂贵 staging。
- 原生 dot/scaled-dot、descriptor、MMA 和 pipeline 各守职责。实际缺少执行事实才扩 IR，不把 API 拼写变成 shared policy。

### W4：将知识规律扩到 CPU 与 DSA

结果是“相同知识、适合各执行模型的改写”真实成立，不强行复用 GPU physical topology。

- CPU 使用已有 ProducerVersions、BufferStorageAnalysis、task/cohort、向量 carry 与目标 implementation：稳定输入只准备一次，规范化后的计算进入同一消费者，准备成本和完整工作集一起判断。
- DSA 使用 completed version、transfer lifetime、NRAM/WRAM 和首次写入：复用准备具有真实 completion 与 final binding，不能由 leaf 猜测状态。
- 使用各自既有完整 config、provider/runtime 和原产品入口。必要时修 Weft 主线；合法 DSL 的 lowering 缺失仍是 compiler 问题。
- BANG C 先核当下工具链/设备，完成原 30 program 中适用组合的 native 和数值；不以 source 或历史数量收口。性能推进独立组织，不把 GPU 里程碑悄悄扩成全后端重测。

### W5：收束可维护的 v1 产品

- 既有编译、IR 优化、续生成、MCP 和安装入口保持单一；从新增组件确认陌生开发者能找到阶段、合同、公共事实和消费位置。
- 新算法进入同一 DSL/compiler；新 pass 通过标准 pipeline 组合；必要组与可选组的边界准确，mutation 后重建所需证明。
- 工具链/API/ABI 变化集中在所属 legalization、adapter 和 binding，已有 family 知识继续成立；实际改变语义或 capability 的变更明确处理。
- 各维护后端的支持范围由生成、native、运行、数值和完整调用分别说明。冷编译、安装、诊断与分发只处理真实阻碍，不重建已存在的框架。

v1 由可用入口、可靠语义、可扩展优化资产和适用后端的真实程序质量共同收束；不把记录数、代码量或某个快例子代替整个产品，也不把所有未来 SDK/硬件的无限适配当作本阶段门槛。

## 5. 本地成熟实现对照

| 语义/职责 | 本地参照 | Intent 的对应事实与后果 |
|---|---|---|
| 普通循环与 carry | [Triton code_generator](../../ref/triton/python/triton/compiler/code_generator.py):1290–1370 按原 bounds/step 建 SCF For/yield；[TileLang loop_vectorize](../../ref/tilelang/src/transform/loop_vectorize.cc):1085–1169 判断当前窗口、extent/base 与 stride | [IterationDependencies](../lib/Dialect/GPU/Analysis/IterationDependencies.cpp):177 先证明独立效果；不宣称 Triton 自动并行化作者 scalar loop，不套用 TileLang 已标 vectorized 的 surface 合同。 |
| 不变读取与零次循环 | [Triton LICM](../../ref/triton/lib/Dialect/Triton/Transforms/LoopInvariantCodeMotion.cpp):22–61 检查 effects 和执行保护 | Replay 保读取版本与 effect；完成 SSA 的快照复用不等于重新 global load，不能混为一种 hoist。 |
| 动态 gather 内部成本 | [GatherOpToLLVM](../../ref/triton/lib/Conversion/TritonGPUToLLVM/GatherOpToLLVM.cpp):35–47 可用 warp-local；:66–107 否则 storeShared 全源并同步 | [GatherStorage](../lib/Target/Triton/Transforms/Access/GatherStorage.cpp) 已利用循环复用、实际容量和源存活选择；不把规模门槛当 warp-local 不可能的证明，不复制 layout、不一概改成 global。 |
| 阶段与规范化 | [Triton NVIDIA compiler](../../ref/triton/third_party/nvidia/backend/compiler.py):310 起；[TileLang pipeline](../../ref/tilelang/tilelang/cuda/pipeline.py):89 起分开 pipeline、layout 和 primitive consumption | 保标准 manager 与 group closure；有顺序不等于模块化失败，新事实跨阶段消费不全才是待修问题。 |
| 矩阵 native 合法性 | [Triton min_dot_size](../../ref/triton/third_party/nvidia/backend/compiler.py):19–35；[TileLang gemm_op](../../ref/tilelang/tilelang/language/gemm_op.py):62–86、113–132 | 依据 typed axis/capacity 形成完整输入，下层处理机器 padding；不把 M16/N8 硬编码进 shared ownership。 |

这些是本地参考快照的职责，不替代已安装 SDK 的行为。网络用于需要时核对公开 API/version；main GPU 产品只维护 Triton/cuTile，不恢复 TileLang 后端。

## 6. 当前推进顺序

W1 已完成共同构造、重复标量消费、坐标组合及盈利回退收口。下一轮推进 W2：整体解决仍保留的串行 prefix consumer，独立性、source epoch/replay、活跃写入与 source slice 共同关闭，而非继续只去一条 barrier。之后按 W3、W4 扩供数质量和各执行模型的实际消费；W5 收束产品维护与交付。

每轮从明确的物理成本和可复用规律出发。运行确认语义和实际成本，不用反复换配置、扩矩阵、追测量最小值代替 pass。相同配置 native 未变时，完整调用差距保留为观察；候选资源超限、程序失败、正确性失败与持平分开说明。论文 reference 可以校准数量级，不强制每个产品 program 建新外部对手。
