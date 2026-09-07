# cuTile Provider 闭合

本文件属于已停止、被替代的 change，仅作为历史提案保留，未作为已验收 capability 安装。归档结论见 [brief](../../brief.md)；后续工作不继承本文件的旧性能门槛或验收要求。

## 目标状态

Intent 的 shared GPU Program 是 cuTile lowering 的唯一完整 executable authority。cuTile provider 从 current typed program确定 block identity、tile/index form、access spelling、native structured operation、physical/provider parameters与 launch artifact；provider rewrite 后的 current program在 serialization 前已闭合，terminal serializer不回读 KIR、kernel identity 或逻辑 shape重建执行决策。持续修复当前 registry 可达的 lowering 与运行缺口，同算法的 generated/source 开展性能比较，已完成的双机结果增量发布；ratio 不超过 1.05 是改进目标，不是要求全表同时达标或全量重跑才继续推进的门槛。

## Authority 与 provider 边界

- Program mapping、ownership、physical loops、fragment/value relations、coordinates、source axes、validity/fill、effects、buffer obligations与 structured-operation schema来自 current shared GPU Program。
- cuTile-local structure只包含 provider surface真正要求的 block-rank投影、tile-space index、checked bounds/padding、advanced indexing、native operation form、provider option和 launch/runtime binding。
- 若 local form改变 operands、results、types、control或 candidate legality，它必须在 serialization 前成为 current cuTile program的一部分并由 provider verifier检查。纯 API 名称和参数顺序差异由 serializer机械拼写。
- Provider不得从 immutable KIR、origin、kernel/entry/source名称、结果 shape 或相邻 op猜测 shared mapping、access relation、validity、structured schema或 candidate role。
- Fresh cuTile lowering若证明 shared program缺少可执行语义或存在 correctness反例，缺口必须在唯一 shared authority中闭合；cuTile leaf不得建立旁路事实或第二套 program。
- 既有 Triton 成功或 foundations 验收只证明其实际覆盖的程序，不代替当前 cuTile 输入的完整性判断。每项修复先确定 current shared facts 是否足够：足够则在 provider realization 内闭合，不足则在共享事实所属层修复并保持其他 provider 的语义。
- Provider pass 可以用公共 GPU/scf operations 形成等价 local expansion；op 命名空间和修改文件数量不定义职责归属。共同 program/fragment 与 external typed capability 保持原有语义，cuTile 专有 tile-index、底层 layout/pipeline 与 tuning winner 不成为新的 shared policy。
- 随实际可达 lowering 问题允许有界重构：让 carrier、analysis、mutation 与 verification 归属清楚，删除已被取代的重复事实推导和旧路径。每项改动对照真实 Triton/TileLang compiler 的同类实现及 cuTile surface；不以成熟度名义复制 TTGIR、资源 allocator 或整套机器 lowering，不做无反例的大范围重建。

## Access 与 resource form

- 每个 access必须在进入 cuTile 时已经携带 resource、typed coordinates、source-axis mapping、active validity、load fill或 write effect以及 current fragment relation。
- 只有 current facts证明 tile-space native load/store能够完整表示相同访问、边界和 fill/effect时，provider才可选择 native tile form。边界、间接或其它 advanced access使用语义等价的 checked gather/scatter form；无法保持语义时在 provider legality层明确拒绝。
- Invalid read不访问 source并返回声明的 fill；invalid write不产生 effect。Bounds checking不能覆盖或丢失来自算法的 custom validity，两者必须按同一 access relation组合。
- Native/checked form的选择只能依赖 typed relation与实际 cuda-tile capability，不得由 serializer、entry-specific adapter或 source template临场决定。

## Structured compute、control 与 atomic

- Reduce、scan、ordinary/scaled contraction、atomic及 current registry中其它 structured/control operations保留 shared program给出的 axes、identity/combine、direction/inclusivity、format/group relation、accumulator、validity、memory order和 result flow。
- 若 cuda-tile具有语义等价的 native form，provider legalization形成完整 local op或薄映射；若可由同一 provider程序中的合法组合精确展开，可以形成 provider-local expansion；两者都必须在 serialization前验证。
- Provider API缺少某种 dtype、NaN、atomic kind/order、layout或 control组合的等价表达时，在最早拥有充分 capability信息的层给出精确 diagnostic。不得把 propagating maximum替换为忽略 NaN 的 operation，也不得生成数量级更慢的串行程序冒充支持。
- 同环境、同语义 cuTile source能够执行时，它是 provider capability存在的直接证据；generated失败应归为 current shared/provider/runtime实现缺口，除非进一步证明 source 与 Intent 的算法或数值契约不相同。

## Provider program、parameters 与 runtime

- Provider legalization完成后，local MLIR schema和 whole-program provider verifier共同保证 grid/block identity、fragment/static requirements、local op operands/results/regions、access/resource legality、structured form和 parameter binding完整。
- Physical/provider parameter declaration是 typed finite domain；每个 candidate concrete binding都产生完整合法的 current provider program。Intent可以删除能够证明非法的 candidate，但不复制 cuda-tile的下层 layout、register或 resource allocator。
- Fixed profile或有限 candidate集合本身合法。只有实际程序或运行证明某个 domain 阻塞 lowering 或造成性能问题时，才调整 typed parameter/form；双方可以独立选择配置，不要求搜索空间完全一致或先穷举全部候选，adapter 不得按 entry 名称偷偷筛选。
- Shared/provider profile 数据沿用当前有限 JSON 输入与 typed family/role 绑定。旧 stash 中的调优分类和取消 metadata specialization 不构成已证明决定；metadata 的 specialization、runtime scalar dtype 和索引宽度必须各自保持明确的 ABI 契约。
- Artifact携带已选择的 compile device与 launch-visible grid/config binding。Runtime在同一设备上物化、JIT和 launch，并对跨设备输入执行既定 typed检查；不得在 wrapper中重新选择 capability或默认到另一设备。
- Serializer只输出 import、kernel/signature、current ops、已声明 candidate/config和 launch wrapper。它不得新增 legality、fallback、grid/access、workspace、candidate或 algorithm decisions。

## Terminal correctness 与性能

- Production registry workflow分别记录 generated与 source的 compile、JIT、launch、numerical和 measurement结果。Terminal failure必须指出发生的一侧和最早可判定阶段；总 worker timeout不能成为无法继续归因的最终分类。
- Worker 在进入耗时阶段前保留当前侧和阶段；父进程处理 timeout、异常退出或缺失最终结果时消费该阶段，不能覆盖成无侧无阶段的错误。JIT 与 launch 若由同一 provider 调用完成，只能报告实际可观察边界，不猜测尚无证据的内部阶段。
- 对具有完整 Intent算法，且同环境 source或实际 cuda-tile能力证明语义可表达的 entry，generated路径必须闭合实现并通过数值比较。Source自身不成立、hardware不支持或外部 compiler在明确成本界限内不能完成时，保留对应的准确状态而不伪造通过。
- RTX 5090D 与 H100 每完成一项运行就更新对应 CSV，不等待同一提交下的双机全量结果。未重跑项保留此前记录并注明，已完成、未计时、真实失败与未重跑不得混淆；CSV 不定义 DSL、compiler policy 或长期 capability。
- 性能比较以算法相同为前提，保持输入 shape、外部 dtype 等条件一致；舍入位置、FTZ、近似数学和中间精度的细微差异注明即可，不要求逐操作数值契约相同，不以这些差异统一跳过计时或 ratio。
- 完整 callable closure 中的辅助输出、布局转换、workspace 与调用范围差异如实注明，区分 kernel 与端到端时间。比较口径不授权 compiler 改变 Intent 语义；真实 NaN、错误算法或错误结果不作为精度细节放过。已测量的 p50 才能用于成对 ratio，不用调优候选时间替代。
- 若确实算法不同，保留已被 Triton 使用或此前已与 Triton 对齐的作者算法；没有 Triton 使用的 cuTile 专用作者算法可向 cuTile baseline 对齐，不改变语言语义或共享 helper。算法是否相同依据双方实际计算步骤，不从旧 contract gap 状态推断。
- Source runtime 可按实际需要接入候选入口并独立调优，不要求先对齐完整搜索集合，不按 generated winner、entry 名称或旧 timing 隐式筛选候选。
- `generated_p50_ms / source_p50_ms <= 1.05` 是性能改进目标。明显高 ratio 优先从 current Physical Program 的 mapping/blocking/ownership/traversal/materialization、typed config/candidate 或 provider form 调查并修复；不可通过 entry-local 特例、测量挑选、语义缩窄或“契约不同”标签隐去差距。
- CSV 保留真实高 ratio 及其已知原因，未定位时如实标明，不伪造达标。`1.05` 不成为 lowering legality、target capability 或全表同时通过才继续推进的规则；异常仅按需做一次 same-code 复核。

对 shared 修复，说明真实 IR 改写与保持条件，对照 ref/triton 或 ref/tilelang 的同类实现，并选择一条受影响的生产 repro 完成 emit、JIT、launch 和对数值；跨 provider 边界按实际问题定位，不自动扩为全量验证。只采用手动可执行的生产 repro，不增加 pytest、fixture、测试目录或长期检查体系。推进以实现为主，性能结果随实际运行增量发布，不维护 tmp 证据或要求完整调优矩阵。

## Acceptance scenarios

Scenario: Current typed access 决定 native 或 checked cuTile form

Given 一个 current shared GPU Program含完整覆盖、边界覆盖或间接索引的 access，并显式携带 coordinates、source axes、validity、fill/effect和 fragment relation

When cuTile provider完成 legalization、verification、serialization与实际执行

Then provider只从这些 current facts选择语义等价的 native tile或 checked gather/scatter，invalid read/write语义在运行中保持并通过数值比较，Serializer不重建 access legality

Scenario: cuTile 可表达的 structured program 到达数值终端

Given 一个 registry structured/control form可由 cuda-tile 1.5.0或同环境同语义 source表达

When current shared program经过 cuTile local rewrite、closed-surface verification、emit、JIT和 launch

Then axes、identity/combine、format、accumulator、validity、atomic与 result flow均保持并通过数值比较；若实际 surface不能保持某个组合，最早的 provider/capability层给出精确 rejection，而 source成功的同语义 form不被误报为 target unsupported

Scenario: 已完成的运行结果及时进入项目表格

Given RTX 5090D 或 H100 上一个已完成的 generated/source 生产运行

When 发布该次运行结果

Then 对应 CSV 及时记录已获得的时间、ratio 或失败侧与阶段，未计时和未重跑明确标注；不等待双机完整 registry，不把历史记录冒充当前重跑结果

Scenario: 同算法比较性能并暴露真实差距

Given 算法相同且输入 shape 与外部 dtype 等基本条件一致的 generated/source entry

When production workflow 测量双方运行时间

Then 细微数值实现差异不阻断计时，真实 p50 与 generated/source ratio 进入表格并注明额外工作及计时范围；高于 1.05 的差距保留并用于定位改进，不要求全表同时达标才继续推进
