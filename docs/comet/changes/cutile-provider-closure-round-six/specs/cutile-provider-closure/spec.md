# cuTile Provider 闭合

## 目标状态

Intent 的 shared GPU Program 是 cuTile lowering 的唯一完整 executable authority。cuTile provider 从 current typed program确定 block identity、tile/index form、access spelling、native structured operation、physical/provider parameters与 launch artifact；provider rewrite 后的 current program在 serialization 前已闭合，terminal serializer不回读 KIR、kernel identity 或逻辑 shape重建执行决策。当前 registry 中 cuTile 能保持同一 Intent 语义的程序在 RTX 5090D 与 H100 上完成数值执行，稳定可比的 generated/source ratio 不超过 1.05。

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
- Fixed profile或有限 candidate集合本身合法。只有 current运行证明某个 domain真实阻塞 lowering、产生不公平 candidate contract或造成稳定性能差距时，才调整 typed parameter/form；adapter不得按 entry名称偷偷筛选。
- Shared/provider profile 数据沿用当前有限 JSON 输入与 typed family/role 绑定。旧 stash 中的调优分类和取消 metadata specialization 不构成已证明决定；metadata 的 specialization、runtime scalar dtype 和索引宽度必须各自保持明确的 ABI 契约。
- Artifact携带已选择的 compile device与 launch-visible grid/config binding。Runtime在同一设备上物化、JIT和 launch，并对跨设备输入执行既定 typed检查；不得在 wrapper中重新选择 capability或默认到另一设备。
- Serializer只输出 import、kernel/signature、current ops、已声明 candidate/config和 launch wrapper。它不得新增 legality、fallback、grid/access、workspace、candidate或 algorithm decisions。

## Terminal correctness 与性能

- Production registry workflow分别记录 generated与 source的 compile、JIT、launch、numerical和 measurement结果。Terminal failure必须指出发生的一侧和最早可判定阶段；总 worker timeout不能成为无法继续归因的最终分类。
- Worker 在进入耗时阶段前保留当前侧和阶段；父进程处理 timeout、异常退出或缺失最终结果时消费该阶段，不能覆盖成无侧无阶段的错误。JIT 与 launch 若由同一 provider 调用完成，只能报告实际可观察边界，不猜测尚无证据的内部阶段。
- 对具有完整 Intent算法，且同环境 source或实际 cuda-tile能力证明语义可表达的 entry，generated路径必须闭合实现并通过数值比较。Source自身不成立、hardware不支持或外部 compiler在明确成本界限内不能完成时，保留对应的准确状态而不伪造通过。
- 两台机器使用同一 compiler提交和一致的 cuTile运行环境。两张 CSV是该状态的当前观察，不定义 DSL、compiler policy或长期 capability。
- 只有 generated/source都数值通过，并且算法、数值契约、dtype、ABI、shape、调用次数、candidate contract和计时范围一致的稳定 entry才计算 ratio。
- 数值容差通过不证明比较契约相同。比较显式覆盖 full-f32/TF32 等输入精度、cast/中间 dtype、所有 kernel 和额外 tensor 变换，以及实际启用的候选搜索；不一致时先对齐同一既定语义或记录具体不可比原因，不能据此改变 shared semantics 或归因性能。
- Source runtime 可接入比较所需的完整候选集合，并由同一 provider tuner 独立选择 winner；对齐只改变候选入口，不改变 source 算法、数值语义、ABI 或计时范围，不按 generated winner、entry 名称或旧 timing 隐式筛选候选。
- 稳定可比 entry的 `generated_p50_ms / source_p50_ms` 不超过 `1.05`。明显高 ratio优先从 current Physical Program的 mapping/blocking/ownership/traversal/materialization、typed config/candidate或 provider form修复；不可通过 entry-local特例、source模仿、测量挑选或语义缩窄闭合。
- 若 fresh same-code复核证明结果不稳定或不可比，CSV记录具体原因并不保留无解释的高 ratio `pass`；`1.05` 不成为 lowering legality或 target capability规则。

对 shared 修复，语义完整节点需给出真实 IR 改写与保持条件，并在受影响的 Triton/cuTile production 路径 emit、JIT、launch 和对数值；涉及 TileLang storage/copy 边界时覆盖相应路径。只采用手动可执行的生产 repro，不增加 pytest、fixture、测试目录或长期检查体系。先完成数值、比较契约和终端归因，再用同一 current compiler state 收敛双机性能；四项最终验收及 `1.05` 门槛不因工作顺序细化而降低。

## Acceptance scenarios

Scenario: Current typed access 决定 native 或 checked cuTile form

Given 一个 current shared GPU Program含完整覆盖、边界覆盖或间接索引的 access，并显式携带 coordinates、source axes、validity、fill/effect和 fragment relation

When cuTile provider完成 legalization、verification、serialization与实际执行

Then provider只从这些 current facts选择语义等价的 native tile或 checked gather/scatter，invalid read/write语义在运行中保持并通过数值比较，Serializer不重建 access legality

Scenario: cuTile 可表达的 structured program 到达数值终端

Given 一个 registry structured/control form可由 cuda-tile 1.5.0或同环境同语义 source表达

When current shared program经过 cuTile local rewrite、closed-surface verification、emit、JIT和 launch

Then axes、identity/combine、format、accumulator、validity、atomic与 result flow均保持并通过数值比较；若实际 surface不能保持某个组合，最早的 provider/capability层给出精确 rejection，而 source成功的同语义 form不被误报为 target unsupported

Scenario: 双机完整 registry 获得可归因的 current 结果

Given 同一 current compiler state、RTX 5090D和H100上一致的 cuTile环境以及当前 registry登记的 generated/source callable closure

When 两台机器分别执行完整 production workflow

Then 每个 entry都获得可归因到 generated/source和 compile/JIT/launch/numerical/measurement阶段的终端状态，不留下 opaque timeout或宽泛 compile failure，两张 cuTile CSV完整记录同一提交的 current结果

Scenario: 稳定可比 entry 达到 1.05 性能目标

Given 双机完整结果中 generated/source均数值通过且算法、数值、ABI、shape、candidate contract和计时范围可比的稳定 entry

When production workflow记录 fresh p50时间，并对明显异常进行一次 same-code复核

Then 每项 `generated_p50_ms / source_p50_ms` 不超过 `1.05`；真正的差距由 current Physical Program、typed config/candidate或 provider form闭合，不稳定或不可比项记录具体原因而不是无解释的高 ratio `pass`
