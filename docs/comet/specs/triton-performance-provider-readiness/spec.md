# Triton 性能与 provider 终端就绪性

## 目标状态

Shared GPU Program 作为唯一完整 executable authority，在 provider legalization 之前已明确 traversal、ownership、coordinates、source-axis relations、validity、effects 与 complete shared configs。Provider 只基于这些 current typed facts 选择 target-local form 并产生完整 runtime contract；terminal emitter 只确定性输出 current provider program 与 artifact。在此边界上，当前 Triton registry 的稳定可比条目在 RTX 5090D 和 H100 上以默认 config 达到 source 的 1.05× 以内，且同一 shared program 可被 cuTile 与 TileLang 扩展而无需重建共同语义。

## Shared 事实与 provider-local form

- Physical accesses 必须显式携带 resource、coordinates、source-axis mapping、validity/fill、effect 与 current fragment relation。Provider 可将它们投影为 pointer、block pointer、descriptor、tile、gather/scatter 或 copy，但不得从 KIR、kernel 名称、结果 shape 或不完整的 provenance 重建 access 语义。
- Full-validity 与 exact-range coverage 是由 current coordinate/range/view 关系决定的共同事实。各 provider 可以选择不同 native form，但不能因重复实现了较弱的事实识别而对同一 access 得到不一致的 legality 结论。Unknown 只能保留通用合法形态或在依赖 exact 事实的 provider transformation 中放弃。
- Grid rank/permutation、Triton pointer spelling、cuTile tile index、TileLang storage/buffer/copy/sync/pipeline 与 target-native structured operation 属于 provider-local representation。它们的选择必须在 current provider IR 中显式化并由 provider verifier 闭合，而不是被 serializer 隐式决定。
- 不要求 shared IR 复制 TTGIR distributed encodings、provider memory hierarchy、instruction selection 或 pipeline schedule。后续 provider 通过 typed target extension 补充这些形态。

## Triton descriptor 与 terminal runtime contract

- Descriptor-capable access 的 provider program 必须同时显式表达 descriptor 建构、pointer 形态、选择条件、每个 descriptor 的实际 shape/strides/block shape/padding/alignment，以及 allocator 的 ABI 和 lifetime。选择条件是 typed launch/runtime contract，不是 terminal serializer 的内建 policy。
- Compile-time 只能在已证明的 constant layout 上声称 contiguous。若 extent 或 stride 由 runtime ABI 绑定，descriptor 路径必须携带能在 launch 前核验的完整 row-major/alignment/shape 约束，并使用真实 runtime strides；无法证明时选择已物化的 pointer 形态。
- Config 集在 serializer 前必须是非空且完整的 shared tuple 与 Triton-local options 组合。Descriptor/pointer 形态可以成为每份 typed config 的一部分，但 pruning 所消费的资格约束必须来自 provider/runtime contract。
- Serializer 可以输出 imports、decorators、config literals、typed launch wrapper、allocator spelling 与 current IR operations；不得新建 descriptor legality、fallback、grid/access relation、candidate 或 workspace/lifetime 决策。

## Triton 性能终端结果

- 性能比较必须使用同一算法与数值契约、dtype、ABI、shape、调用次数和计时范围。两台机器必须使用显式且一致的 Triton 3.6 环境，两张 CSV 必须来自同一 current compiler state。
- 对双方均完成 JIT、launch、数值比较且计时可比的稳定条目，默认 generated config 的 `generated_p50_ms / source_p50_ms` 应不超过 `1.05`。差距优先从 current Physical Program、typed config 或 provider form 改善，不得由 entry-local 特例或 source-template 模仿解决。
- 明显异常的高 ratio 只进行一次 fresh same-code 复测。若 current source 使用不同的 approximate/exact 数值契约，或 source resource/compatibility、worker timeout、adapter/measurement 不能形成可比时间，必须保留准确的非性能可比终端状态，不得写入伪 ratio，也不得保留无解释的高 ratio `pass`。
- `1.05` 性能目标不得反向定义 DSL 语义、lowering legality 或默认 provider policy；fixed table、profile 组合或缺少 cost model 只有在被 current 运行证明真实阻塞 lowering 或性能时才是缺口。

## Acceptance scenarios

Scenario: Descriptor 双形态由 typed provider/runtime contract 决定

Given 一个 descriptor-capable Triton kernel 和 contiguous 或 dynamic padded-stride external view

When provider legalization 闭合 terminal program 并生成 launch artifact

Then artifact 在 serialization 前已显式携带 descriptor 资格、真实 shape/strides、allocator 及 pointer/descriptor branches，runtime 只选择合法形态，两路都能实际 JIT/launch 并保持数值语义，Serializer 不新建 legality 或 fallback

Scenario: 各 provider 消费同一 shared access 事实

Given shared GPU Program 含显式 coordinate/source-axis mapping、exact range-tail validity 和 structured computation

When 它分别经过 Triton、cuTile 与 TileLang provider legalization

Then 三者只基于 current typed facts 选择合法 target-local form，cuTile 的 exact full-coverage access 不会退化为 gather/scatter，TileLang storage/copy/pipeline 仍是 provider extension，每个 terminal program 无需 KIR side record 或 emitter 重建即能验证

Scenario: 双机 Triton 结果闭合稳定可比的 1.05× 性能

Given 当前 Triton registry、同一 current compiler state、两台机器显式的 Triton 3.6 环境与各 entry 登记的 source baseline

When RTX 5090D 和 H100 分别执行完整 generated/source 数值与计时 workflow，并对明显异常项做一次 fresh same-code 复测

Then 所有稳定、语义和计时可比的 entry 通过数值比较且默认 config ratio 不超过 1.05，其余项保留准确的 source/measurement 非可比状态而非无解释的高 ratio `pass`，两张 CSV 完整记录同一 compiler state
