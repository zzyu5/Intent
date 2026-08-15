# 后端 lowering

Backend translator 接收同一 MLIR module 中的 canonical Intent Kernel IR、已经确定的 machine realization 与可选 search space，构造具体目标程序。

完整 semantic path 位于 C++：`intent-compile` 分析 Kernel IR、构造并验证 machine plan，再由共享遍历框架按照 target capability 与 spelling table 逐 op 发射 Triton、cuTile 或 TileLang 源码。中间不物化第三份 target dialect。Python 只负责调用 compiler、加载产物和运行 entry。

## 共享层与 target 叶子

三个 GPU surface 共享：

- Kernel IR 的 operation/region/def-use 遍历；
- ABI、domain、ragged relation、state stream 与 contraction 分析；
- ownership、traversal、tile role、storage、boundary 和合法搜索轴的 machine 决策；
- execution-stage operation slice、stage-axis binding、同步，以及由这些内容派生的 dependency/intermediate index；
- operation handler registry 与 unsupported-op 诊断机制。

每个 target 叶子只提供：

- capability check：这门 surface 能表达 realization 的哪个子集；
- projection：共享物理概念在该 surface 中的 capability 与 spelling；
- emission：canonical operation 加 Physical Plan 到目标语法的逐 op 机械映射；
- compile/run 接线以及对下层 tuner 的显式委托。

接入第四门 GPU tile surface 不应增加 kernel 分析路径，也不应修改共享 emitter traversal；只新增 capability/spelling、leaf handlers 与 runtime adapter。

## 逐轴物理投影

Machine realization 不保存 row/tiled/ragged 之类的 kernel 类别。它逐轴记录可组合的角色，并按用途、层级记录 ownership、lane、ordered traversal、reduction 与 access footprint 等 range。同一逻辑轴可以同时承担 parallel、ordered、reduction、ragged member 或 packed lane 中的合法组合。

| machine concept | Triton | TileLang | cuTile |
|---|---|---|---|
| parallel ownership range | `program_id` 与 grid projection | `T.Kernel` block projection | `ct.bid` 与 grid projection |
| packed scalar lane range | `tl.arange` | `T.Parallel` lane | `ct.arange` |
| ordered traversal range | state-carried loop | `T.Pipelined` / serial loop | state-carried loop |
| access footprint / logical validity | pointer mask 或收紧范围 | guarded copy / range predicate | checked load、gather 或 scatter |
| persistent program choice | grid-stride program loop | persistent block loop | persistent block loop |

一条 realization 可以让不规则 membership、ordered stream、head mapping 与尾块同时作用；target 不得把组合重新压回 kernel 类别字符串。Region argument、row-vector logical extent 与 stream/ragged relation 都由 Plan 显式绑定；target surface 只可读取并拼写，或拒绝自己不能表达的组合，不能另选 tile、ownership、流终点或遍历。

## Operation 对应关系

| Intent / Plan | Triton | TileLang | cuTile |
|---|---|---|---|
| logical view/index | pointer + masked load/store | buffer region + `T.copy` | array/tile load/store |
| `contract` role | `tl.dot` | `T.gemm` | tile MMA/matmul |
| fixed `reduce/scan` role | `tl.max` / `tl.sum` / `tl.cumsum` | fixed target reduction/scan | tile reduction/scan |
| generic `reduce/scan` combiner | typed `@triton.jit` helper | 当前 PrimFunc surface 明确 unsupported | typed function/lambda |
| logical validity | mask / tightened loop | predicate / range | boundary handling |
| Plan storage | compiler-local representation | shared/fragment/local | tile/register storage |
| Plan stream | explicit state-carried loop | pipelined state-carried loop | explicit state-carried loop |

使用 target 的高性能内层 primitive，不等于把数学语义交给 target。Reduce/scan closure、contract reduction axes、operand dtype、accumulator dtype 与 result role先由 Kernel IR/Plan 固定；leaf emitter 只选择对应 spelling并让下层完成 collective tree、layout、指令和 machine code generation。Generic reduce/scan不意味着 arbitrary contract semiring；当前矩阵原语只承接正式声明的 multiply/add 与 dtype capability。

## Execution-stage 投影

Plan 给出 stage operation slice、stage-axis logical binding、tile/worker 与 `same_stream` synchronization。公共 emission index 从 Kernel IR def-use 和 memory effects 派生 dependency、input/output、terminal、intermediate lifetime/visibility；三个 leaf 共用这份索引并按拓扑顺序发射 private kernels，不各自重建。Leaf 不重新划分 stage、不自行决定 workspace owner，也不允许合并 stages。不同 target family 可以在各自 realizer 中产生不同的初始 grouping，surface provider不能改写同一份 machine Plan。

## 失败边界

Translator 开始前验证 Kernel IR、machine plan 与 target projection。以下情况直接以关联的 source location 报错：

- Plan binding 缺失或引用错误 logical node；
- 逐轴 role/range 与 operation binding 的组合不合法；
- target capability 不包含某个 machine concept；
- 某个 canonical operation 没有注册 target handler；
- target spelling 无法保持 dtype、boundary、effect 或 state semantics。

禁止旁路 Python Plan、根据 kernel 名称套模板、整-kernel matcher，以及 emitter 从 tensor shape 重新猜 physical structure。

明确失败必须区分三种性质：

1. **Target capability subset**：目标 API/程序模型没有等价机械投影，在 emission 前以源码位置拒绝。当前例子包括 TileLang 的 CAS、generic reduce/scan closure、部分多轴 checked transfer、single-row contraction 与 runtime-lane FP8 contraction；Triton/cuTile 的 2:4 sparse contraction，以及 cuTile 的 block-scaled matmul。
2. **Lower compiler cost/failure**：目标源码已经合法生成，但下层首次编译超时、候选资源无效或特定设备/toolchain 失败。这是运行记录，不得冒充 Core 或 target-language 不支持；token-sparse attention 的编译超时和某些超长首次 JIT 属于此类。
3. **Intent implementation gap**：Kernel IR 能表达、target 也有能力，但缺少 Plan binding、handler 或机械投影。这一类必须明确报“尚未实现”并继续修，不能登记成 target subset。

同一个 kernel 在一台设备失败而在另一台通过，也不能据此修改 Core；除非能力检查能用设备属性表达，否则它只是具体下层编译结果。CSV 中的 `failed`/`compile_timeout` 是测量状态，不是冻结后的语言能力声明。

## 与下层系统的关系

Intent 决定依赖算法结构信息的部分：合法 ownership、遍历、tile 关系、片上复用边界、logical validity 与数值角色。Triton、TileLang、cuTile 负责其程序模型能自行推断的 layout、寄存器分配、指令选择、低层流水线和候选评测。下层能力增强时，surface leaf 应变薄；共享 Kernel IR 与 machine decision space 不随某门语言版本改变。
