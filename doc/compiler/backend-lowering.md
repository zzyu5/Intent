# 后端 lowering

Backend translator 接收同一 MLIR module 中的 canonical Intent Kernel IR、已经确定的 machine realization 与可选 search space，构造具体目标程序。

完整 semantic path 位于 C++：`intent-compile` 分析 Kernel IR、构造并验证 machine plan，再由共享遍历框架按照 target capability 与 spelling table 逐 op 发射 Triton、cuTile 或 TileLang 源码。中间不物化第三份 target dialect。Python 只负责调用 compiler、加载产物和运行 entry。

## 共享层与 target 叶子

三个 GPU surface 共享：

- Kernel IR 的 operation/region/def-use 遍历；
- ABI、domain、ragged relation、state stream 与 contraction 分析；
- ownership、traversal、tile role、storage、boundary 和合法搜索轴的 machine 决策；
- execution-stage dependency、intermediate lifetime/visibility、同步与 fusion policy；
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

Plan 已给出 stage operation slice、dependency、intermediate producer/consumer/owner/lifetime/visibility、synchronization、fusion 与 grouping policy。当前三个 GPU surface 只接受 `same_stream + forbidden + fixed_operation_slice`，按拓扑顺序发射 private kernels；leaf 不从 def-use 重新划分 stage，也不自行决定 workspace owner。不同 target family 可以在各自 realizer 中产生不同 grouping，surface provider不能改写同一份 GPU Plan。

## 失败边界

Translator 开始前验证 Kernel IR、machine plan 与 target projection。以下情况直接以关联的 source location 报错：

- Plan binding 缺失或引用错误 logical node；
- 逐轴 role/range 与 operation binding 的组合不合法；
- target capability 不包含某个 machine concept；
- 某个 canonical operation 没有注册 target handler；
- target spelling 无法保持 dtype、boundary、effect 或 state semantics。

禁止旁路 Python Plan、根据 kernel 名称套模板、整-kernel matcher，以及 emitter 从 tensor shape 重新猜 physical structure。

## 与下层系统的关系

Intent 决定依赖算法结构信息的部分：合法 ownership、遍历、tile 关系、片上复用边界、logical validity 与数值角色。Triton、TileLang、cuTile 负责其程序模型能自行推断的 layout、寄存器分配、指令选择、低层流水线和候选评测。下层能力增强时，surface leaf 应变薄；共享 Kernel IR 与 machine decision space 不随某门语言版本改变。
