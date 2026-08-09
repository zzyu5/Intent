# 后端 lowering

Backend translator 接收同一 MLIR module 中的 canonical Intent Kernel IR、已经确定的 machine realization 与可选 search space，构造具体目标程序。

完整 semantic path 位于 C++：`intent-compile` 分析 Kernel IR、构造并验证 machine plan，再投影到 Triton、cuTile 或 TileLang target dialect，最后由共享遍历框架逐 op 发射源码。Python 只负责调用 compiler、加载产物和运行 entry。

## 共享层与 target 叶子

三个 GPU surface 共享：

- Kernel IR 的 operation/region/def-use 遍历；
- ABI、domain、ragged relation、state stream 与 contraction 分析；
- ownership、traversal、tile role、storage、boundary 和合法搜索轴的 machine 决策；
- operation handler registry 与 unsupported-op 诊断机制。

每个 target 叶子只提供：

- capability check：这门 surface 能表达 realization 的哪个子集；
- projection：共享物理概念在 target dialect 中的字段与 spelling；
- emission：target dialect operation 到目标语法的机械映射；
- compile/run 接线以及对下层 tuner 的显式委托。

接入第四门 GPU tile surface 不应增加 kernel 分析路径，也不应修改共享 emitter traversal；只新增 target dialect、projection、leaf handlers 与 runtime adapter。

## Program 投影

Machine `program` 的 `ownership` 与 `traversals[]` 分开投影：

| machine concept | Triton | TileLang | cuTile |
|---|---|---|---|
| row ownership | `program_rows` | `block_rows` | `block_rows` |
| tiled ownership | `program_tiles` | `block_tiles` | `block_tiles` |
| ragged ownership | `program_ragged` | `block_ragged` | `block_ragged` |
| persistent traversal | grid-stride program loop | persistent block loop | persistent block loop |
| grouped traversal | grouped program-id mapping | grouped block-id mapping | grouped block-id mapping |
| ordered stream | register-carried loop | fragment-carried `T.Pipelined` loop | register-carried loop |
| staged traversal | target kernels + workspace | target kernels + workspace | target kernels + workspace |

一条 realization 可以同时具有 ragged ownership 与 ordered stream；target 不得把它重新压回一个 kernel 类别字符串。Target surface 只可拒绝自己不能表达的组合，不能另选 tile、ownership 或遍历。

## Operation 对应关系

| Intent / Plan | Triton | TileLang | cuTile |
|---|---|---|---|
| logical view/index | pointer + masked load/store | buffer region + `T.copy` | array/tile load/store |
| `contract` role | `tl.dot` | `T.gemm` | tile MMA/matmul |
| `reduce` role | `tl.max` / `tl.sum` | target reduction | tile reduction |
| logical validity | mask / tightened loop | predicate / range | boundary handling |
| Plan storage | compiler-local representation | shared/fragment/local | tile/register storage |
| Plan stream | explicit state-carried loop | pipelined state-carried loop | explicit state-carried loop |

使用 target 的高性能内层 primitive，不等于把数学语义交给 target。Contract 的 reduction axes、operand dtype、accumulator dtype 与 result role先由 Kernel IR/Plan 固定；leaf emitter 只选择对应 spelling并让下层完成 layout、指令和 machine code generation。

## 失败边界

Translator 开始前验证 Kernel IR、machine plan 与 target projection。以下情况直接以关联的 source location 报错：

- Plan binding 缺失或引用错误 logical node；
- ownership/traversal/stream/ragged/stage 组合不合法；
- target capability 不包含某个 machine concept；
- 某个 canonical operation 没有注册 target handler；
- target spelling 无法保持 dtype、boundary、effect 或 state semantics。

禁止旁路 Python Plan、根据 kernel 名称套模板、整-kernel matcher，以及 emitter 从 tensor shape 重新猜 physical structure。

## 与下层系统的关系

Intent 决定依赖算法结构信息的部分：合法 ownership、遍历、tile 关系、片上复用边界、logical validity 与数值角色。Triton、TileLang、cuTile 负责其程序模型能自行推断的 layout、寄存器分配、指令选择、低层流水线和候选评测。下层能力增强时，surface leaf 应变薄；共享 Kernel IR 与 machine decision space 不随某门语言版本改变。
