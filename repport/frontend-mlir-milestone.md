# Intent DSL 前端与 MLIR 里程碑报告

## 结论

项目已经形成一条独立于 lift/IntentMLIR 的算子编译链：Python eDSL 源码经静态 AST lowering 生成经过语义验证的 Kernel IR，再序列化为注册的 Intent MLIR 方言；首个后端切片能够继续降低到 SCF/MemRef/Arith、LLVM IR 和本机代码，并完成大尺寸输入的数值执行。

本节点所称“完整前端”，指当前设计文档中的语言构造均已有公共 Python 表面、Kernel IR 节点、静态 lowering 路径和统一 verifier 契约，而不是只针对一个示例拼接 MLIR。后端目前只实现了用于打通真实执行链的首个明确子集，不宣称已经覆盖全部算子语义。

## 编译边界

```text
@intent.kernel Python 源码
  -> Python AST 静态分析
  -> Intent Kernel IR（SSA、Region、Effect、Verifier）
  -> Intent MLIR（注册 Type/Op，可由 intent-opt 解析）
  -> Intent-to-SCF 首个 realization
  -> SCF + MemRef + Arith
  -> LLVM dialect / LLVM IR
  -> 本机可执行代码
```

前端不执行被装饰的 Python 函数，也不依赖旧的 lift 结果。DSL 是公开输入；Kernel IR 和 Intent MLIR 是编译器内部语义边界。

## 已建立的前端

### Python eDSL

- 定义层：`@intent.kernel`、`@intent.fn`、`In`、`Out`、`InOut`、`Constexpr`、枚举和 View 约束。
- 值与类型层：bool、index、有符号/无符号整数、浮点与 bfloat、标量、tensor、tuple、record、buffer、domain、region、partition、ragged、stream 和 unit。
- 控制层：静态/运行时 `if`、`for`、`while`、parallel、ordered、state stream、yield、break 和 continue。
- 计算层：一元/二元运算、比较、选择、cast、mask、reshape、transpose、broadcast、reduce、scan 和 contract。
- 数据与副作用层：View load/store、gather/scatter、logical buffer、atomic、fence、random，以及 helper call effect summary。

### Kernel IR

Kernel IR 明确保存 SSA value、block、region、source location、结果类型、索引关系和 effect/resource。统一 verifier 负责检查：

- Kernel ABI 与 View 读写权限；
- operation operand/result/region schema；
- structured control-flow terminator 与 carried state；
- dtype、shape、broadcast、axis 和 combiner 合法性；
- index relation 与访存目标；
- helper 单态化、调用类型与副作用摘要；
- 禁止在前端 IR 中提前引入 worker、layout、pipeline 和 launch 等物理决策。

### Intent MLIR

- 使用本机 MLIR TableGen/CMake 技术栈生成并注册 `intent` dialect。
- 每个 Kernel IR opcode 都有独立的 `intent.*` operation，而不是一个通用字符串 operation。
- logical index、domain、region、partition、ragged、buffer、record、constexpr、enum、stream、unit 和 external view 均有独立注册类型。
- 外部 View 使用 `!intent.view<tensor<...>, "in|out|inout">`，不会与纯 tensor SSA 混为一谈。
- tensor 的符号/动态维度、函数参数约束、operation effect 和源位置作为稳定元数据保留。
- MLIR emitter 在输出前再次调用 Kernel IR verifier；未知 Kernel IR 类型直接失败，不生成兜底类型。

当前公开入口只接受 DSL 生成并通过 verifier 的模块；任意手写 Intent MLIR 不是受支持的公开输入，因此 C++ ODS 不重复实现整套 Python 语义 verifier。

## 首个真实后端切片

`convert-intent-to-scf` 当前明确支持首条可执行路径所需的构造：

- scalar/index constant；
- 一维 domain 与 structured `for`；
- point View load/store；
- 标量浮点加、减、乘、除和整数加、减、乘；
- kernel return；
- external View 到 ranked memref ABI realization。

遇到未实现的 Intent operation 时 pass 会报告错误并失败，不保留 Intent op，也不伪造结果。

## 唯一数值 repro

手动命令：

```bash
bash examples/repro/run_vector_add.sh
```

该命令把构建目录放在 `/tmp/intentdsl-build`，仓库内不生成虚拟环境、build、缓存或日志。repro 定义三个 `1 << 20` 元素的 `f32` View，执行 DSL vector add，经 Intent MLIR、SCF/MemRef/Arith、LLVM IR 和 clang 生成本机程序；程序内部逐元素计算期望值并比较。

本节点实际输出：

```text
backend numerical comparison: PASS (1048576 f32 elements)
```

## 目录职责

```text
python/intent/language/       Python eDSL 公共语言表面
python/intent/frontend/       Python AST 到 Kernel IR
python/intent/ir/             Kernel IR、builder 与 verifier
python/intent/mlir/           Kernel IR 到 Intent MLIR 文本
include/Intent/Dialect/       Intent MLIR TableGen 与公共声明
lib/Dialect/                  Intent dialect 实现
include/Intent/Conversion/    lowering pass 公共入口
lib/Conversion/               Intent realization/lowering
tools/intent-opt/             方言与 pass 驱动器
examples/repro/               唯一手动端到端数值 repro
```

## 对应提交

- `19fabf4`：Python eDSL 语言表面。
- `a99e9c9`、`472214e`：Kernel IR、verifier 与结构化退出契约。
- `1c26ee0`：Python AST 到 Kernel IR 的完整前端 lowering。
- `64c930c`：修正 dtype 公共名遮蔽 Python 内建类型。
- `f6bb7db`：Intent MLIR 方言、emitter 与 Intent-to-SCF pass。
- `dc166b5`：LLVM 本机执行与大尺寸数值 repro。

## 本节点边界

完整前端已经建立；完整后端尚未建立。除上述首个执行子集外，其余 Intent operation 尚无 SCF/GPU realization，性能调优、调度、layout、存储层级、并行映射和目标代码选择也不属于本节点完成范围。当前没有阻塞前端交付的问题。
