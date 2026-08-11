# 编译产物与运行边界

## Compile result

```python
compiled = intent.compile(
    kernel,
    target=intent.TritonTarget(device=0),
    compiler=intent_compile,
)

print(compiled.source)
print(compiled.ir)
```

编译产物至少包含：

- 一个 callable target kernel entry；
- 包含 Kernel IR 与 Physical Plan 的组合 MLIR；
- 可读、可导出的 target source；
- 第一次真实 launch 后由 Triton JIT 产生的 backend 或 lower-level IR。

Launch policy 由组合 MLIR 中的 Plan 和生成源码共同保存，artifact 不维护第二份 Python launch/Plan 数据模型。

当前 Triton target 的调用形式为：

```python
compiled = intent.compile(
    stable_softmax,
    target=intent.TritonTarget(device=0),
    compiler="/path/to/intent-compile",
)
compiled(input, output)
```

`source` 与包含 Kernel IR/Physical Plan 的 MLIR 在 compile 返回时即可读取。`intent-compile` 在同一进程内完成 realization、组合 MLIR 验证与 target emission，并分别输出组合 MLIR 和目标源码。Triton 的 TTIR、TTGIR、LLVM IR、PTX 等 backend IR 由第一次真实 launch 触发 JIT 后写入同一个 artifact；artifact 不用第二套编译路径伪造这些结果。

## Generated code 是正式输出

生成代码应：

- 保留 source variable 与 region 的可识别命名；
- 明确显示 Intent 已决定的 tile、grid、ownership 与 storage；surface 要求显式拼写的 layout 或 pipeline 参数也保留在源码中，下层自行推断的部分不伪造出来；
- 使用结构化 helper，不生成难以阅读的一次性字符串；
- 可以脱离 Intent 继续编译、调试和人工修改；
- 成为性能理解与极端 kernel 人工接管的直接接口。

Profiling、cost breakdown 或 `plan.explain()` 可以作为 compiler tooling，但不是 source language semantics。

## Runtime invocation

调用 Intent kernel 时，用户不提供 `[grid]`。Runtime 根据 compiled artifact 的 entry 与 launch configuration，在当前 device/stream 提交一次 invocation。

底层 `compiled(input, output)` launch 不负责：

- framework graph partition；
- 输出或 workspace 分配；
- 多-kernel 调用顺序；
- 自动融合或拆分 kernels；
- source variant 的 library policy。

这些仍属于普通 Python wrapper。当前 artifact 额外提供的 `compiled.run(input)` 是一个会执行 `empty_like` 的 convenience wrapper；它用于与同样包含 output allocation 的上游 softmax wrapper 做公平比较，不改变底层 launch 边界。

## Single-kernel invariant

一个 source kernel invocation 对应一个 target entry invocation。Compiler-private scratch、completion counter 或 target 内部多级 physical implementation 可以存在，但不得暴露为额外 runtime dispatch。
