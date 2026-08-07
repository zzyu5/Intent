# 编译器模块架构

目录与模块边界应直接表达下面的数据流：

```text
Python wrapper / Host API
          ↓
Source Frontend
          ↓
Kernel IR
          ↓
Realizer
          ↓
Physical Plan
          ↓
Backend Emitter
          ↓
Compiled Artifact
          ↓
Runtime Dispatch
```

## Host API 与 wrapper boundary

Host 侧接收 Python kernel object、runtime tensor/scalar、用户 specialization、target 与 compile options。Wrapper 负责分配和多-kernel orchestration，不进入 Kernel IR。

## Source Frontend

Frontend 读取受限 Python eDSL，解析：

- `@intent.kernel` 与 `@intent.fn`；
- signature、view kind、symbolic shape、dtype 与 specialization；
- domain/region、tensor expressions、控制流、structured primitives；
- logical buffers 与 effects。

Frontend 的输出是保持 source algorithm 的 Kernel IR。它不选择 tile、worker mapping、storage 或 target primitive。

## Kernel IR

Kernel IR 是 source-visible kernel algorithm 的权威表示。它保存 ABI、logical workset、tensor-flow、state、control、structured nodes、index relation 与 effects。

详见 [Kernel IR](kernel-ir.md)。

## Realizer

Realizer 接收 Kernel IR、target information 与 compile policy，联合选择内部 extent、ownership、storage、layout、target primitive、pipeline、boundary 与 launch。

Realizer 不修改 source algorithm，不执行 graph-level fusion/fission，也不改变 wrapper-visible ABI。

## Physical Plan

Physical Plan 是 realizer 的 target realization 结果，是独立于 source language 的 compiler IR。它可以被验证、比较、搜索与交给 backend lowering。

详见 [Physical Plan](physical-plan.md)。

## Backend Emitter

Emitter 将 `Kernel IR + Physical Plan` 具体化为 Triton、TileLang、cuTile、CPU SIMD 或 RVV program，并生成 target entry 与 launch configuration。

详见 [后端 lowering](backend-lowering.md)。

## Compiled Artifact 与 runtime

Artifact 保存可调用 entry、launch configuration、可读生成源码和后端/低层 IR。Runtime 只负责提交这个 entry；完整图与多-kernel 调度仍在 Python wrapper。

详见 [编译产物与运行边界](compiled-artifact.md)。
