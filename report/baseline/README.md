# baseline-v1（frozen）

本目录的两张 CSV 与 `compiler-closure.md` 冻结为“Intent 已经能生成并运行这些 kernel”的历史证据。它们不再承担高性能 baseline 的职责，也不作为 baseline-v2 的 source 覆盖率或性能结论。

- `kernel-performance.csv`：RTX 5090D 的旧统一矩阵。
- `kernel-performance-h100.csv`：H100 的旧统一矩阵。
- 其中的 variant、generated-only、参考实现以及不同计时 scope 均按原样保留；不再补 source、不重排、不删列。
- 后续编译器改动不回写这里。新的高性能比较只写入 `report/baseline-new/` 将生成的六张 provider/device 表。
