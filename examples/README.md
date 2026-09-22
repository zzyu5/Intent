# Intent kernel 示例

`examples/kernels/` 只保存作者编写的目标无关算法，按 activation、contraction、normalization、streaming 等算法职责分类。编译器保持这些程序的类型、shape、数值和 effects；target 在编译调用中选择。

执行、reference baseline、生产 registry、实验结果和 pass 对照在 [experiments/](../experiments/README.md)：

- [GPU：Triton / cuTile，保留 TileLang](../experiments/gpu/README.md)
- [CPU：Mojo / Weft](../experiments/cpu/README.md)
- [MLU：DSA / BANG C](../experiments/mlu/README.md)
- [Agent TritonBench](../experiments/agent_tritonbench/README.md)

新增算法示例放在 `kernels/`，相应运行适配与实验数据放在所属实验组。不要在 examples 下重新建立 runner 或 baseline 副本。
