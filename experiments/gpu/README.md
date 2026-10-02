# GPU 算子实验

本组包含 Triton、cuTile 的现有生产实验。

- [registry.py](registry.py)：Triton 54 项、cuTile 37 项，保持原生产输入与顺序。
- `providers/`：Intent 与公开 source 的完整 callable 适配。
- `baselines/{triton,cutile,cuda}/`：原 provider source/runtime corpus；各上游子树整体保留。
- `results/*-{5090,h100}.csv`：原机器上的历史结果。新 pass 或单点实验也输出到本目录下的 `results/`。

## 执行

先按 [公共构建说明](../README.md#构建与调用) 构建。以下从仓库根目录执行，`python` 分别来自 Triton、cuTile 对应环境：

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/examples"
python -m experiments.gpu triton \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --kernel fused_softmax --jobs 1 \
  --output experiments/gpu/results/triton-single.csv

python -m experiments.gpu cutile \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --tuning-config experiments/gpu/providers/cutile/tuning.json \
  --kernel relu --jobs 1 \
  --output experiments/gpu/results/cutile-single.csv
```

去掉 `--kernel` 执行该 provider 的现有全量。按实际机器显式指定 `results/triton-h100.csv` 或 `results/triton-5090.csv` 等输出；保持两台机器的结果分开。

也可使用统一构建脚本：

```bash
./experiments/run.sh triton experiments/gpu/results/triton-single.csv fused_softmax
./experiments/run.sh cutile experiments/gpu/results/cutile-single.csv relu
```

## 结果

[triton-5090.csv](results/triton-5090.csv)、[triton-h100.csv](results/triton-h100.csv)、[cutile-5090.csv](results/cutile-5090.csv)、[cutile-h100.csv](results/cutile-h100.csv) 保留原观察值。目录整理后的单点见 `results/integration-triton.csv`、`results/integration-cutile.csv`，只用于确认生成、编译、运行与原数值比较可接通，不代表全量性能验收。

2026-09-22 在 RTX 5090D 上，Triton `fused_softmax` 与 cuTile `relu` 均通过上述单点数值比较。
