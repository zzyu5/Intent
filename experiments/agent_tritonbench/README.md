# Agent TritonBench 实验

本组保存固定题集、生成指令、手册 MCP 接入、单次提交 runner 和结果。每题每语言组只提交一次完整程序，提交后不向同一 agent 反馈错误或性能进行修复；后续 compiler 开发复测与首次成绩分开。

## 已有结果

- [h100-selected.csv](results/h100-selected.csv)：100 题已选程序的逐题性能对照。
- [h100-selected.json](results/h100-selected.json)：172 条测量、程序来源、容差、计时方式、配置 winner 和统计口径。
- 这批是 2026-09-22 开发复测：Intent 98/100 数值通过；有效 GPU 算子 78/96 不慢于 ref；同精度的 agent Triton 配对中 39/69 不慢于对方。
- 耗时比几何均值为 Intent/ref 0.6612（96 题）、Intent/agent Triton 0.9549（69 题）；不是全新一轮生成成绩，也不意味着逐题性能达标。
- 题集保留 [suite.json](suite.json) 与 [suite-100.json](suite-100.json)。历史分析放在 `reports/`，不当作当前语言规格。

原大批次产物仍由结果 JSON 的 `source_root` 和 `program` 定位；编译/JIT cache、隔离 Codex state 与 provider 凭据留在仓库外。

## 重测已有程序

不重新调用生成模型。下面使用已有的 abs 提交，保持题集原输入与容差：

```bash
export PYTHONPATH="$PWD:$PWD/python:$PWD/examples"
mkdir -p experiments/agent_tritonbench/results/integration
python -m experiments.agent_tritonbench.benchmark \
  --reference ../ref/tritonbench \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --suite experiments/agent_tritonbench/suite-100.json \
  --task abs --language intent \
  --program "$HOME/.local/share/intentdsl/agent-evaluation/first-stage-high-100-intent-contraction-closure/abs/intent/candidate.py" \
  --gpu-lock "$HOME/.local/share/intentdsl/agent-evaluation/gpu.lock" \
  --result experiments/agent_tritonbench/results/integration/abs-intent.json
```

2026-09-22 目录整理后，已有 Intent `abs` 程序在 RTX 5090D 上重测 pass，结果见 [abs-intent.json](results/integration/abs-intent.json)。该单点不更新 H100 全量成绩，也没有重新调用生成模型。

## 独立单次生成

只有需要新生成实验时执行。使用已配置的独立 Luna 环境，`--state-root` 必须在仓库外；材料仍是通用语法/语义，不加入题解或 reference：

```bash
python -m experiments.agent_tritonbench \
  --reference ../ref/tritonbench --triton-ref ../ref/triton \
  --compiler /tmp/intentdsl-build/tools/intent-compile/intent-compile \
  --codex "$INTENT_STUDY_CODEX" --state-root "$INTENT_STUDY_STATE_ROOT" \
  --suite experiments/agent_tritonbench/suite-100.json \
  --tasks abs --arms intent --workers 1 --benchmark-workers 1 \
  --output experiments/agent_tritonbench/results/new-abs
```

`--output` 必须是新目录。去掉 `--tasks` 执行固定全题集；是否进行全量生成由当前任务授权决定。准备与生成可并发，GPU 执行沿用同一 lock。后续单点、pass 对照及正式复测都输出到本组 `results/`，记录实际计时合同，不把 CUDA Graph 之外的 host 优化计为 GPU kernel 收益。
