# Baseline V2

## 用途

Baseline V2 用来观察 Intent compiler 在真实 kernel 上是否正确、是否能生成有竞争力的 provider source。它不是编程模型、DSL 或 compiler 的版本，也不参与 compiler 的 pass、form 或 tuner 选择。

一行性能差距只用于定位问题：先确认作者 DSL 与 source 是否在计算同一个算法，再判断问题位于 KIR、shared GPU passes、provider lowering、下层 compiler，还是测量本身。不能为了让某行变快而增加 kernel-name 分支、source template 或 fallback。

## Source 与输入

每个 entry 都来自 `source/` 中保存的公开 provider implementation，并有相邻的可执行 runtime：

- Triton：[source inventory](../../source/triton/README.md)；
- cuTile：[source inventory](../../source/cutile/README.md)；
- TileLang：[source inventory](../../source/tilelang/README.md)。

可执行 entry 顺序由 [`examples/repro/v2/registry.py`](../../examples/repro/v2/registry.py) 给出。Triton 表包含 54 条 source-backed entries：38 条 Baseline V2 主清单和 16 条继续观察旧覆盖面的 entries；cuTile 与 TileLang 各 37 条。

输入采用模型部署中会实际出现的 operator shape，例如大尺寸 GEMM、长序列 attention、paged decode、MoE、Mamba 和大 hidden-size normalization。`M=1`、`B=1` 若对应 decode 或 state-space phase，仍是真实 workload，不是为了让 kernel 容易通过而缩小的 toy case。MoE alignment、expert projection、mHC pre/post 等记录的是上游公开 callable 对应的真实子算子，不冒充完整模型端到端时间。

## 六张结果表

| provider | RTX 5090D | H100 |
|---|---|---|
| Triton | [`triton-5090.csv`](triton-5090.csv) | [`triton-h100.csv`](triton-h100.csv) |
| cuTile | [`cutile-5090.csv`](cutile-5090.csv) | [`cutile-h100.csv`](cutile-h100.csv) |
| TileLang | [`tilelang-5090.csv`](tilelang-5090.csv) | [`tilelang-h100.csv`](tilelang-h100.csv) |

两台机器各看自己机器上的 generated/source 结果，不直接比较 5090D 与 H100 的绝对延迟。三家 provider 也各自保存独立表，不合并成一张总表。

当前 CSV 恢复的是完整重构前最后保存的真实测量值；Triton 与另外两家并非同一时刻更新。它们保留为重新测量前的观察基线，不能冒充重构后 compiler 的当前结果。下一次对应机器全量运行会直接覆盖该表。

## 字段

```text
kernel,case,generated_p50_ms,source_p50_ms,ratio,status
```

- `generated_p50_ms`：Intent generated provider callable 的 steady-state 中位延迟；
- `source_p50_ms`：相邻 upstream runtime 的 steady-state 中位延迟；
- `ratio`：`generated/source`，大于 1 表示 generated 较慢；
- `status=pass`：已经完成生成侧、source侧、数值比较和计时；
- 空时间字段：该 entry 没有形成合法可比较的性能值，原因保存在 `status`。

数值失败不记录性能。编译、首次 JIT、输入和 workspace 构造不计入 p50；一次 entry 若本来就是多-kernel pipeline，两侧都计完整 pipeline launch。

## 运行

从仓库根目录、对应 provider 环境执行：

```bash
./examples/run/baseline-v2.sh triton report/baselinev2/triton-5090.csv
./examples/run/baseline-v2.sh cutile report/baselinev2/cutile-5090.csv
./examples/run/baseline-v2.sh tilelang report/baselinev2/tilelang-5090.csv
```

H100 使用对应的 `*-h100.csv` 路径。单条复现把 entry 名作为第三个参数传入；单条运行仍会覆盖输出文件，因此正式表由完整 provider run 产生。

CSV 只保存观察结果，不附带阈值检查、自动结论或 pass/fail policy。具体根因和修复记录写入独立 `report/`，不写进表格。
