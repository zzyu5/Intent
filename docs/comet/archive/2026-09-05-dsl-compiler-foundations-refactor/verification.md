---
generated_from_state_version: 8
---

# 验证

## 当前结果

- 结果: **已归档**
- 验证情况: **已完成检查，验证结果已确认**
- 目标周期: 1
- 迭代: 1
- 验证器尝试次数: 1
- 完成时间: 2026-09-05T18:19:30.980Z
- 摘要: 独立核对 brief/spec、当前源码、Triton/cuTile/TileLang reference 差异及已有 production DSL→emit→backend→numerical repro；四项验收均满足。未运行新测试体系、未扩大到全量 examples/provider 或性能目标。

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | A1：作者能通过已确认的具名计算接口编写对应运算，签名、shape/dtype/default 和诊断可发现；操作机械归一到已有语义，不要求重复填写可由操作定义确定的轴/组合规则。 | 具名 dot/matvec/vecmat/matmul/outer、cumsum/cummax、scaled_matmul、sparse_matmul 已注册并暴露完整签名，见 python/intent/language/signatures.py:22-60；矩阵操作统一生成 batch broadcast 与 CONTRACT，见 python/intent/frontend/lowering/intrinsics/matrix.py:15-76。生产 author-ops repro 已通过，生成源码中的签名与数值结果见 /mnt/hdd/tmp/intentdsl-foundations-refactor.YmotCl/author-ops.log:1-3,111,219,328-329。 |
| A2 | passed | brief.md | A2：已有复杂算法仍能使用具名操作、typed helper、generic structured constructs 和普通 control/effects 组合；在本轮涉及的目标路径上形成可执行后端程序，不把局部形态差异误判为算法能力边界。 | generic reduce/scan、typed @intent.fn helper、region 与普通控制路径仍保留；helper region 纯性和 schema postcondition 检查见 python/intent/frontend/lowering/intrinsics/structured.py:407-564,880-995。具名操作复用 emit_contract/emit_scaled_contract/emit_sparse_contract，见同文件:567-778。author-ops 与 batch Triton/cuTile 生产生成及执行均为 max_abs=0，见 /mnt/hdd/tmp/intentdsl-foundations-refactor.YmotCl/author-ops.log:111,219,328-329、batch-triton.log:203、batch-cutile.log:252-253。 |
| A3 | passed | brief.md | A3：纳入本轮的已知数值/effect 问题兑现明确契约；f32 输入精度、signed floor/remainder、logical index 和一次调用的可观察 effect 不继承不等价的 provider 默认值。 | Triton contraction serializer 明确输出 input_precision="ieee"，见 lib/Target/Triton/Serialization/Serializer.cpp:977-999；Triton signed floor/remainder 保持显式关系，见同文件:1278-1290；cuTile logical ABI 使用 int64，见 lib/Target/CuTile/Serialization/Serializer.cpp:27-30,374-381；autotune writable alias 通过 TuningState 的 scratch/reset/restore 隔离并恢复，见 python/intent/runtime/tuning.py:13-77。既有 numeric/effect repro 已记录 f32=1.00146484375、signed quotient/remainder=-2/1、logical loop=2147483648、三 provider inout 两次均 +1，见 docs/comet/changes/dsl-compiler-foundations-refactor/comet-state.yaml:85-95。 |
| A4 | passed | brief.md | A4：纳入本轮的 transformation/config 重构具有清楚职责：完整阶段的 postconditions 可定位，有限候选数据与选择逻辑分离，合法性不被固定搜索预算冒充；保留既有 typed IR 权威。 | shared transformation groups 明确定义并在每组后执行 verifyGPUProgram postcondition，见 lib/Dialect/GPU/Transforms/Passes.cpp:8-25,137-173；有限 JSON 数据与选择逻辑分离，配置读取、字段/家族/类型/空表诊断及整组替换见 lib/Dialect/GPU/Transforms/TuningProfiles.cpp:12-146，typed domain 合法性过滤和结果写入 typed IR 见 lib/Dialect/GPU/Transforms/MaterializeConfigTuples.cpp:458-679；CLI/Python tuning_config 入口贯通，见 tools/intent-compile/intent-compile.cpp:77-145、python/intent/compiler/pipeline.py:15-73。 |

## 检查

| 检查 | 命令 | 工作目录 | 状态 | 退出码 | 耗时 |
| --- | --- | --- | --- | ---: | ---: |
| DSL operations and typed helper production repro | PYTHONPATH=python:examples PYTHONDONTWRITEBYTECODE=1 /home/kingdom/.venvs/intentdsl-cutile/bin/python /tmp/intentdsl-foundations-refactor.YmotCl/author_ops.py --tuning-config /tmp/intentdsl-foundations-refactor.YmotCl/tuning.json | . | passed | 0 | 5110 ms |

## 阻塞项

_无。_

## 风险与跳过的工作

_未报告风险。_

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | pass | — | 独立核对 brief/spec、当前源码、Triton/cuTile/TileLang reference 差异及已有 production DSL→emit→backend→numerical repro；四项验收均满足。未运行新测试体系、未扩大到全量 examples/provider 或性能目标。 | 2026-09-05T18:19:30.980Z |



## 结论

独立核对 brief/spec、当前源码、Triton/cuTile/TileLang reference 差异及已有 production DSL→emit→backend→numerical repro；四项验收均满足。未运行新测试体系、未扩大到全量 examples/provider 或性能目标。
