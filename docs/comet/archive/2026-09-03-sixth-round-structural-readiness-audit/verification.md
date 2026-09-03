---
generated_from_state_version: 25
---

# 验证

## 当前结果

- 结果: **已归档**
- 验证情况: **已完成检查，验证结果已确认**
- 目标周期: 2
- 迭代: 4
- 验证器尝试次数: 2
- 完成时间: 2026-09-03T17:51:36.487Z
- 摘要: Ready: shared indexed-access correctness and provider validity boundaries are closed; the sixth round can proceed predominantly through cuTile and TileLang leaf lowering without a planned shared/core redesign.

## 验收

| 编号 | 结果 | 来源 | 验收项 | 原因 |
| --- | --- | --- | --- | --- |
| A1 | passed | brief.md | Scenario 1：给定包含动态或间接索引的 canonical kernel，shared construction 产生的每个可达访问都由 current typed validity、可验证 relation 范围或匹配的 canonical in-bounds 前提证明安全；无法证明的访问在 provider lowering 前被精确拒绝。 | Shared construction materializes bounds validity/fill/no-effect or exact proof for every indexed access; current accessBounds and buffer-dataflow verification reject incomplete programs after composition and structured replay. |
| A2 | passed | brief.md | Scenario 2：给定同一 shared GPU Program 分别进入 Triton、cuTile 与 TileLang，provider legalization 只从 current typed facts 选择 local form，改写后的 local ops/resources/control/config 由 MLIR op verifier 与 provider verifier 完整闭合，serializer 仅作确定性 spelling 与已声明 wrapper 绑定。 | Triton, cuTile and TileLang legalization consume current typed facts. cuTile preserves active validity through checked gather/scatter; TileLang copy ops carry boundary_axes, locally verify their region mapping and disable TMA exactly when boundary predicates must remain active; serializers only spell the closed local form. |
| A3 | passed | brief.md | Scenario 3：给定当前 cuTile/TileLang 未覆盖或低效的可达形态，审查结果证明它们要么能以 provider-local IR/pass/verifier/serializer/runtime 工作完成，要么在最早拥有足够信息的 provider/capability 层精确拒绝；任一必须新增 shared 语义或大改 common program 的反例都使 Ready 结论失败。 | No remaining cuTile or TileLang gap requires a new shared semantic carrier or common-program redesign. Unsupported forms are provider-local leaf work or precise capability rejection. |
| A4 | passed | brief.md | Scenario 4：给定一条现有 production compile/emit/JIT/launch 命令，它实际运行一个由目标 provider 当前完整支持、同时覆盖 shared access 与 provider-local form 的代表性 kernel，并通过数值比较；审查结论不依赖新测试体系或重跑全量 registry。 | The existing production path rebuilt all changed compiler/provider components and completed RTX 5090D Triton 3.6 compile, emit, JIT and launch with numerical pass; generated=2.394216 ms, source=2.361344 ms, ratio=1.013921. |

## 检查

_没有记录 Runtime 检查。_

## 阻塞项

_无。_

## 风险与跳过的工作

- TileLang boundary-bearing external copies intentionally exclude automatic TMA until a provider-local complete in-bounds proof exists.
- Broader cuTile and TileLang feature coverage remains leaf work and does not invalidate shared structural readiness.

## 之前的迭代

| 目标周期 | 迭代 | 尝试 | 结果 | 未解决项 | 摘要 | 完成时间 |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 1 | 1 | fail | A1, A4 | Verifier verdict is fail: A1 and A4 failed; A2 and A3 passed. Provider boundaries and leaf rejection are structurally adequate, but reachable canonical-to-shared indexed accesses can lack resource-bounds validity and reach unmasked provider addresses. Missing assume_in_bounds/validity must be treated as a verifier rejection or explicit validity construction, not silently accepted caller error. | 2026-09-03T10:44:30.831Z |
| 1 | 2 | 0 | recovery | — | Native confirmed acceptance criteria changed | 2026-09-03T16:46:16.582Z |
| 2 | 1 | 1 | fail | A1 | Provider discipline and production execution pass, but the verifier requests a concrete construction reachability ruling for advanced-index transpose AxisMap selection before accepting A1. | 2026-09-03T16:57:22.190Z |
| 2 | 2 | 1 | recovery | — | Repair verification passed for A1; final full verification is required. | 2026-09-03T17:02:41.143Z |
| 2 | 2 | 2 | fail | A1, A2 | Final verification is Not Ready until cuTile tile-form legalization preserves or precisely restricts shared tail validity. | 2026-09-03T17:08:19.818Z |
| 2 | 3 | 1 | fail | A2 | A1 and cuTile pass; A2 remains pending a source-level TileLang ordinary-copy boundary semantics ruling. | 2026-09-03T17:25:33.985Z |
| 2 | 4 | 1 | recovery | — | Repair verification passed for A2; final full verification is required. | 2026-09-03T17:45:27.050Z |
| 2 | 4 | 2 | pass | — | Ready: shared indexed-access correctness and provider validity boundaries are closed; the sixth round can proceed predominantly through cuTile and TileLang leaf lowering without a planned shared/core redesign. | 2026-09-03T17:51:36.487Z |



## 结论

Ready: shared indexed-access correctness and provider validity boundaries are closed; the sixth round can proceed predominantly through cuTile and TileLang leaf lowering without a planned shared/core redesign.
