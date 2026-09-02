# 目标

完成当前 05c 尚未闭合的 shared GPU executable correctness、typed config/resource/device 边界与 provider lowering 骨架，使现行 Triton registry 能在 RTX 5090D 和 H100 上真实完成生成代码、数值校验与默认 config 性能比较，并留下两张当前 CSV 与简短状态结论。

# 范围

- 收紧 shared physical relation authority：相同 logical dimension 不再被当作相同 source occurrence 或 traversal；current IR mutation 后从精确 source/range/def-use facts 重算，unknown 不提升为 exact。
- 补齐 complete config tuple、resource legality 与 compile-device binding：只删除 current Physical Program 和 typed capability 能证明非法的候选；provider-local candidate set 在 serializer 前闭合；artifact、allocation 与 launch 绑定同一编译设备。
- 闭合当前仍成立的 public/provider 缺口：`I.sparse_contract_2to4` 机械归一到唯一 canonical sparse contract；TileLang 为 ref 已证明存在的 scaled contract 建立 provider form 与 capability rejection；cuTile rewrite 后执行 op schema verification；TileLang serializer 不再拥有 config legality 决策。
- 用当前 production 路径运行 Triton registry；明显超过默认 config 1.1× 的项只从 Physical Program、typed config 或 provider form 修复，不增加 entry-local policy。

## 来源覆盖

| 单元 | 来源定位 | 读取 | 保留语义 | Spec / 验收 | 状态与理由 |
| --- | --- | --- | --- | --- | --- |
| P1 | `report/prompt/05c-intentdsl-shared-gpu-correctness-closure-prompt.md:1-19` | complete | shared correctness、config 分层、Triton 双机数值与默认 config 1.1× 结果 | Spec 全文；A1-A4 | covered；历史计数只作当时起点 |
| P2 | 同上 `:22-194` 及其要求完整阅读的四份历史报告 | complete | 规格入口、ref 对照和冻结全量纪律 | 约束与验证预期 | background；旧通过率、失败数与未提交状态已由当前代码调查取代 |
| P3a | 同上 `:197-215,225-413` | complete | authority、complete tuples、provider-local options、可证明 resource legality | Spec「Shared program」「Config、resource 与 device」；A1-A2 | covered；预写执行顺序不保留为需求 |
| P3b | 同上 `:216-223` | complete | 当时排除 cuTile/TileLang provider forms | Spec「Public shorthand 与 provider forms」；A3 | superseded；当前用户明确把 TileLang scaled contract 与 cuTile verifier 作为调查入口，当前 code/ref 核验确认仍是本轮缺口 |
| P4a | 同上 `:416-442` | complete | cuTile/TileLang 纯编译 probe | 验证预期 | superseded；当前用户要求其它全量扫描仅在帮助实现时运行，不设为验收项 |
| P4b | 同上 `:443-487` | complete | Triton 双机真实 JIT/launch/numerical/performance 与两张 CSV | Spec「Triton 终端结果」；A4 | covered |
| P4c | 同上 `:488-510` | complete | shared 不变量与 build-mode 调查 | 约束与验证预期 | background；按受影响范围自查，不机械增加 verifier 或强制额外全量 |
| P5 | 同上 `:513-565` | complete | ref 双向自查、唯一 executable path、语义完整提交 | 约束与验证预期 | background；不把逐项证据、调试命令或长报告变成验收项 |
| R1 | `report/reviwe.md:1-133` | complete | relation、device、TileLang config/scaled form、cuTile verifier、sparse shorthand 的当前调查入口 | Spec 前三节；A1-A3 | covered；逐项对照当前代码，已修或不成立者不进入范围 |
| R2 | `report/reviwe.md:135-197` | complete | fixed profiles、profile index 组合、重复 relation pass、宽 serializer 文件本身均不是缺陷 | 非目标 | non-goal；只有真实 lowering、legality 或性能后果才处理 |
| R3 | `report/reviwe.md:199-211` | complete | 以当前运行和 current IR authority 判断 05c 状态 | Spec 全文；A1-A4 | covered；旧 CSV 不代表当前结果 |

# 非目标

- 不修改 `doc/` 已定义的 DSL/canonical semantics，不恢复旧 Plan/materializer、第二 executable path、compatibility switch、fallback 或异常切换。
- 不建设 test 目录、pytest、fixture、兼容层、版本管理、额外证据文档或长期 inventory runner。
- 不把 fixed tuning table、profile index 组合或缺少 cost model 本身当问题；不做公平 candidate 对齐、source autotune config 匹配、generated winner 搜索、1.05× 闭环或 1.1× 内进一步优化。
- 不为 kernel 名、entry 名、source ID、裸 shape/rank 或单条语料增加 compiler policy。
- 不要求 cuTile/TileLang 双机 GPU 全量或性能表；它们只做本轮实现所需的 production-path repro，额外全量仅在确实帮助当前实现时运行。

# 验收示例

- A1：受影响的 canonical kernels 经 shared construction 与 mutation 后形成可独立验证的完整 GPU Program；source occurrence、traversal、range、ownership、access、validity 与 effects 由 current typed facts 唯一决定，不再以 dimension equality 猜 exact relation。
- A2：每个终端候选在 serialization 前都是完整的 shared tuple 与 provider-local options，resource filter 只删除可证明非法项；默认 config 使用同一候选 authority；compile device 进入 artifact/runtime binding，跨设备输入被明确拒绝，无 tensor 输入也显式使用绑定设备完成 allocation 与 launch。
- A3：rank-2 `I.sparse_contract_2to4` 形成唯一 canonical sparse contract；受支持的 TileLang scaled contract 到达 provider-native form、不支持的 capability 在 serializer 前精确拒绝；cuTile rewrite 后 op schema 与 closed surface 均被验证；TileLang serializer 只打印已闭合 config 集合。
- A4：当前 Triton registry 的 54 个 entries / 62 个 component references 在 RTX 5090D 与 H100 上真实执行完整 generated/source workflow 并得到可归因的终端结果；双方完成且计时可比的 entry 记录实测数值与 ratio，默认 config 以 1.1× 为目标，明显超标项优先从 Physical Program、typed config 或 provider form 调查；source resource/compatibility、整体 worker timeout 或 measurement gap 保留准确状态而不伪造 ratio，两张 Triton CSV 记录同一当前代码状态。

# 约束与不变量

- `doc/` 是设计权威；历史 report、source corpus、registry 与 CSV 只提供调查和运行事实。
- canonical KIR 在 physical construction 后保持 immutable；shared GPU IR 是唯一完整 executable authority；serializer 不重建 structure、legality、device 或 candidate decisions。
- Analysis unknown 保守保留合法程序或在最早拥有充分信息的 legality 层精确拒绝，不能转成默认 source、dimension、range、tuple 或 device。
- 所有普通与 atomic write effects、multi-source traversal、buffer lifetime/resource uses、provider forms 和 runtime binding 必须消费 typed carrier；mutation 后相关 analysis 失效或重算。
- 修改按语义完整节点提交；不留下旧/新路径并存、artifact 后过滤、临时脚本、cache 或生成源码。

# 决策

- 使用一个普通 Native Change，不拆 Supervisor。剩余问题共同修改同一 shared/provider/runtime executable path，并由同一 Triton 双机闭环验证；拆分会增加 candidate、artifact 与 provider 集成的重复协调。
- 已确认 shared config tuple carrier、Triton closed configs、atomic write ownership 与 runtime multi-reduction logical stop 已存在；本轮不重复实现，只在后续改动影响它们时保持语义。
- 当前真实剩余包括 occurrence-aware traversal/extent authority、physical liveness/interference 与 resource legality、compile-device artifact binding、TileLang config authority、cuTile post-rewrite MLIR verification、typed sparse shorthand 和 TileLang scaled provider form。
- `I.sparse_contract_2to4` 保留历史 rank-2 convenience shape，但 metadata 使用现行 canonical `{first, second}` logical-position record；它固定 two-of-four、lhs axis 1、rhs axis 0、`reduce=((1,0),)` 与空 batch，不能接受 opaque packed `i16` metadata 冒充 canonical schema。
- 最终状态只更新 `report/baselinev2/triton-5090.csv`、`report/baselinev2/triton-h100.csv` 和一份足以说明 05c 当前结论的简短结果，不生产长证据报告。

# 待解决问题

- [blocking] CONFIRM: A1-A3 保持原确认范围；A4 恢复为双机完整真实运行与可归因终态，对可比项记录数值和 ratio、以默认 config 1.1× 为目标并调查明显异常，对 source resource/compatibility、整体 timeout 或 measurement gap 如实记录而不伪造 ratio；非目标不变。

# 验证预期

- 每个语义完整实现节点先用现有 production compile/emit/run 路径做一个足以说明数值正确的 repro，再继续，不建立测试体系。
- 最终在本机 RTX 5090D 与 H100 执行环境分别运行当前 Triton registry 的真实 generated/source 数值和计时流程；两侧保持算法、dtype、ABI、shape、调用次数与计时范围一致，明显异常项只独立复测一次。
- 主要 authority/provider 修改按 `ref/triton` 或 `ref/tilelang` 同类实现自查，交付时给双方 `file:line`、具体差异与实际后果。
- 全量运行使用构建完成后不会被 relink 的同一 compiler 状态；修改 compiler 后废弃旧结果，不混合不同 binary 的数据。
