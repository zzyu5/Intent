# 目标

按 `doc/compiler/cpu-program-ir.md` 重构 CPU 编译分层：保留 task/block 与 structured compute 的共同程序，由正式 lowering 查询目标实现需求、选择并实例化专家编写的微程序。收束当前 f32 向量偏好过早进入 shared blocking、配置职责混合，以及 Mojo/Weft 缺少可编程实现接口的问题；不是取消 CPU IR、整算子库旁路或只做接口搬家。

# 范围

- 保留 canonical KIR→CPU 的唯一 construction、typed ABI、任务/坐标、effects、共同 fusion/reuse/lifetime 与 verifier；共同程序不无条件展开为 Mojo SIMD，也不复制 GPU warp/lane 模型。
- 将外层 task/cache blocking 与实现内部 microtile、packing、vector/replica 等需求分开；需求协调、布局/供应和展开消费同一个候选选择。
- 建立有实际消费者的 implementation 适用性、需求、有限参数与结构化展开机制，现有 f32 Mojo contraction/vector 路径迁入同一正式编译链，保留 native artifact 与现有五项 f32 registry 的可达能力。
- Weft 继续以 Canonical Weft IR 为输入；补专业微程序所需的结构化实例化/集成边界，保留简单操作直接转换，不在 emitter 拼接整段算法。
- 用一个实际专业计算块证明机制不只包装原 f32 builder。首个格式/计算块是 Q1；建议采用已有预量化 Q4_K×Q8_K 输入的局部点积，仅生成接入，不连带 activation quantization、persistent repack 或设备 runtime。
- 运行复用既有 Mojo 性能入口，只重跑受影响项，在同次 benchmark 做一次原容差检查并更新项目 CSV；不为归档或 Shape 重跑性能。

## Source coverage

当前 CPU doc 与已提交报告已完整读取。下表覆盖相应文档全部章节；`covered` 表示已映射到本草案，不表示 Shape 已获最终确认。参考链接仅作为实现证据，不把其全部上游能力扩成需求。

| 来源单元 | 读取 | 归类与保留内容 | Spec 对应 | 验收 | 覆盖状态 |
|---|---|---|---|---|---|
| CPU doc §1–2 | complete | execution family、task/block、完整 structured IR、唯一程序 | §1–2 | A1 | covered |
| CPU doc §3 | complete | 作者、共同 passes、implementation、外部 compiler 的职责 | §2–3 | A1、A2 | covered |
| CPU doc §4 | complete | 适用性、需求、参数、选择/展开与程序集成 | §3 | A2、A3 | covered |
| CPU doc §5 | complete | ABI/effects/lifetime 与格式、数值义务 | §2、§4–5 | A1、A3 | covered |
| CPU doc §6 | complete | 能力与共享/实现/外部参数分工，有限选优、cache | §3、§6 | A2、A4 | covered |
| CPU doc §7–8 | complete | Mojo/Weft 停止层次、verifier、native 与 generation 边界 | §4、§6–7 | A2–A4 | covered |
| 报告导言、§1 | complete | 已确认架构方向；报告本身非设计权威 | §1–3 | A1、A2 | covered |
| 报告 §2、§3（含表格与链路） | complete | 当前实现与 TileLang 证据背景；不移植完整上游能力 | §1、§3 | — | background |
| 报告 §4–6（含程序示意与职责表） | complete | structured CPU 模型、作用范围与可编程实现接口 | §2–3 | A1、A2 | covered |
| 报告 §7（含量化准备例子） | complete | 格式语义与实现分开，不隐藏量化准备/跨调用资源；首个计算块待选 | §5 | A3 | needs-clarification：Q1 |
| 报告 §8 | complete | 复用 Mojo 现有微核；Weft helper/Canonical adapter 边界 | §4–5 | A2、A3 | covered |
| 报告 §9（含模块表、配置与唯一迁移路径） | complete | 分层迁移范围与实际参数消费者，不只重命名 | §2–4、§6 | A1、A2、A4 | covered |
| 报告 §10 | complete | 架构已由 CPU doc 固定；具体新增量化合同不得假定已定义 | §1、§5 | A1、A3 | needs-clarification：Q1 |
| 报告 §11 | complete | 真实实现闭合、生产 benchmark；首个专业计算块与范围仍需确认 | §5–7 | A2–A4 | needs-clarification：Q1 |

来源：`doc/compiler/cpu-program-ir.md`、`report/cpu-programmable-lowering-reassessment.md`；同层 compiler/DSL 文档与 `AGENTS.md` 继续约束本 change。

# 非目标

- 不重构 GPU、不进入 DSA/Ascend，不自动扩展所有 dtype、量化格式或输入矩阵。
- 不把 Weft Encoding/Level 整套加入 Intent public surface，不根据 kernel 名选整算子模板，不新增 runtime 算法替代路径。
- 不隐式启用 fast-math/FTZ，不以放大容差换通过；不新增性能 benchmark 之外的测试。
- 不自动接入 Weft 设备 discovery/JIT/远程执行；外部 TianchenRV 默认只读，若实际接口缺口要求修改外部仓库，先明确范围并取得授权。
- 不新增 worktree，不推送、不创建 PR；未受影响的历史性能保留，不默认全量重跑。

# 验收示例

- A1：现有 CPU 程序保留完整 task/block、访问、依赖与 structured compute；共同 blocking/参数不再无条件绑定特定 f32 向量微核的内部组织，Mojo/Weft 从同一共同程序边界继续 lowering。
- A2：实际计算块通过明确的实现适用性、需求协调、参数绑定与微程序展开形成可验证 IR；现有 f32 Mojo native 路径在新机制中真实可调用，布局/供应与展开使用同一选择，无新旧旁路并存。
- A3：经 Shape 选定的专业计算块沿正式 Intent→CPU→provider 编译链形成真实目标程序，格式/数值与宿主 captures、axes、domain、输出和 lifetime 完整连接；不是原 f32 builder 改名或整算子库调用。具体计算块与运行边界待 Q1 固定。
- A4：shared/implementation/provider 参数都有真实消费者，有限候选实测选优且 artifact/winner 复用；受影响的既有 Mojo benchmark 得到同算法 generated/source 时间和 G/S，并在同次运行通过原容差，结果写入现有 CSV。

# 约束与不变量

- `doc/` 为设计权威；本 change 是其实现收束。报告提供已读证据与范围线索，旧归档的非目标、失败状态和性能门槛不自动变成本轮约束。
- 专家 implementation 可编程，展开必须成为 current IR；serializer 只拼写。合法直接映射与专业实现按适用条件共存，不以失败后 fallback 决定程序。
- 格式/算法选择、实现选择、参数选择分别归属。若 Q1 选中的格式尚无 DSL/KIR 数值合同，先明确该最小合同，不以一个 format 字符串冒充完成。
- 外围共享量化准备、数据供应与资源可见；微程序不能每个 output tile 重复 activation quantization，不能隐藏 persistent repack/缓存。
- 自查对照 current/spec/ref 的 file:line、具体差异与后果；不增加独立测试门禁。

# 决策

- 用户已确认将 CPU 模型和可编程 lowering 边界固定到 `doc/`，文档与报告已提交为 `b164302`。
- 用户已明确接受 `cpu-pointwise-reductions` 的 4/4 验收并授权归档；归档提交为 `44c422c`。本轮不再恢复该任务。
- 用户授权创建新 change，沿用 main/current；此授权不是本 Shape 的最终 Build 确认。
- 采用单个普通 Native change：共享阶段、实现接口、bindings 与 Mojo/Weft 集成会共同修改同一组核心边界，不按 provider 拆分 Supervisor。
- 关键参考：TileLang `tileop/gemm/__init__.py:121–139` 分开 infer_layout/lower，`cuda/op/gemm/gemm_mma.py:109–142` 生成可编程 PrimFunc，`src/op/gemm.cc:198–236` 插回 IR；当前 CPU `BlockContractions.cpp:69–144` 提前固定向量微 panel，`MaterializeRegisterContractions.cpp:51–145` 是单一 f32 实现，Weft `Legalize.cpp:566–588` 尚无专业微程序实例化接口。

# 待解决问题

- [blocking] Q1：本轮首个专业微程序是否采用预量化 Q4_K×Q8_K 的局部点积，补最小格式/数值合同并生成 Canonical Weft IR，但不包含 activation quantization、persistent repack 和设备 runtime？建议采用，以真实量化实现检验新的 lowering 机制；最终 Shape 将据此固定完整接口与验收范围。

# 验证预期

仅使用既有生产编译/生成与 `examples/repro/v2` 性能入口。既有 f32 五项保持可达，仅受实际改动影响的项进行 benchmark；测 native 算子时间，包含 invocation 内 packing/任务同步，排除编译、tuning、加载与外部输出分配。沿用各项现有 tolerance，不新增数值/边界/回归/组合测试。Weft generation 不冒称数值运行或性能达标；新专业项的接口与完成边界由 Q1 及最终 Shape 确认。
