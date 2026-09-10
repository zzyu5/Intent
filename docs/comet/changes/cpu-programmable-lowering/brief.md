# 目标

按 `doc/compiler/cpu-program-ir.md` 重构 CPU 编译分层：保留 task/block 与 structured compute 的共同程序，由正式 lowering 查询目标实现需求、选择并实例化专家编写的微程序。收束当前 f32 向量偏好过早进入 shared blocking、配置职责混合，以及 Mojo/Weft 缺少可编程实现接口的问题；不是取消 CPU IR、整算子库旁路或只做接口搬家。

# 范围

- 保留 canonical KIR→CPU 的唯一 construction、typed ABI、任务/坐标、effects、共同 fusion/reuse/lifetime 与 verifier；共同程序不无条件展开为 Mojo SIMD，也不复制 GPU warp/lane 模型。
- 将外层 task/cache blocking 与实现内部 microtile、packing、vector/replica 等需求分开；需求协调、布局/供应和展开消费同一个候选选择。
- 建立有实际消费者的 implementation 适用性、需求、有限参数与结构化展开机制，现有 f32 Mojo contraction/vector 路径迁入同一正式编译链，保留 native artifact 与现有五项 f32 registry 的可达能力。
- Weft 继续以 Canonical Weft IR 为输入；补专业微程序所需的结构化实例化/集成边界，保留简单操作直接转换，不在 emitter 拼接整段算法。
- 用一个实际专业计算块证明机制不只包装原 f32 builder。Q1 已确认采用已有预量化 Q4_K×Q8_K 输入的局部点积，仅生成 Canonical Weft IR，不连带 activation quantization、persistent repack 或设备 runtime。
- 最终 Shape 提议新增闭合的 `I.quantized_dot` 与 `I.quant.q4_k/q8_k` 格式 schema：输入为已有 `u8` record tensors `[G,144]`、`[G,292]`，输出 rank-0 f32；具体布局、整数统计和浮点舍入合同见 Spec §5。不新增可编程 Encoding 类型或 opaque kernel handle。确认后先补 `doc/dsl/` 的最小规范，再迁移实现。
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
| 报告 §7（含量化准备例子） | complete | 格式语义与实现分开；采用预量化 Q4_K×Q8_K，不隐藏量化准备/跨调用资源 | §5 | A3 | covered |
| 报告 §8 | complete | 复用 Mojo 现有微核；Weft helper/Canonical adapter 边界 | §4–5 | A2、A3 | covered |
| 报告 §9（含模块表、配置与唯一迁移路径） | complete | 分层迁移范围与实际参数消费者，不只重命名 | §2–4、§6 | A1、A2、A4 | covered |
| 报告 §10 | complete | CPU 架构保持；新增量化入口、布局和数值合同先经最终 Shape 确认，再进入正式 DSL 规范 | §1、§5 | A1、A3 | covered |
| 报告 §11 | complete | 真实实现闭合、生产 benchmark；Q4_K×Q8_K 仅生成，Mojo 保留 native 边界 | §5–7 | A2–A4 | covered |
| 用户 Q1 回答“可以” | complete | 同意预量化局部点积与 generation-only 范围，不代表最终 Build 确认 | §5 | A3 | covered |

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
- A3：真实 Intent 作者 kernel 用 `I.quantized_dot` 消费 `[G,144]`/`[G,292]` u8 records 并写出 f32 点积；经唯一 KIR→CPU→Weft 链形成可验证 Canonical Weft IR。§5 的格式、i32 统计、f32 校正和 reduction 合同完整保留；输入 captures/轴/域、结果消费、effects/lifetime 与必要 ABI 条件显式连接。生成物包含专家程序展开后的实际计算，不残留待 emitter/runtime 解释的 recipe，也不调用完整 source kernel。不包含 activation quantization、persistent repack 或设备运行。
- A4：shared/implementation/provider 参数都有真实消费者；Mojo 的有限合法候选实测选优且 artifact/winner 复用，Weft generation 不冒称实测选优。受影响的既有 Mojo benchmark 得到同算法 generated/source 时间和 G/S，并在同次运行通过原容差，结果写入现有 CSV。

# 约束与不变量

- `doc/` 为设计权威；CPU 分层按既有规范收束。唯一新增语言设计是待最终确认的 §5 闭合量化点积与格式/数值合同，确认后先补对应 DSL 规范；不为了保留旧实现改写其它语义。报告提供已读证据与范围线索，旧归档的非目标、失败状态和性能门槛不自动变成本轮约束。
- 专家 implementation 可编程，展开必须成为 current IR；serializer 只拼写。合法直接映射与专业实现按适用条件共存，不以失败后 fallback 决定程序。
- 格式/算法选择、实现选择、参数选择分别归属。Q4_K×Q8_K 尚无对应 DSL/KIR 数值合同，先明确该最小合同，不以一个 format 字符串冒充完成。
- 外围共享量化准备、数据供应与资源可见；微程序不能每个 output tile 重复 activation quantization，不能隐藏 persistent repack/缓存。
- 自查对照 current/spec/ref 的 file:line、具体差异与后果；不增加独立测试门禁。

# 决策

- 用户已确认将 CPU 模型和可编程 lowering 边界固定到 `doc/`，文档与报告已提交为 `b164302`。
- 用户已明确接受 `cpu-pointwise-reductions` 的 4/4 验收并授权归档；归档提交为 `44c422c`。本轮不再恢复该任务。
- 用户授权创建新 change，沿用 main/current；此授权不是本 Shape 的最终 Build 确认。
- Q1 已获用户确认：首个专业微程序采用预量化 Q4_K×Q8_K 局部点积，仅生成 Canonical Weft IR；不包含 activation quantization、persistent repack 或设备 runtime。本次回答只解决计算块与运行边界，完整接口和数值合同仍须纳入最终 Shape 确认。
- 最终 Shape 的接口提议：普通 u8 carrier 保持现有 ABI；closed format schema 定义固定 record mapping、`bsum` 一致性及量化点积数值语义。Q4_K/Q8_K 不是新 scalar dtype，也不借 `scaled_contract` 或 arbitrary `assume` 承载。具体专业程序可以用结构化 IRBuilder 定义，无需为本轮新建通用 Weft Python module importer；其展开须进入当前宿主程序。
- 采用单个普通 Native change：共享阶段、实现接口、bindings 与 Mojo/Weft 集成会共同修改同一组核心边界，不按 provider 拆分 Supervisor。
- 关键参考：TileLang `tileop/gemm/__init__.py:121–139` 分开 infer_layout/lower，`cuda/op/gemm/gemm_mma.py:109–142` 生成可编程 PrimFunc，`src/op/gemm.cc:198–236` 插回 IR；当前 CPU `BlockContractions.cpp:69–144` 提前固定向量微 panel，`MaterializeRegisterContractions.cpp:51–145` 是单一 f32 实现，Weft `Legalize.cpp:566–588` 尚无专业微程序实例化接口。
- 新增合同的事实依据：`doc/dsl/core.md:331–358` 的 microscaling schema 不能表达 Q4_K；CPU `CPUDialect.cpp:24–35` 与 `KIRToCPU.cpp:39–42` 目前拒绝 u8 views，Weft `Legalize.cpp:113` 固定 dense.f32，因此需贯通 typed carrier，而非只新增一个 leaf 名称。TianchenRV `examples/formats/ggml.py:88–97,276–283` 定义两种 records，`examples/kernels/vec_dot/q4_k_q8_k.py:8–29` 提供整数统计与 scale/min 校正的专业实现证据；它不替代本项目的正式数值规范。

# 待解决问题

- [blocking] CONFIRM: 是否确认本轮完整 Shape：保留 structured CPU task/block 并重构实现选择、需求协调和真实 IR 展开；保持现有 Mojo 五项 native 能力与有效 tuning；按 Spec §5 新增闭合量化点积/格式合同，以预量化 Q4_K×Q8_K 经正式链路生成 Canonical Weft IR；只运行受影响的既有 Mojo 性能 benchmark 并同次检查原容差；不包含量化准备、persistent repack、Weft 设备执行、其它后端或外部 TianchenRV 修改，沿用 main/current？确认后进入 Build。

# 验证预期

仅使用既有生产入口。Q4_K×Q8_K 的生成交付通过 `intent.generate(..., target=WeftTarget(...))`，不造设备 runtime 或虚假 benchmark registry 条目。既有 f32 五项保持可达，仅受实际改动影响的项通过 `examples/repro/v2` 做 benchmark；测 native 算子时间，包含 invocation 内 packing/任务同步，排除编译、tuning、加载与外部输出分配。沿用各项现有 tolerance，不新增数值/边界/回归/组合测试。Weft generation 不冒称数值运行或性能达标；本轮不新增统一 G/S 硬门槛。
