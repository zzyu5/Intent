# 目标

按 `doc/compiler/cpu-program-ir.md` 重构 CPU 编译分层：保留 task/block 与 structured compute 的共同程序，由正式 lowering 查询目标实现需求、选择并实例化专家编写的微程序。收束当前 f32 向量偏好过早进入 shared blocking、配置职责混合，以及 Mojo/Weft 缺少可编程实现接口的问题；不是取消 CPU IR、整算子库旁路或只做接口搬家。

# 范围

- 保留 canonical KIR→CPU 的唯一 construction、typed ABI、任务/坐标、effects、共同 fusion/reuse/lifetime 与 verifier；共同程序不无条件展开为 Mojo SIMD，也不复制 GPU warp/lane 模型。
- 将外层 task/cache blocking 与实现内部 microtile、packing、vector/replica 等需求分开；需求协调、布局/供应和展开消费同一个候选选择。
- 建立有实际消费者的 implementation 适用性、需求、有限参数与结构化展开机制，现有 f32 Mojo contraction/vector 路径迁入同一正式编译链，保留 native artifact 与现有五项 f32 registry 的可达能力。
- Weft 继续以 Canonical Weft IR 为输入；补专业微程序所需的结构化实例化/集成边界，保留简单操作直接转换，不在 emitter 拼接整段算法。
- 用实际专业计算的生产者/消费者组合证明机制不只包装原 f32 builder：调用内 Q8_K activation 量化准备、Q4_K×Q8_K 局部点积和跨输出复用都进入正式 CPU/provider 程序。原 generation-only、排除量化准备和只接外部 records 的裁剪已被替代。
- 新增闭合的 `I.quantize(..., format=I.quant.q8_k)`、`I.quantized_dot` 与 `I.quant.q4_k/q8_k` 格式 schema；完整输入/输出、布局、整数统计和浮点合同见 Spec §5。不新增可编程 Encoding 类型或 opaque kernel handle。确认后先补 `doc/dsl/` 的最小规范，再迁移实现。
- 接入 Weft CLI artifact→system compile→native load/run，使用 typed buffer ABI，不要求 RVV 安装 Torch；量化中间值的资源、CPU tasks、scalars 和 join 不能丢失。生成与 native materialization 使用同一编译链，SSH 仅用于显式部署/benchmark。
- 既有 Mojo 五项保持可达，只重跑受影响 benchmark；Weft 首个实际性能交付选 SG2044 标准 RVV 上 `N=K=4096`、单 f32 activation vector 的 Q4_K 量化投影，包含调用内 Q8_K 准备。复用生产入口，同次原容差检查，结果进入项目 CSV，不为 Shape 重跑性能。

## Source coverage

当前 CPU doc 与已提交报告已完整读取。下表覆盖相应文档全部章节；`covered` 表示已映射到本草案，不表示 Shape 已获最终确认。参考链接仅作为实现证据，不把其全部上游能力扩成需求。

| 来源单元 | 读取 | 归类与保留内容 | Spec 对应 | 验收 | 覆盖状态 |
|---|---|---|---|---|---|
| CPU doc §1–2 | complete | execution family、task/block、完整 structured IR、唯一程序 | §1–2 | A1 | covered |
| CPU doc §3 | complete | 作者、共同 passes、implementation、外部 compiler 的职责 | §2–3、§5.6 | A1–A3 | covered |
| CPU doc §4 | complete | 适用性、需求、参数、选择/展开与程序集成 | §3、§5.4–5.6 | A2、A3 | covered |
| CPU doc §5 | complete | ABI/effects/lifetime 与格式、数值义务 | §2、§4–5、§7.1 | A1、A3、A4 | covered |
| CPU doc §6 | complete | 能力与共享/实现/外部参数分工，有限选优、cache | §3、§6 | A2、A5 | covered |
| CPU doc §7–8 | complete | Mojo/Weft 停止层次、verifier、native 与 generation 边界 | §4、§6–7 | A2、A4、A5 | covered |
| 报告导言、§1 | complete | 已确认架构方向；报告本身非设计权威 | §1–3 | A1、A2 | covered |
| 报告 §2、§3（含表格与链路） | complete | 当前实现与 TileLang 证据背景；不移植完整上游能力 | §1、§3 | — | background |
| 报告 §4–6（含程序示意与职责表） | complete | structured CPU 模型、作用范围与可编程实现接口 | §2–3 | A1、A2 | covered |
| 报告 §7（含量化准备例子） | complete | 格式语义与实现分开；Q8_K 准备、Q4_K×Q8_K 点积与共享资源均显式可组合 | §5 | A3 | covered |
| 报告 §8 | complete | 复用 Mojo 现有微核；Weft helper/Canonical adapter 与外部 compiler 边界 | §4–5、§7.1 | A2–A4 | covered |
| 报告 §9（含模块表、配置与唯一迁移路径） | complete | 分层迁移范围与实际参数消费者，不只重命名 | §2–4、§6 | A1、A2、A5 | covered |
| 报告 §10 | complete | CPU 架构保持；新增量化入口、布局和数值合同先经最终 Shape 确认，再进入正式 DSL 规范 | §1、§5 | A1、A3 | covered |
| 报告 §11 | complete | 真实实现闭合、生产 benchmark；量化组合贯通 Weft native，Mojo 既有能力保持 | §5–7 | A2–A5 | covered |
| 用户 Q1 回答“可以” | complete | 保留 Q4_K×Q8_K 专业计算块；generation-only、排除调用内量化被最新授权替代 | §5 | A3 | superseded：最新量化准备/runtime 授权 |
| 用户“按照你的理解继续吧，weft可以接一下runtime之类的啊” | complete | 调用内 Q8_K 准备、复用、点积组合与 Weft AOT native artifact/runtime；首个 SG2044 实际性能交付 | §5–7 | A3、A4 | covered |

来源：`doc/compiler/cpu-program-ir.md`、`report/cpu-programmable-lowering-reassessment.md`；同层 compiler/DSL 文档与 `AGENTS.md` 继续约束本 change。

# 非目标

- 不重构 GPU、不进入 DSA/Ascend，不自动扩展所有 dtype、量化格式或输入矩阵。
- 不把 Weft Encoding/Level 整套加入 Intent public surface，不根据 kernel 名选整算子模板，不新增 runtime 算法替代路径。
- 不隐式启用 fast-math/FTZ，不以放大容差换通过；不新增性能 benchmark 之外的测试。
- 不扩展跨调用权重预处理、persistent repack、全量量化库或无关设备平台；不强制双设备全量、IME 或在板卡安装完整 compiler/frontend 后才能验收。Weft runtime 接入已获授权；外部 TianchenRV 先保持只读，确有必需缺口时拿出证据、取得最小修改授权并独立提交，不覆盖他人工作。
- 不新增 worktree，不推送、不创建 PR；未受影响的历史性能保留，不默认全量重跑。

# 验收示例

- A1：现有 CPU 程序保留完整 task/block、访问、依赖与 structured compute；共同 blocking/参数不再无条件绑定特定 f32 向量微核的内部组织，Mojo/Weft 从同一共同程序边界继续 lowering。
- A2：实际计算块通过明确的实现适用性、需求协调、参数绑定与微程序展开形成可验证 IR；现有 f32 Mojo native 路径在新机制中真实可调用，布局/供应与展开使用同一选择，无新旧旁路并存。
- A3：真实 Intent 作者 kernel 从 f32 activation 产生 Q8_K 中间值并让多个 Q4_K×Q8_K 点积消费；沿唯一 KIR→CPU→Weft 链形成量化/点积专业程序的真实展开。§5 的格式、i32 统计、f32 校正和 reduction 合同完整保留；producer/captures/轴/域、消费者、effects、共享 storage 与 lifetime 显式连接。量化不在每个点积里重复，不残留 emitter/runtime recipe，不调用完整 source kernel。
- A4：Weft native artifact 可通过 typed buffer ABI 加载并执行完整单次 CPU invocation，任务、scalar、内部资源和 join 均闭合；SG2044 的固定量化投影 case 获得真实 generated/source ms 和 G/S，包含 activation quantization，并在同次 benchmark 通过 `1e-4 + 2e-3*abs(expected)` 容差，结果进入 `report/baselinev2/weft-rvv.csv`。不是仅生成 C、手动运行 reference 或把 SSH/编译耗时当算子时间。
- A5：shared/implementation/provider 参数都有实际消费者，有限合法候选以相应目标上的完整 invocation 时间选优并复用 artifact/winner。既有 Mojo f32 affine、RMSNorm、dense GEMM、Softmax、LayerNorm 保持可达；受影响项使用原生产 benchmark、原算法/source/容差，实际 generated/source 时间和 G/S 更新到既有 CSV。

# 约束与不变量

- `doc/` 为设计权威；CPU 分层、内部资源和调用边界按既有规范收束。新增语言设计仅为待最终确认的 §5 闭合量化准备、点积及格式/数值合同，确认后先补对应 DSL 规范；不为了保留旧实现改写其它语义。报告提供已读证据与范围线索，旧归档的非目标、失败状态和性能门槛不自动变成本轮约束。
- 专家 implementation 可编程，展开必须成为 current IR；serializer 只拼写。合法直接映射与专业实现按适用条件共存，不以失败后 fallback 决定程序。
- 格式/算法选择、实现选择、参数选择分别归属。Q4_K×Q8_K 尚无对应 DSL/KIR 数值合同，先明确该最小合同，不以一个 format 字符串冒充完成。
- 外围共享量化准备、数据供应与资源可见；微程序不能每个 output tile 重复 activation quantization，不能隐藏 persistent repack/缓存。内部物化不强迫上移到作者 public ABI；下层 View 参数与本次 invocation 的 storage owner/lifetime 显式对应。
- 自查对照 current/spec/ref 的 file:line、具体差异与后果；不增加独立测试门禁。

# 决策

- 用户已确认将 CPU 模型和可编程 lowering 边界固定到 `doc/`，文档与报告已提交为 `b164302`。
- 用户已明确接受 `cpu-pointwise-reductions` 的 4/4 验收并授权归档；归档提交为 `44c422c`。本轮不再恢复该任务。
- 用户授权创建新 change，沿用 main/current；此授权不是本 Shape 的最终 Build 确认。
- 用户最新授权替代原 Q1 的范围裁剪：纳入调用内 Q8_K 量化准备与 Q4_K×Q8_K 点积组合，并接入 Weft runtime；不再把量化准备或设备执行一概排除。量化准备可以专业微程序化，其生产结果、共享范围与 lifetime 仍属于 current CPU/provider program。
- 最终 Shape 的接口提议：普通 u8 carrier 保持现有 ABI；closed format schema 定义固定 record mapping、`bsum` 一致性及量化点积数值语义。Q4_K/Q8_K 不是新 scalar dtype，也不借 `scaled_contract` 或 arbitrary `assume` 承载。具体专业程序可以用结构化 IRBuilder 定义，无需为本轮新建通用 Weft Python module importer；其展开须进入当前宿主程序。
- 采用单个普通 Native change：共享阶段、实现接口、bindings 与 Mojo/Weft 集成会共同修改同一组核心边界，不按 provider 拆分 Supervisor。
- 关键参考：TileLang `tileop/gemm/__init__.py:121–139` 分开 infer_layout/lower，`cuda/op/gemm/gemm_mma.py:109–142` 生成可编程 PrimFunc，`src/op/gemm.cc:198–236` 插回 IR；当前 CPU `BlockContractions.cpp:69–144` 提前固定向量微 panel，`MaterializeRegisterContractions.cpp:51–145` 是单一 f32 实现，Weft `Legalize.cpp:566–588` 尚无专业微程序实例化接口。
- 新增合同的事实依据：`doc/dsl/core.md:331–358` 的 microscaling schema 不能表达 Q4_K；CPU `CPUDialect.cpp:24–35` 与 `KIRToCPU.cpp:39–42` 目前拒绝 u8 views，Weft `Legalize.cpp:113` 固定 dense.f32，因此需贯通 typed carrier，而非只新增一个 leaf 名称。TianchenRV `examples/formats/ggml.py:88–97,276–283` 定义两种 records，`examples/kernels/vec_dot/q4_k_q8_k.py:8–29` 提供整数统计与 scale/min 校正的专业实现证据；它不替代本项目的正式数值规范。
- 量化准备参考：TianchenRV `examples/kernels/quantize/q8_k.py:8–30` 已有可编程 Q8_K helper，`examples/kernels/mul_mat/q4_k.py:17–20` 先量化再复用；相较本项目原 Spec 的外部-record-only consumer，真实缺口在内部 producer/consumer 和资源集成，不是微程序不能执行量化。
- Runtime 参考：TianchenRV `tools/weft-compile/weft-compile.cpp:109–163,317–320` 可直接编译 Canonical IR 并输出完整 artifact；其 Python runtime `compiler.py:76–111,196–227` 依赖 KernelDefinition、单 kernel 和 native host。Intent `targets/weft.py:6–30` 目前 source-only，`runtime/artifact.py:94–107` 无条件导入 Torch；需补 Intent 的 artifact/ABI/native 调用适配，不因 Python frontend API 限制就改外部 compiler。Weft 默认 `compiler.py:223` 的 contraction flags 不可反向放松 Intent 数值合同。
- 已核实的执行缺口：Intent `lib/Target/Weft/Transforms/Legalize.cpp:55–69` 把 task coordinate 生成为 scalar 参数，并明确拒绝 captured scratch；外部 `lib/Target/Emission/Emitter.h:308–315` 的 native ABI 要求 typed memory 参数。因此不能只在现有生成物后接一个 loader：必须完成 scalar/task ABI 与内部资源 lowering，且不能把 coordinate 偷换为 shape 或把临时区强塞给作者。
- 当前可用性：SG2044/`rvv` 与 K1 SSH 均可达、原 Clang18 与 GGML 目录存在；两端未安装 Torch/Weft/Intent Python 包。先以 SG2044、标准 RVV、单核 `N=K=4096` decode shape 做真实 runtime/性能交付，复用既有显式部署方式；不把完整板卡端 JIT 安装或双设备全量设为前置条件。配置的 VLEN/ISA 不是已完成的 native 运行证明，Build 中按实际执行能力兑现。

# 待解决问题

- [blocking] CONFIRM: 是否确认更新后的完整 Shape：保留 structured CPU task/block，重构实现选择/需求协调/IR 展开并保持 Mojo 五项与有效 tuning；新增调用内 Q8_K 量化准备、Q4_K×Q8_K 点积及中间值复用；贯通 Weft AOT native artifact、typed buffer load/run，在 SG2044 固定一个含量化准备的投影 case 测真实性能和原容差；不强制双设备全量、IME、板卡完整 JIT 安装或跨调用 repack，外部 TianchenRV 先只读，沿用 main/current？确认后进入 Build。

# 验证预期

仅使用生产编译/运行入口及所需的 Weft provider 接线，不新增独立数值/边界/回归/组合测试或临时脚手架。Mojo 只重跑受影响既有项、沿用原容差；Weft 固定一个 §7.2 量化投影性能 case，以同算法 source 配对，同次一次 f32 输出容差检查，不引入 standalone quantize 的 byte-exact 门禁。Native 算子时间包含每次量化准备、packing/materialization、任务执行与同步；编译、tuning、部署、加载与外部输出分配不计时。结果只更新项目正式 CSV；本轮不新增统一 G/S 硬门槛。
