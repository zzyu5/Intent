# CPU 区域程序与跨执行模型语义优化

## 1. 编译边界

Canonical KIR 继续定义完整作者算法。CPU 从 KIR 构造独立的非 SIMT structured program，保留 task/workset、数据块、计算、访问、状态和资源；GPU 保留其 program/fragment 模型。本能力不新建横跨 execution families 的第二份可执行 planning IR，不复制 GPU topology，不让 provider/runtime 根据 KIR、kernel 名称或 source 重新构造算法。

本轮新增区域计算以 f32 数值数据及所需 bool/index/integer 坐标和状态为基础，沿用 DSL 中已定义的 region fold/scan、ordinary control、record/tuple 与访问语义。不增加作者 hint、provider 选择或新的算法操作。既有 f32 和 Q4_K/Q8_K 路径继续可达，不宣称其它 dtype/format 已全面支持。

## 2. CPU 区域与状态程序

### 2.1 结构化语义

CPU current program 独立保存 source-axis identity、绝对坐标、同步切片的 source components、captures、summary/transition schema、identity、combine、ordered control、访问有效域与 effects。Typed records 和 shaped fields 可用于摘要与状态，不能只留下不透明 key 给后续 emitter 解释。

Region fold 按原 source 顺序覆盖连续非空 slices，所有 source components 采用同一边界；summarize 内的 MatMul、统计和 pure helper 仍为明确的计算。Combine 保持 DSL 允许的顺序及重结合，空 source 返回原 identity。物理分段边界不变成作者可观察的 chunk ABI，不重新编号绝对坐标。

Region scan 另外保留 transition composition、initial state、apply、emit、完整输出和 final state。Emit 的 output slice 必须与 source member relation 对应，incoming state 按原 action 语义传递；空 source 的输出与 state 遵循 DSL。不得用 region scan 替代不可重结合的 effectful recurrence。其必需的 ordinary scan/ordered control 通过已有语义实现，不新增算法替代路径。

### 2.2 一致的分块与访问

Task 拥有工作范围与完成责任，可遍历多个 data tiles；data tile 用于工作集与复用，implementation microtile 与 SIMD width 是其内部组织。选定外围分块沿 current IR 中的轴、source 和结果关系传播，同时形成相连 MatMul/统计的局部 shapes、输入切片、谓词、遍历步长、尾部、carry 和输出责任。

不能逐个计算孤立选块而丢失它们之间的成员关系。例如同一 source axis 在前一个 MatMul 中是输出列、在后一个 MatMul 中是 reduction axis，访问和中间值必须保持一致。边界 validity/fill 和 source coordinates 由正式 IR 表达，不由 serializer 猜测。

当前程序可用标准 MLIR carriers 和必要的 CPU structured ops。选择 carrier 依据是否完整保存上述事实及是否有后续消费者，不以新增 dialect/op 的数量作为完成条件。保持结构化信息到相关分析与协调完成，再展开成所选 provider 可表达的程序。

## 3. 跨执行模型的分析与特化

### 3.1 复用边界

GPU 与 CPU 必须有实际共用的区域、identity 和状态性质分析逻辑，通过各自 current program 的 typed facts 提供输入。共同规则不依赖 GPU ProgramId、FragmentType、provider 名称或某个 kernel 的形状树；family adapter 可以读取本 family 的 types/ops，把范围、轴和数值/效果事实供给分析。

结论由各自 transformation 写回当前程序的循环、访问、状态和资源。仅复制相同思想、搬文件、增加空查询接口或保留执行旁表不构成复用。IR 修改后相关分析失效/重算；不可把失效前的 KIR adjacency 或旧 shape 当作当前执行关系。

### 3.2 有效遍历

在已知当前工作范围时，规则从 typed coordinate predicate 推导可证明的 all-true、mixed、all-false 范围。只有完整摘要在无效区域等于 identity，且不存在必须保留的 effects，才可跳过对应计算与专属访问。All-true 部分可简化谓词，mixed 部分保留原条件；证明不足保留原程序，不暗中缩小输入域。

报告中 query `[64,96)`、key `[0,160)`、区域大小 32 的例子仅解释规则，不固定实际候选或宣称加速比。其它绑定可能产生不同范围；编译决定来自当前关系和证明，而非 causal/attention 名称。

### 3.3 状态特化与数值条件

只有逐输出成立的首段非空事实、identity、summary validity 与 combine 关系足以证明后续状态不变量时，才从 physical carry 中去掉冗余 validity 或字段。首次迭代、空遍历、后续 part 的有效性、原 combine 和最终外部结果必须完整保留；不能因首段整体非空就假定每个 free lane 非空。

Identity/常量传播遵循 DSL 的整数和浮点语义。普通浮点乘法、contraction 或 cast 不因一个操作数为零就无条件视作精确零；需要可证明的数值条件，并保存 NaN/Inf、signed zero、dtype、approximation 和舍入边界。浮点有限性没有明确来源时不得假定成立，也不得启用全局 fast-math 或要求作者更改算法来配合优化。已有 GPU 规则中的相应问题在共享化时修正，合法已有优化继续可达。

## 4. 实现需求、外围复用与候选

延续既有 implementation registry、binding 与真实展开。实现对当前计算给出适用性、参数合法域，以及确需外围协调的块尺寸/输入表示/数据供应/资源需求；只添加实际使用的需求，不预建任意格式、设备或全局求解器。

尺寸合法性过滤本身是有效约束，但不能代替 input representation、共享准备与 lifetime 的真实集成。共同或 provider 协调 passes 使用同一候选的需求，形成必要的准备、输入绑定、资源 owner、初始化、消费者和释放点，再展开该候选；需求规划和展开不得重新选择不同实现或各自绑定不同参数。

一个实现内部的微块、decode、partial、packing 与私有 scratch 可继续由该实现形成。跨计算节点或任务共享的准备结果及其 lifetime 由外围程序协调，不能由每个消费者重复完整准备，也不能隐藏 persistent cache、跨调用 repack 或 caller-visible workspace。已由作者明确表达的共享 producer 必须保持，但不能把保持这一关系宣传成已经实现任意全局复用推断。

有限候选中的 task grain、外围 tile 和局部参数必须有实际消费者；输入/输出 tile、访问、carry 与资源一致。工作集按实际 dtype、表示及 lifetime 估计，不把逻辑数据量直接相加当硬件存储。Native 选优使用完整算子调用时间，不用编译或部署时间代替。

## 5. Mojo、Weft 与 native 调用

Mojo 与 Weft 在共同 CPU region/task 程序边界分流。Mojo 可进一步形成寄存器微块、SIMD、循环和 native ABI；Weft 可保留较高层 structured compute 并生成 Canonical Weft IR。不能为让公共 pass 可调用而强迫 Weft 先经过 Mojo 的 scalar/vector materialization。

专业实现展开必须成为 current IR，继续参与合法化、分析和验证。Serializer 只拼写已经形成的程序；runtime 只绑定 typed ABI 和已声明资源，执行完整 native entry，并在返回前完成任务与依赖。内部函数/任务不变成额外作者可见 kernel launches，不执行 Python/Torch 算法替代生成物。

原有 FP environment、typed buffer、alias/alignment、target capability、artifact 与 winner 分离等合同保持。外部 compiler 未支持的形式必须按其实际边界诊断；不能通过整算子 reference 调用、source template 或异常 fallback 假装完成。用户已授权在 TianchenRV 补齐本轮所需逐元素 select、区分浮点 maximum 语义，以及动态私有状态的有界窗口读写、extent 绑定和相应 physical IR/pass。窗口更新保留未写区域及循环状态，采用通用 Weft 能力，不引入 Intent 专用路径；外部修改独立提交并保留并发修改，其它外部仓库仍保持只读。

## 6. 性能与交付结果

正式性能运行复用现有生产 benchmark、native artifact 和 provider/source 目录。用少量代表性程序覆盖：包含 MatMul/统计/summary 的区域 fold，以及包含 transition/state/输出的 region scan。复用现有作者算法，必要 f32 实例化不改变计算语义；不以只运行普通 GEMM 或已有量化投影代替区域程序的运行证据。

Mojo 与 Weft 均需有上述新增区域能力的 native 性能证据。Source 是独立的同算法 provider-native 参考及完整 callable closure，可复用上游原语和现有参考；必要时在 `source/` 的既有语言/来源/职责边界补组合参考，不调用 Intent 生成代码作为 source，也不在 Python runtime 中实施参考算法。Source 缺失时如实处理，不能从另一硬件或旧吞吐量折算性能。

成对比较保持 shape、外部 dtype、任务/线程预算及明确计时范围。计入本次调用所需准备、packing/materialization、任务与 join，排除编译、tuning、部署、加载、预热及外部输出分配；生成与 source 的 workspace 范围一致并注明。真实 generated/source ms、G/S 与失败说明进入 `report/baselinev2/` 的 provider CSV，不进入稳定设计文档。

同次 benchmark 使用已有输入做一次既定容差检查，容差内即可；没有现成同类容差时不得为了通过随意设宽，须按实际参考的数值合同处理。既有 tolerance 不变。只运行得到所需性能结果的编译、预热与计时，不新增独立数值、边界、回归、兼容或压力测试，不用 `/tmp` 脚本绕过限制。

共享 GPU 规则改变时，仅复用受影响的原生产性能项；既有 CPU 能力同理。未受影响结果保留，准备可在资源预算内并发，计时避开争用。本轮不新增统一 G/S 门槛；结构变化、编译成功、运行正确与性能结果分别如实说明，显著慢路径不能冒充高性能能力。

验收只采用 brief 的 A1–A4，内部问题和文件位置不拆成额外 Scenario。只读复核对照实际 ref/triton 或 ref/tilelang 的同类代码给出具体差异与后果。

## 7. 工作区与非目标

本能力在独立 `comet/cpu-region-programs` worktree 实现，以 main 为目标分支。TritonBench agent 实验由其已有 change 继续，互不提交对方文件，不在本 change 改论文图稿。

不实现 DSA/Ascend、全量 dtype/量化格式、AMX/IME 微核库，不重写 GPU execution family，不新增全局 planning IR、模板选择器或额外测试框架。绘图报告中的尺寸、数据量和未来分支只按其说明用于设计推理，不自动变成固定输入、硬件假设或当前支持声明。
