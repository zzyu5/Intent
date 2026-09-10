# CPU 可编程目标 lowering

## 1. 目标能力与权威

本能力实现 `doc/compiler/cpu-program-ir.md` 定义的 structured CPU program 与可编程目标 realization。CPU 仍从 canonical KIR 独立构造，形成非 SIMT 的 task/block 程序；不改为整算子库选择器，不复制 GPU execution topology，也不另起 planning IR。

当前 brief 的 Q1 尚未解决，本 Spec 是 Shape 草案；它不授权进入 Build，也未定义或启用任何新的量化格式。

## 2. 共同程序与分层

共同 CPU 程序保存 typed ABI、logical axes、task worksets/captures、访问与有效域、ordered control/carries、structured compute、共享值和资源 lifetime。CPU construction/analysis 不依赖 kernel 名、provider 源码或旁路 recipe。

外层 task/cache blocking、跨操作 fusion/reuse、输出 ownership 和资源协调由共同 passes 完成。目标实现可以提出 microtile、输入表示、数据供应、输出形式和资源需求；有关结构在需求查询后按一致的候选 binding 形成，不能无条件先按 f32 SIMD 宽度固定。

现有标准 dialect 可继续作为 carrier。Dense contraction 识别不是所有数值计算的语义定义；新增格式或累加类型必须完整穿过 construction、structured-operation 接口及 verifier。仅改 operation 名或登记字符串不算支持。

## 3. 专业 implementation 的编译接口

每个实现明确承接的计算语义、operand/result relation、适用条件、外围需求、有限参数和真实 IR 展开逻辑。IRBuilder、结构化宏或目标 DSL helper 都可作为实现定义；普通 Intent 作者不必复述其内部实现树。

编译顺序为：从 current IR 查询适用实现→协调 block/layout/supply/resource 需求→固定实现及参数→展开微程序→连接 effects/lifetime→继续合法化与验证。需求规划与展开必须消费同一选择。

专业微程序可含局部循环、decode、partial、转换及 scratch，仍须保持承接计算的数值合同与外围依赖。展开结果成为 current program，后续 analysis 失效/重算，verifier 只检查；serializer 不恢复算法或重选实现。

简单操作的直接映射是明确、合法的 lowering，不是专业实现失败后的隐式退路。迁移后的能力只保留一条正式执行链，不维护旧 pipeline 开关。

## 4. Mojo 与 Weft

Mojo：现有 f32 register contraction/vector 程序作为明确的实现迁入新机制，保留 native ABI、FP environment、任务 join、数值义务与已有可达算法；具体 vector/unroll/prefetch 策略及参数由实际实现消费，不成为整个 CPU family 的固定定义。

Weft：消费 structured CPU task，而非先经过 Mojo 的 SIMD materialization。简单操作可直接生成 Canonical Weft IR；专业实现通过结构化 adapter 绑定宿主 views/scalars、axes/symbols、domain、输出和资源，不能直接拼接 standalone kernel body。

Weft 现有 helper/front-end 及 Canonical dialect 可复用；Intent 不复制整个 Weft parser/语言或 RVV/IME 指令选择器。外部 repository 默认只读，需要修改其接口时另行取得明确授权。

## 5. 首个专业计算块：待 Q1 确认

推荐候选为预量化 Q4_K weights 与 Q8_K activation 的局部点积。推荐边界只承接已有量化输入的计算，不包含 f32 activation 的量化准备、跨调用 weight interleave、persistent repack 或设备运行。

这项建议不把 Q4_K 当作现有 FP4 microscaling schema。若采用，应先明确其最小硬件无关格式与数值合同，再让 KIR/CPU 保存选择及组合实现所需的信息；目标微程序可封装具体 Encoding、Level、整数统计和 scale/min correction。完整公开接口、数据布局及允许的数值 realization 必须在最终 Shape 前确定。

选中的实现必须从真正的 Intent 作者入口经过 canonical KIR/CPU 到达目标程序，不允许绕过上层格式语义，直接拿 Weft example kernel 或 reference runtime 充当生成路径。

Q1 未确认前不修改 DSL 数值规格、不新增算子格式、也不执行上述建议。若用户选择其它专业计算块，本节与 A3 一并改为确认后的完整范围。

## 6. 参数、artifact 与性能

共同参数约束 task grain、外层 block 和跨块组织；implementation 参数约束真实局部微块/vector/replica/unroll 等；外部 compiler 参数由其消费者拥有。跨边界参数只有一个 binding owner 与显式约束，不各自选择两个值。

保留有限配置覆盖、适用性过滤、完整候选实例化和实测 winner；不对所有格式/算法建立笛卡尔积。Artifact 与 winner cache 分开，其身份包含实际依赖的 specialization、view facts、target、implementation 定义及 bindings。

既有 Mojo registry 的 f32 affine、RMSNorm、dense GEMM、Softmax、LayerNorm 保持可达；不改变它们的作者算法、同算法 source 及既定数值容差。受影响项复用现有生产 benchmark，在同次运行做一次容差检查，记录 generated/source native 时间、G/S 和必要说明到 `report/baselinev2/mojo-x86.csv`。未受影响结果直接复用，不要求重跑全表。

不以候选搜索、JIT 或 Python 调度耗时冒充算子时间；同机准备允许预算内并发，计时避免互相干扰。Weft 仅生成的项不填写虚构设备性能，不把 compilation verifier 当作数值运行。

## 7. 验收与非目标

验收仅使用 brief 的 A1–A4：共同 CPU 边界、真实可编程实现、选定专业计算块、有效参数/native 性能。技术定位与参考材料不拆成独立测试门禁。

不扩展无关 GPU/DSA，不接全量量化库，不引入隐藏 host invocations 或跨调用 workspace，不启用全局 fast-math/FTZ，不放大容差，不增加性能 benchmark 之外的测试。本 change 使用 main/current；Shape 完整确认后才进入 Build。
