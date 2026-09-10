# CPU 可编程目标 lowering

## 1. 目标能力与权威

本能力实现 `doc/compiler/cpu-program-ir.md` 定义的 structured CPU program 与可编程目标 realization。CPU 仍从 canonical KIR 独立构造，形成非 SIMT 的 task/block 程序；不改为整算子库选择器，不复制 GPU execution topology，也不另起 planning IR。

用户已确认完整 Shape 并授权进入实现，范围为调用内 Q8_K 量化准备、Q4_K×Q8_K 点积组合与 Weft runtime 接入。原 generation-only 裁剪已被替代；完整范围见 §5–7。先将最小语言扩展写入 `doc/dsl/`，其它设计继续按现有 `doc/` 实现。

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

Weft：消费 structured CPU task，而非先经过 Mojo 的 SIMD materialization。简单操作可直接生成 Canonical Weft IR；专业实现通过结构化 adapter 绑定宿主 views/scalars、axes/symbols、domain、输出和资源，不能直接拼接 standalone kernel body。Canonical IR 继续经过外部 Weft physical compiler、system compiler 和 §7 的 native artifact/runtime；不在 Intent 重写 RVV/IME leaf selector。

Weft 现有 helper/front-end 及 Canonical dialect 可复用；Intent 不复制整个 Weft parser/语言或 RVV/IME 指令选择器。外部 repository 默认只读，需要修改其接口时另行取得明确授权。

## 5. 专业计算组合：Q8_K 量化准备与 Q4_K×Q8_K 点积

保留预量化 Q4_K weights 与 Q8_K activation 的局部点积，并增加 f32 activation→Q8_K 的调用内准备。量化与点积各自拥有完整数值合同，可由专业微程序实现，结果在同一 CPU invocation 内形成真实 producer/consumer 和复用关系。跨调用 weight quantization、interleave/persistent repack 不在本轮范围。

### 5.1 最小作者入口与 canonical relation

```python
value = I.quantized_dot(
    w_records, x_records,
    lhs_format=I.quant.q4_k,
    rhs_format=I.quant.q8_k,
    acc_dtype=I.f32,
)
```

`w_records`、`x_records` 是已有 u8 tensor values，shape 分别为 `[G,144]`、`[G,292]`；可由普通 external-view indexed reads 取得，不新增 public encoded-view element type。`G` 必须相等，支持静态或共享 identity 的动态 record extent；logical 点积长度是 `256*G`。只接受完整 records，不隐式补齐 partial record。结果为 rank-0 f32 tensor；不隐含 batch、矩阵轴、alpha/beta 或 mutable accumulator，外围遍历与结果存储继续使用普通作者 constructs。

`I.quant.q4_k/q8_k` 是语言定义的 closed typed format schemas，不是 numeric dtypes、provider 字符串、实现 key 或任意可执行回调。首个支持的组合仅为上述 lhs/rhs 与 f32 result。前端归一到一个独立的 canonical quantized-dot operation，完整保存 operand/result relation、format 与数值合同；不复用 FP4 microscaling 的 `scaled_contract`，不把 bytes 误解释为 ordinary integer dot，也不从通用 decode 树识别整算子。

既有普通 packed INT4/INT2 的显式 decode→ordinary contract 路径不变；本扩展仅增加以下完整定义的局部 structured operation，不意味着任意 encoding 或量化算法均有隐含语义。

### 5.2 固定存储解释

两个 schema 都使用 little-endian bytes、低位优先 bit numbering，每条 record 对应 256 个 logical elements。以下 offset 均相对当前 record，所有 bytes 都有定义，不增加隐式 padding。

| Schema | 字段 | Byte span / logical type |
|---|---|---|
| Q4_K，144 bytes | `d`、`dmin` | `[0,2)`、`[2,4)`，各一个 IEEE f16 |
| Q4_K | `sc[8]`、`m[8]` | 共用 `[4,16)`，各元素为 unsigned 6-bit |
| Q4_K | `q[256]` | `[16,144)`，unsigned 4-bit，按以下 nibble 公式 |
| Q8_K，292 bytes | `ds` | `[0,4)`，一个 IEEE f32 |
| Q8_K | `q[256]` | `[4,260)`，连续 two's-complement i8 |
| Q8_K | `bsum[16]` | `[260,292)`，连续 little-endian two's-complement i16 |

令 `a` 为 Q4_K record 的 u8 bytes。对 role `r=0(sc)` 或 `1(m)`：

- `i=0..3`：`field_r[i] = a[4+4*r+i] & 63`；
- `i=4+t, t=0..3`：`field_r[i] = ((a[12+t] >> (4*r)) & 15) | ((a[4+4*r+t] >> 6) << 4)`。

对 `i=0..255`，`q[i]` 读取 `a[16+32*(i//64)+(i%32)]`；`i%64<32` 取低 nibble，否则取高 nibble。不是相邻两个 logical elements 简单共用一个 byte。

Q8_K 的格式合法性另包含 `bsum[j] = sum(q[16*j:16*j+16])`。这是预量化输入的调用方数据合同，不是由 Weft Encoding 自动赋予的不变量，也不是优化 hint。Compiler 不插入一次量化、repack 或全数据扫描来修补不合法输入；实现可以直接消费该统计字段。Shape/dtype/format 关系静态检查，动态 shape equality 与数据有效性按明确调用合同处理。

### 5.3 数值合同

对每条 record `b`，先在数学整数域定义两个统计量；以下所有乘法之前按各字段 signedness 解码并提升，不在 u4/i8 中截断乘积：

```text
S_b = sum(s=0..7, sc_b[s] * sum(i=0..31, wq_b[32*s+i] * xq_b[32*s+i]))
T_b = sum(s=0..7, m_b[s] * (bsum_b[2*s] + bsum_b[2*s+1]))
```

合法 records 下 `abs(S_b) <= 30,965,760`、`abs(T_b) <= 2,064,384`，i32 可精确承载，因此基准 realization 使用 i32 统计；可使用满足相同精确结果的分组/更窄局部 partial，但不能以窄乘法或饱和丢失结果，不要求通用 compiler 推导专家完整数值树。

令 `f32(v)` 为 round-to-nearest-even 转换，`R32` 表示一次 f32 rounding：

```text
p_b = R32(f32(d_b)    * f32(S_b))
n_b = R32(f32(dmin_b) * f32(T_b))
c_b = R32(p_b - n_b)
v_b = R32(ds_b * c_b)
result = f32 add-reduction of v_b in increasing record order
```

记录间采用普通 structured reduction 的 logical-order-preserving parenthesization 自由，不允许 arbitrary permutation；专家 helper 的顺序累加是其中一种合法 realization。`G=0` 返回 f32 `+0`，不读取 records。保留上述 record-local 浮点运算与转换边界，不把它默认改写为逐元素 f32 解量化后 ordinary dot，也不隐式 FMA、FTZ 或全局 fast-math。NaN、infinity、signed zero、subnormal 和整数转 f32 遵循现有数值规范；不新增 finite-only 假设或自定义容差。

这里固定的是闭合计算的可观察数值合同，不是 RVV/IME 指令、寄存器 layout 或固定专家程序树。编译器选择满足合同的 realization，而不是根据 provider 临时改变算法。

### 5.4 CPU carrier、实现及程序集成

CPU construction/typed ABI 必须真实保存 f32 量化输入、u8 carrier、f32 点积结果、record extent identity、原访问/快照/effects 与 format relation；拆除共享层对所有输入均为 f32 的无条件假设，不因此宣称其它 dtype/operations 全部支持。CPU structured operations 保存完整计算，不提前展开 nibble/decode 或强制走 f32 matrix matcher。

专业实现通过 §3 的同一查询/绑定/展开接口接收该 operation。首个 Weft realization 可用结构化 IRBuilder 编写，产生真实 Encoding declarations、字段读取、integer dot/reduce、转换、校正、循环/carry 与结果 SSA；不要求外部仓库新增通用 callable-module importer，也不能只把原 f32 builder 改名。若使用目标 source helper，则必须经其 frontend 形成 IR，再显式绑定进入宿主，不做字符串替换。

Adapter 将已知 byte-carrier relation 映射到相应 encoded views/values，明确区分 record axis、256-element logical axis 和 byte axis，保持外部 bytes，不隐含 copy/repack。Canonical Weft 输出中的捕获、shape/axis symbols、domain 父子关系、返回值的宿主 consumer、store、alias/effects 和资源 lifetime 必须完整；共享宿主唯一 root/终止结构，不能直接拼接 standalone kernel body。合法空域不产生 encoded-record 读取。

首个基于现行 Weft formats 的 realization 同时消费连续外部 records 和本次 invocation 内量化 producer 生成的 records。Q4_K/Q8_K record 起址分别满足 2-byte/4-byte alignment，输入 pointer/offset/stride 与 provider View 的映射必须完整。外部 view 的适用条件由 native admission 兑现；内部物化的大小、对齐和 lifetime 由 compiler 显式形成，不能仅因 dtype 是 u8 就假定条件成立。其它没有对应 realization 的 carrier producers 明确诊断，不静默换格式或 host 算法。外部 Weft 规范示例的 alignment=1 不作为当前 helper 的 2/4-byte pin 合同证据。

完整路径为真实 Intent 作者 kernel→canonical KIR→structured CPU→implementation expansion→Canonical Weft IR→Weft compiler→native artifact。`intent.generate` 保留生成能力，native materialization/load/run 消费同一编译结果，不从 runtime 重建量化或点积程序。示例只是通用 lowering 的 consumer，不根据示例名称选择程序；Mojo/GPU 对这项新格式尚无实现时明确拒绝，不影响其已有合法能力。

### 5.5 调用内 Q8_K 量化操作

新增闭合入口 `I.quantize(values, format=I.quant.q8_k)`：输入 f32 tensor `[G,256]`，输出 u8 tensor `[G,292]`，保存同一 record axis identity；没有外部写入、隐藏 launch 或 caller-visible workspace。外部存储通过普通 view store 表达。该入口归一到独立 canonical quantize operation，使用明确 format/numerics，而非识别任意 min/max/cast 树；本轮不宣称其它 formats 已有量化实现。

每个 256-element block 独立计算，采用现行 Weft Q8_K helper 的数值方案：

1. 取得 block 的最大值 `maximum` 与最小值 `minimum`，选择绝对值较大的 signed extreme；绝对值相等时选 `minimum`。
2. 全零 block 生成 `ds=+0`、所有 `q=0` 和 `bsum=0`；非零 block 计算 `inverse=R32(-127.0/extreme)`，再计算 `ds=R32(1.0/inverse)`。
3. `q[i]` 为 `R32(values[i]*inverse)` 按 nearest-even 取整、饱和到 `[-128,127]` 的 i8；不是普通 float→integer 的向零截断 cast。
4. `bsum[j]` 是对应 16 个已经量化的 q 的精确整数和，保存为 i16；按 §5.2 的 Q8_K mapping 写出完整 records。

输入域要求有限 f32；非零 block 的 inverse、ds 及转换前 scaled values 必须具有上述操作可表示的有限结果。域外数据是量化操作的非法输入，不默默把 NaN/overflow 变成零或改变算法。编译器不插入全数据扫描作为运行前置条件；该限制不授权放大 benchmark 容差。`G=0` 产生 empty records，不读取输入。

### 5.6 生产者、复用与 native 组合

正式作者示例采用 Q4_K weights `[N,G,144]`、f32 activation `[G,256]` 和 f32 output `[N]`：先由 `I.quantize` 定义一个 Q8_K 中间值，再以普通 parallel/output control 调用局部 `I.quantized_dot`，让 N 个输出消费同一组量化结果。它是普通构造组成的量化投影，不新增 whole-kernel GEMV/matmul operation 或 kernel-name template；K 等于 `256*G`。

共同 CPU passes 保存 producer、消费者集合、跨 task 依赖和唯一写入；确定共享物化范围、分块供应和释放点。量化/点积 implementation 各自展开内部组织。单消费者可按合法性融合，多消费者场景保留量化结果复用；不能在每个输出点积里重新量化完整 activation。它不要求把所有输入一次物化为大 tensor，也不允许由 emitter 临时决定准备阶段。

量化结果可以保持内部 SSA/logical buffer，并由编译器形成 invocation-local storage；不强迫 Intent 作者把它变成外部参数。若下层 Weft callable 需要一个编码 View 参数，该 View 对应的内部 storage owner、大小、对齐、初始化、跨 task 使用与 lifetime 仍显式保存在当前编译程序。外部 source 示例的显式 Xq workspace 不是本项目 public ABI 的设计权威。

一个 Intent kernel 最终仍为一次 host-visible native invocation。内部 CPU tasks/本地函数不是额外独立 kernel launches；任务调用、传参、同步与资源组织必须由当前程序形成，runtime 只调用已编译入口，不在 Python 逐输出循环中实施算法。

## 6. 参数、artifact 与性能

共同参数约束 task grain、外层 block 和跨块组织；implementation 参数约束真实局部微块/vector/replica/unroll 等；外部 compiler 参数由其消费者拥有。跨边界参数只有一个 binding owner 与显式约束，不各自选择两个值。

保留有限配置覆盖、适用性过滤、完整候选实例化；Mojo native 继续实测选优并缓存 winner。Weft 只枚举已有真实 consumer 的少量合法参数，使用目标机的完整 invocation 时间选优，不在 x86 或 Python/SSH 耗时上决定 winner；单个合法专业实现不需要虚构第二份候选。格式/算法不作为可互换的 tuning 参数，不对所有格式/算法建立笛卡尔积。Artifact 与 winner cache 分开，其身份包含实际依赖的 specialization、view facts、target、implementation 定义及 bindings。

既有 Mojo registry 的 f32 affine、RMSNorm、dense GEMM、Softmax、LayerNorm 保持可达；不改变它们的作者算法、同算法 source 及既定数值容差。受影响项复用现有生产 benchmark，在同次运行做一次容差检查，记录 generated/source native 时间、G/S 和必要说明到 `report/baselinev2/mojo-x86.csv`。未受影响结果直接复用，不要求重跑全表。

不以候选搜索、JIT、Python 调度、SSH 或部署耗时冒充算子时间；同机准备允许预算内并发，计时避免互相干扰。Weft 运行与比较按 §7，仅生成的其它项不填写设备性能，不把 compilation verifier 当作数值运行。

## 7. Weft runtime 与首个性能交付

### 7.1 Native artifact 与调用接口

复用正式 `weft-compile --emit=artifact`：直接输入 Intent lowering 的 Canonical Weft IR，取得同次编译的 RISC-V Physical IR、intrinsic C 和 typed ABI，再由明确的 RISC-V system toolchain 形成可加载 native artifact。不能通过 source example/KernelDefinition 替换 Intent 生成的程序，也不需要为此复制 Weft frontend。

Intent 的 Weft materialization/runtime adapter 保留生成入口，并提供 native artifact 的加载、typed buffer 参数绑定、launch/run 与释放能力。它消费真实 compiler ABI 中的 symbol、encoding、shape、record span、alignment、access/alias、scalar/shape 参数与 target binding，不解析 C 文本补 ABI。CPU 调用不强制依赖 Torch；连续 Python buffer storage 与显式 shape/dtype 可用于 RVV native invocation，现有 Mojo/Torch CPU 与 GPU 的调用行为保持不变。

CPU task coordinates、scalar captures 和多 task 依赖不能被丢弃，也不能把 scalar 冒充 shape symbol。Provider ABI 不直接接收的形式通过正式 lowering 变成明确的内部传参/调用程序。下层 CLI 可包含多个 native helper symbols，但对外仍是 §5.6 的完整单次 CPU invocation；不把 Weft Python loader 的单 kernel 限制变成新 kernel ABI 的任意裁剪。

编译 profile 明确保存 march/ABI/VLEN 和实际使用的 extensions；加载/执行核对目标能力、vector state、执行 CPU 集合及 buffer 合同，不由设备名称推断能力。RNE/FP environment 与普通运算保持 Intent 语义，必要状态在调用后恢复；system compiler 不沿用会改变 §5 数值边界的默认 `-ffp-contract=fast`，采用 `-ffp-contract=off`，显式且合法的 FMA/点积 primitives 不因此被改写。外部 Weft compiler 已有的程序优化仍须保持输入合同。

本轮包含显式 profile 的 AOT materialization、部署后 native load/run。具备本机 compiler/toolchain 时可复用同一适配链立即编译；不把在板卡上安装完整 Intent/Weft/LLVM frontend 或接通自动发现的原生 JIT 作为本轮验收前置条件。主机编译/板卡 system compile 与调用是显式部署方式，不是本机执行失败后的 fallback；SSH 仅在生产部署/benchmark 层使用，不属于核心 runtime 调用语义。

### 7.2 固定运行范围与性能口径

本轮首先在现有 `rvv`/SG2044 的标准 RVV 环境验收，使用对应显式 profile、VLEN128 与现有 Clang18 工具链；固定单核且双方 worker budget 为 1，运行前从当前设备事实核对。K1 已可连接，但不把双设备全量或 IME 使用变成本轮门槛；硬件 identity 不进入 compiler policy。

只新增一个正式量化组合性能 case：`N=4096, K=4096`、一个 f32 activation vector，外部 Q4_K weights；对应现有 Weft `production_mul_mat_q4_k` 的 decode `M=1`。Source baseline 使用该 Weft 作者程序及完整 Q8_K helper closure，generated 与 source 使用同算法、输入、toolchain、target 和 FP flags；必要的 source/runtime 按项目 `source/` 语言/来源/职责边界接入，registry 只连接入口，不参与编译选择。现有 GGML 数据准备/数值参考可复用，不用旧 GGML 或 Weft GOP/s 折算本次 source 时间。

复用现有生产 benchmark 机制并补 Weft provider 的必要接线，远端调用正式 native entry；不得另建临时测试 runner 绕过它。一次既有 warmup、相同 cold-cache 方式和 10 次计时，报告 median ms、source ms、G/S；每次 invocation 包含 activation quantization、必要 packing/materialization、内部任务执行与同步。编译、tuning、部署、加载、外部输入/输出分配、warmup 和 eviction 不计时；workspace 存储准备的边界双方一致并注明，不把每次量化工作移到计时外。

同一次 benchmark 只做一次最终 f32 输出数值检查，使用现有量化组合容差 `abs(actual-expected) <= 1e-4 + 2e-3*abs(expected)`，沿用该 case 的有限输出要求；不增加 standalone quantize 的 byte-exact 检查或独立数值测试。语义错误或超差直接修复，只重跑这个受影响 benchmark，不放大容差。

成对实际时间进入项目现有 `report/baselinev2/` 下的 Weft provider CSV（`weft-rvv.csv`），不以外部仓库表格或 `/tmp` 报告作为交付。表格不承担候选耗时/逐轮历史审计；没有实测不得宣称 runtime、数值或性能已达标。本轮不新增统一 G/S 硬门槛。

## 8. 验收与非目标

验收仅使用 brief 的 A1–A5：共同 CPU 边界、真实可编程实现、量化 producer/consumer、Weft native 完整调用与性能、有效参数及既有 Mojo 能力。技术定位与参考材料不拆成独立测试门禁；数值与域边界属于实现义务，不据此另建输入矩阵、单元测试或临时测试脚本。

不扩展无关 GPU/DSA，不接全量量化库、不增加 whole-kernel matmul DSL，不做跨调用权重量化/repack，不引入隐藏 host invocations 或未建模 workspace/cache，不启用全局 fast-math/FTZ，不放大容差，不增加性能 benchmark 之外的测试。外部 TianchenRV 先保持只读；若发现必需的外部接口/语义缺口，拿出具体证据并就最小改动取得授权，独立提交且不覆盖他人工作。本 change 使用 main/current；完整 Shape 确认后才进入 Build。
