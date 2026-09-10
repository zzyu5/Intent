# CPU 可编程目标 lowering

## 1. 目标能力与权威

本能力实现 `doc/compiler/cpu-program-ir.md` 定义的 structured CPU program 与可编程目标 realization。CPU 仍从 canonical KIR 独立构造，形成非 SIMT 的 task/block 程序；不改为整算子库选择器，不复制 GPU execution topology，也不另起 planning IR。

Q1 已确认采用预量化 Q4_K×Q8_K 局部点积并仅生成 Canonical Weft IR。本 Spec 固定完整目标，等待最终 Shape 确认；本次 Q1 确认不授权进入 Build，也不代表现有语言已支持这些格式。确认后先将 §5 的最小语言扩展写入 `doc/dsl/`，其它设计继续按现有 `doc/` 实现。

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

## 5. 首个专业计算块：Q4_K×Q8_K 局部点积

用户已确认采用预量化 Q4_K weights 与 Q8_K activation 的局部点积。边界只承接已有量化输入的计算并生成 Canonical Weft IR，不包含 f32 activation 的量化准备、跨调用 weight interleave、persistent repack 或设备运行。

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

CPU construction/typed ABI 必须真实保存 u8 carrier、f32 result、record extent identity、原访问/快照/effects 与 format relation；拆除共享层对所有输入均为 f32 的无条件假设，不因此宣称其它 dtype/operations 全部支持。CPU structured operation 保存完整计算，不提前展开 nibble/decode 或强制走 f32 matrix matcher。

专业实现通过 §3 的同一查询/绑定/展开接口接收该 operation。首个 Weft realization 可用结构化 IRBuilder 编写，产生真实 Encoding declarations、字段读取、integer dot/reduce、转换、校正、循环/carry 与结果 SSA；不要求外部仓库新增通用 callable-module importer，也不能只把原 f32 builder 改名。若使用目标 source helper，则必须经其 frontend 形成 IR，再显式绑定进入宿主，不做字符串替换。

Adapter 将已知 byte-carrier relation 映射到相应 encoded views/values，明确区分 record axis、256-element logical axis 和 byte axis，保持外部 bytes，不隐含 copy/repack。Canonical Weft 输出中的捕获、shape/axis symbols、domain 父子关系、返回值的宿主 consumer、store、alias/effects 和资源 lifetime 必须完整；共享宿主唯一 root/终止结构，不能直接拼接 standalone kernel body。合法空域不产生 encoded-record 读取。

首个基于现行 Weft formats 的 realization 消费可追溯到连续外部 records 的读取，要求 record 起址分别满足 2-byte/4-byte alignment；任意计算产生的 u8 tensor 不在首个 realization 的覆盖范围。连续性和对齐是实现/生成 artifact 的显式 ABI 适用条件，不从普通 u8 dtype 假定。输入 pointer/offset/stride 与 provider View 的映射和这些条件随生成物保留，generation-only 不声称已经对实际调用 buffers 验证。未满足条件或 target 无对应实现时明确诊断，不静默换格式或 host 算法。外部 Weft 规范示例的 alignment=1 不作为当前 helper 的 2/4-byte pin 合同证据。

完整路径为真实 Intent 作者 kernel→canonical KIR→structured CPU→implementation expansion→Canonical Weft IR；通过已有 `intent.generate(..., target=WeftTarget(...))` 交付。示例只是该通用 lowering 的一个 consumer，不根据示例名称选择程序。不新增设备 runtime、虚假的可运行 registry 条目或第二套生成入口；Mojo/GPU 对这项新格式尚无实现时明确拒绝，不影响其已有合法能力。

### 5.5 本轮支持边界

只补上述局部点积与最小专业实现机制，不扩成完整量化 GEMM/GEMV、activation quantization、跨调用 Q4K_I interleave、设备 RVV/IME 执行或所有 layout/tail 组合。外部 TianchenRV 保持只读。最终 Shape 确认前不修改 DSL 数值规格、不新增算子实现、不执行生成。

## 6. 参数、artifact 与性能

共同参数约束 task grain、外层 block 和跨块组织；implementation 参数约束真实局部微块/vector/replica/unroll 等；外部 compiler 参数由其消费者拥有。跨边界参数只有一个 binding owner 与显式约束，不各自选择两个值。

保留有限配置覆盖、适用性过滤、完整候选实例化；Mojo native 继续实测选优并缓存 winner。Weft 本轮只生成，不伪造设备测量或 winner；单个合法专业实现不需要虚构第二份候选。格式/算法不作为可互换的 tuning 参数，也不对所有格式/算法建立笛卡尔积。Artifact 与 winner cache 分开，其身份包含实际依赖的 specialization、view facts、target、implementation 定义及 bindings。

既有 Mojo registry 的 f32 affine、RMSNorm、dense GEMM、Softmax、LayerNorm 保持可达；不改变它们的作者算法、同算法 source 及既定数值容差。受影响项复用现有生产 benchmark，在同次运行做一次容差检查，记录 generated/source native 时间、G/S 和必要说明到 `report/baselinev2/mojo-x86.csv`。未受影响结果直接复用，不要求重跑全表。

不以候选搜索、JIT 或 Python 调度耗时冒充算子时间；同机准备允许预算内并发，计时避免互相干扰。Weft 仅生成的项不填写虚构设备性能，不把 compilation verifier 当作数值运行。

## 7. 验收与非目标

验收仅使用 brief 的 A1–A4：共同 CPU 边界、真实可编程实现、选定专业计算块、有效参数/native 性能。技术定位与参考材料不拆成独立测试门禁；数值与域边界属于实现义务，不据此另建输入矩阵、单元测试或临时测试脚本。本轮不新增统一 G/S 硬门槛。

不扩展无关 GPU/DSA，不接全量量化库，不引入隐藏 host invocations 或跨调用 workspace，不启用全局 fast-math/FTZ，不放大容差，不增加性能 benchmark 之外的测试。本 change 使用 main/current；Shape 完整确认后才进入 Build。
