# DSL 与编译基础收尾

## 目标与边界

本能力关闭前两轮 DSL/compiler 工作已经识别但未完成的作者迁移、generic contraction lowering、退化 region canonicalization 与 TileLang f32 数值缺口。结果是一套保留原语言能力、职责明确且在本轮语义类上可执行的 compiler，为后续 cuTile 工作提供基础，不表示所有算法、provider、硬件或性能目标已经完成。

现行 `doc/` 定义语言和 compiler 分层；本 Spec 只落实已有语义及用户确认的收尾范围，不授权通过改写语言定义迁就失败。旧 cuTile change、其验收与独立暂存试验继续暂停并保持不动。

## 作者表面

`examples/kernels/` 和 `doc/dsl/examples/` 中的普通向量、矩阵、batch/transpose 乘法以及固定 builtin prefix 以已有 `dot/matvec/vecmat/matmul/outer/scaled_matmul/sparse_matmul/cumsum/cummax` 或 `reduce.sum/max/any/all` 表达。按计算角色和实际 tensor value rank/axis 判断，不按 kernel 名字或原 external-view rank 机械替换。

迁移覆盖语义同类的 dtype 版本、helpers、variants 与规格示例，不只修改少量可运行样本。显式 dtype、转换/舍入位置、result axis order、logical indexing、普通 loop、ABI、effects、可观察 chunks 和 kernel 数量不变。固定加法 scan 改成 cumsum 时保持原 accumulator dtype，不能因 builtin widening 改变算法。

Generic contract/reduce/scan/region 保持公开可用。确实需要一般 paired axes、多 free/reduction axes、自定义 combine/identity、summary/state action 的调用仍使用一般构造；严格 recurrence 使用 ordinary control。完成标准不是 generic 名字零出现，而是所有剩余用法有具体语义理由，无重复机械轴关系或协议仍无理由地留给普通计算作者。

## 统一 lowering

公开具名接口只生成现行语义已经确定的 logical transpose/broadcast 和 paired relations，再进入与 generic 相同的 canonical path。相同 canonical semantics 不因公开名字不同获得另一套 config、provider 或 serializer policy。

### 多轴 contraction

一个 generic contraction 的多个 paired reduction axes 由 compiler 根据 typed axis relation、extent identity 和 source provenance 形成等价 physical realization。轴顺序与允许的乘加重结合来自原 contract；需要的重排/双射 flatten 及其逆 result mapping 是 compiler 的工作，不能要求作者把已合法的高阶输入写成手工二维 workaround。

原始 `[32,2,32] × [2,32,32]`、`reduce=((1,0),(2,1))`、f32 accumulator 的程序须原样产出 `[32,32]` 正确结果。修复作用于同类 typed relation，不能依赖该尺寸、dimension ID、变量名或输入值。所有新增 fragment、coordinate、broadcast、tail validity、accumulator 和 store 关系在其完整 transformation group 后成立；不得仅修改 verifier 放过冲突。

### 直接 batch contraction

原始 `[2,32,64] × [2,64,32]`、`reduce=((2,1),)`、`batch=((0,0),)` 的程序须原样产出 `[2,32,32]` 正确结果。paired batch 只配对不归约，在结果中只出现一次，不因 extent 与其他轴数值相同而混同 source identity。

固定和 dynamic logical extents 采用一致的 source/axis/ownership/validity 规则。修改须关闭原始 fixed-shape store-validity failure，保持已有 dynamic named batch；不能以改用外层 batch 循环、改变 source annotation 或另一个 named sample 来替代原例。实际 mapping/blocking/访问关系保存在 current Physical Program，provider 只消费已形成的程序。

## 退化 region 归一

### Fold

当 region summarizer 只是以与 region combine 等价的 pure combine、identity 和 captures 折叠其 source slice elements 时，canonical 输出采用 ordinary reduce。归一保存 component schema、归约/source axes、logical order、identity/empty、NaN/dtype 与允许的 reassociation。

识别依据 typed helper body 与关系，不依据源码文本、helper 名称或 summary 字段名字，也不要求证明任意用户函数的代数等价。真实含 slice-level contraction、mask/normalization 或其他 tensor algorithm 的 region_fold 保留其 region semantics。

### Scan

对 element-summary/element-output 的退化 region_scan，在 typed summary、combine、state action 和 output relation 能机械证明等价时，canonical 路径使用 ordinary scan 与必要的显式 state 应用/输出操作。完整 output 和 final_state 均保持；不能只生成 prefix tensor而丢掉 initial_state、apply、emit 或 captures。

Empty source 返回 empty output 与原 initial_state；prefix 顺序和 source member relation 保持原有契约。Transition 与 state schema 不假设相同，identity 不与 initial_state 合并。不能对真实 slice-level algorithm或不可重结合的有序 recurrence 套该归一。

Physical region realization 已存在并不构成这两条 canonicalization 的完成证据：必须检查 canonical 输出和对应的实际生成/运行结果。

## Provider 数值契约

TileLang lowering 对现行 f32 contraction 保留 full-f32 输入、显式 accumulator/result dtype、empty/NaN 与允许的重结合语义。不因 external `T.gemm` 默认选择 TF32 而静默降低输入精度；采用何种等价 provider form 由对应 provider 层负责，必要的 form/type/control 必须在 serialization 前进入 current program。

原始 `64×64` 程序中，每个结果只有 `(1 + 3×2^-11) × 1` 一个非零乘积，其余乘积为零；完整 production 路径结果应为 `1.00146484375`，而非 `1.0009765625`。该例不靠 tolerance 放宽通过，也不把 benchmark 程序仅退出 0 当成数值正确。

修复保留同类 f32 输入的规则，不匹配本例数值、函数名或 shape。不增加公开近似模式来回避现行契约，不把新 unsupported/拒绝编译当修复完成；若实现确实需要改变用户可见语义或存在不可解决的 provider 边界，回到 Shape 报告具体原因，由用户决定。

## 已有能力与职责

保持上一轮已完成的具名公开签名与 binding、typed helper keyword/source-order、builtin identity、向量 contraction 和 batch broadcast。保持 Triton full-f32 与 signed floor/remainder、cuTile signed64 index/loop/scalar ABI，以及三 provider 每次 autotune trial 的 writable state/alias 隔离和一次最终可观察 invocation。

Shared construction 与具名 transformation groups 分工保持，改写在相应 group 内关闭 type/value/access/aggregate/accumulator relation，group 后 verifier 只检查、不修程序。真实 reference 用来校准成熟 compiler 的职责和 carrier，不以抽象名称或 pass 数量作成熟度标准。

JSON profiles 仍为编译期有限数据输入，默认/override 行按既有契约进入 typed domains/config；family classification、角色绑定和合法性在所属代码层。occupancy 可接受值与当前默认搜索集合分离，不因默认只选 1 而强制扩大候选，也不退回将默认值当唯一合法值。runtime winner 不回写 canonical/shared program。

## 验收与交付

验收项唯一使用 brief 的 A1-A7。各项的输入/输出、source/current/ref 关系、当前候选实际命令与数值结果独立可追踪。原始失败必须有自己的当前结论，不能被相似的新例子、不同 shape、另一 provider 或历史日志代替。

源码迁移必须检查完整作者目录并解释保留 generic 的语义理由；编译能力必须经过项目允许的 production DSL -> backend source -> JIT/launch -> 数值 repro。静态判断、实际运行、失败、未运行和范围外未知明确区分，不增加 pytest、fixtures、长期测试/性能框架或 hash/checksum 校验。

必要修复在单个 Native Change 内按统一职责推进，由新的只读复核和独立 Verifier 核对完整范围。未通过或未实现的验收项不得仅塞入 known_limits 后仍全部标记 passed；改变范围须用户明确确认。

本轮不恢复旧 cuTile 性能目标，不要求任意 examples×provider 全量或新硬件覆盖，不新增无关 IR/profiling 重构，不推送或创建 PR。归档与本地合并/工作区清理在 Archive 阶段按实际用户授权处理。
