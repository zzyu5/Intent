# Target Extensions 与 Lowering

## 1. 两个正交选择

Compilation context分别选择：

- source provider：Triton、cuTile、TileLang或未来的直接backend；
- hardware target：NVIDIA/AMD及具体SM/gfx，或其它GPU architecture。

Provider与hardware不是同一分类。Triton source可以继续由外部Triton编译到NVIDIA或AMD；Intent不得为每个provider预建一份vendor IR，也不得为每个SM/gfx版本建立dialect。

## 2. Target capabilities

Physical module可以读取external target-capability object。它以typed features表达：

- grid/program limits；
- warp/wave size与resource limits；
- supported scalar/fragment dtypes；
- structured primitive与atomic能力；
- provider source forms；
- architecture-specific extension legality。

Passes查询features而不是匹配设备名称。SM90、SM100、SM120或gfx family通过同一IR上的feature predicates、pipeline selection与lowering patterns表达差异。

## 3. Common IR 与 local extension

共同GPU IR始终是唯一完整executable authority。Target-local extension只能扩展当前program，不能复制program mapping、value/access graph或structured semantics。

一项local extension必须同时满足：

1. 共同IR无法无损表达且不应公共化；
2. 不是单纯API名称、参数顺序或字符串差异；
3. 有后续pass/consumer需要读取或改写；
4. 需要独立legality/verifier；
5. serialization前必须成为current program的一部分。

否则使用thin legalization或直接serialization。

## 4. Triton

共同GPU program可以机械映射：

- program coordinates → `tl.program_id`与grid；
- physical ranges/fragments → `tl.arange`、scalar/blocked tensor values；
- explicit access/validity/fill → pointer expressions、mask/other与`tl.load/store`；
- gather/scatter/atomic →对应Triton operations；
- physical reduce/scan/contract → `tl.reduce/associative_scan/dot/dot_scaled`或合法展开；
- structured control → Python/Triton structured control。

Triton-local form可以包括真正影响source program的descriptor value/access、某些provider compile-time branches与Config binding。若pointer/descriptor只是终端spelling差异，可直接serialize；若descriptor选择改变operands、static block constraints并被多个passes消费，则用local extension op表达。

Tensor-descriptor extension必须在current provider program中显式保存base、shape、strides、block shape、padding、alignment与accesses，并声明descriptor runtime allocation的size/alignment、allocator ABI和launch lifetime。Provider launcher负责绑定已声明的allocator requirement；若selected Triton/runtime无法提供，provider legality在serialization前拒绝。Serializer不得临时创建未声明descriptor workspace或allocator路径。

Intent不复制外部Triton的：

- TTGIR distributed encodings与`convert_layout`；
- coalescing、thread locality与MMA/shared/TMEM representation；
- TMA lowering、software pipeline与warp specialization；
- fence/barrier/register与LLVM/PTX lowering；
- autotune winner。

## 5. cuTile

cuTile与Triton共享block-program核心：program/block identity、compile-time fragment extents、pointwise values、load/store/gather/scatter、reduction/prefix operations、MMA与control。

Thin mapping包括：

- program coordinates → `ct.bid`及所需grid values；
- fragment shape与coordinate relation → cuTile tile-space indices；
- common accesses → `ct.load/store/gather/scatter`；
- structured operations → cuTile支持的`ct.sum`/`ct.max`/`ct.cumsum`等reduction/prefix forms，以及`ct.mma/mma_scaled`。

cuTile-local legality/forms包括最多三维block identity、tile-space index multiplication、check-bounds/padding、advanced indexing、MMA-scaled layout与provider tuning constraints。它们只有在不只是机械index conversion时才形成local extension。

若selected cuTile surface不提供共同program某项协作行为所需的copy/barrier/sync形式，provider必须明确legalize到其已有等价能力或拒绝，不能生成串行慢路径冒充支持。

## 6. TileLang

TileLang同样消费共同的grid、loops、fragments、accesses与structured compute，但其author surface显式要求更多physical structure：

- storage allocation：fragment/local/shared等；
- `T.copy`、async/TMA copy与BufferRegion；
- `T.gemm`/reduce对operand storage的要求；
- `T.Pipelined` schedule metadata；
- synchronization与barrier forms。

TileLang provider pass根据共同IR的lifetime、sharing、access、dependency与structured-op facts形成必要的storage/copy/sync extensions。WGMMA、TCGEN05、mbarrier parity、named barrier、TMA instruction preference等只留在TileLang CUDA-target lowering或local extensions。

若共同program可以直接使用普通TileLang loops/access/compute表达，则不创建extension；不能为了让三家形式对称而强制Triton/cuTile也拥有allocation/copy dialect。

## 7. Vendor 与 architecture extensions

共同IR加局部extensions采用与TritonGPU相同的组织原则：

```text
common GPU program
    + optional provider/vendor extension ops
    → target-specific legalization
    → source or lower IR
```

Extension op可以只在某些features上合法；同一pass也可以按features选择不同rewrite pattern。Architecture差异不得进入KIR，也不通过kernel-name branch表达。

## 8. Provider verifier

Provider legalization完成后检查：

- grid rank、program coordinate与compile-time fragment requirements；
- every common operation具有唯一surface lowering；
- local extension operands/results/regions完整；
- unsupported dtype/primitive/access/sync组合已在serialization前拒绝；
- physical parameters完整绑定到provider constexpr/config；
- no provider pass重新读取KIR去推导shared ownership、axis、range、access或validity。

## 9. Terminal serialization

Serializer只能：

- 发出imports、signature与decorators；
- 按current IR顺序打印values、control与operations；
- 机械转换types、attributes与provider API spelling；
- 发出已声明的Config/candidate集合与launch wrapper。

Serializer不得：

- 从logical result shape猜fragment width；
- 从KIR relation现场重建pointer/mask；
- 根据role或op名称选择ownership/primitive；
- 创建buffer、workspace、persistent loop或pipeline；
- 增加未声明physical parameters；
- 捕获异常并切换fallback。

## 10. Unsupported 的性质

Unsupported必须在最早拥有足够信息的层声明：

- shared GPU legality：该physical program无法满足共同GPU执行约束；
- provider surface：目标语言没有等价operation/form；
- hardware capability：selected architecture不支持必要dtype/resource/primitive；
- lower compilation cost：source合法，但外部compiler在给定成本界限内无法完成。

这四类不能混成一个compile failure，也不能通过改变算法、缩小scope或发射数量级更慢的替代路径隐藏。
