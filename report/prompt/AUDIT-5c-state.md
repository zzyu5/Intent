# 收尾核查：第 5c 轮是否真正闭合

只核查最终状态，不继续推进、不重新调查、不新建报告。实现尚未完成时不要提前执行这份核查。

先读 `report/prompt/05c-intentdsl-shared-gpu-correctness-closure-prompt.md` 和本轮最终报告，再看
`git log`、`git status`、`git diff`。已经完成的 217 shared corpus与 74-entry provider probe不为得到
同一个数字而重跑；当前 54-entry Triton数值与性能结果直接检查本轮留下的报告和两张 CSV。

回答下面四个问题，每项给 current `file:line` 或当前产物证据。

## 一、physical decision 是否唯一

- dense GEMM 的 M/N execution axes是否由 tensor-contraction realization精化已有 ownership，而不是
  追加第二份 mapping；
- 被替代的 `FRAGMENT_D*` parameter、type/range expression、死 SSA和 config dimension是否全部消失；
- ordinary/scaled tensor contraction是否消费同一 ownership replacement原则；
- serializer或 repro adapter是否仍在隐藏、过滤一份 shared/provider program中残留的旧决定。

## 二、parameter 与 config 是否已经分层

- shared IR是否只承载 physical parameter role、current-IR relation和 typed legality；
- 默认编译是否只绑定一组静态合法 config；
- 可选 autotune是否只消费预先给定的少量完整 config tuples，而不是独立 domains的笛卡尔积；
- provider是否只绑定已有 parameters、provider-local launch options并过滤非法 tuple，没有重建
  execution axes或扩张候选；
- Triton terminal source是否已恢复为有界规模，配置列表不再支配整个文件。

## 三、正确性责任是否说清楚

- `doc/compiler/gpu-program-ir.md` 的十二条不变量是否逐条落到“当前层检查”或“明确由后续层拥有”，
  有没有仍处于“规格写了、实现没说”的状态；
- analysis unknown是否保留合法保守表示，semantic illegality是否给 typed diagnostic，provider legality
  是否由明确 carrier延后；
- assertion开关是否可能改变 legality；若本轮没有触及相关 construction/type路径，引用已有结果即可，
  不重跑完整 corpus；
- 本轮 shared修改若触及 cuTile/TileLang共同结构，是否已有受影响 entry的定向证据；不要求重跑全部
  74 entries。

## 四、Triton 是否真正跑完

- 当前 54-entry 是否完成 terminal source、JIT、launch、numerical comparison和 benchmark；
- generated/source是否按 semantic roles使用等价完整 config tuples，各自由 Triton autotune选 winner；
- `triton-5090.csv`、`triton-h100.csv` 是否来自当前代码；
- 超过 `1.05×` 的每项是否有当前 Physical Program、generated/source和定向运行依据；
- 最终报告是否只写当前事实，没有复用历史“下层质量”“算法不同”“target不支持”作为现成结论。

最后确认旧 helper、fallback、临时 runner、日志和巨大生成文件已删除，相关修改形成语义完整提交，
工作区干净。发现未完成项就明确指出，不为了让核查通过而在这里修改实现。
