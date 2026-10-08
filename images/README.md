# Image sources / 图片来源

These diagrams belong to the IntentDSL project and illustrate compiler responsibilities rather than measured performance.

- **`overview.svg`**: the existing IntentDSL workflow illustration, adapted from the paper's overview (`paper/figv2/fig1.pdf`). Target labels reflect the current Triton, cuTile, Mojo, Weft, and BANG C providers.
- **`execution-models.png`**: rendered from `paper/figv2/fig5.pdf` in the IntentDSL paper repository. It shows GPU program ownership, a CPU task hierarchy, and DSA local supply for an author-defined region computation. Exported as a 2400-pixel-wide PNG using `pdftocairo`, without modifying the figure's contents.

以上图片来自 IntentDSL 项目与论文，展示编译职责，不代表性能测评。`overview.svg` 沿用原工作流图并使用当前 provider 名称；`execution-models.png` 来自论文的跨执行模型结构图，保留原图内容。

The project [Apache-2.0 license](../LICENSE) applies to these project-owned images.
