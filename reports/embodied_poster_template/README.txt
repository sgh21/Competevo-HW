具身智能课程 · 大作业 Poster 模板
=====================================

【文件清单】
  poster.tex                   海报源文件（在这里填写你的内容）
  beamerthemesharelatex.sty    海报主题样式（无需修改）
  logo.png                     页眉校徽（可替换为你需要的 logo）
  poster.pdf                   预编译效果预览（占位版，供你参考模板风格）

【如何编译】 ★ 必须使用 XeLaTeX，不能用 pdfLaTeX ★
  · Overleaf（推荐）：
      新建项目并上传以上全部文件 → 左上角 Menu → Settings → Compiler
      选择 "XeLaTeX" → 点 Recompile 即可。
  · 本地（已装 TeX Live / MacTeX 完整版）：
      在本目录执行   xelatex poster.tex
      （依赖 ctex、beamerposter、fandol 中文字体，完整版 TeX 均自带；
        建议连编两遍，确保页脚短标题与引用编号正确。）

【如何填写】
  · 把 poster.tex 中所有【】里的占位文字、以及灰色斜体的填写提示，
    替换成你自己的内容（提交前请确保灰色提示已全部删除）。
  · 插图：用  \includegraphics[width=0.9\columnwidth]{你的图.png}
    替换正文里两处蓝色「图占位」方框。
  · 标题 / 组员姓名学号 / 院系，在 poster.tex 顶部的 \title、\author、
    \institute 处修改；指导教师已填为「刘华平」。
  · 章节可按需增删，但建议保留「背景 → 方法 → 实验 → 结论」主线。
  · 本次成稿 poster.tex 已按任务要求改为 A4 竖版；原 poster.pdf 仍是模板风格预览。

【提交】
  导出最终 PDF 提交，具体方式与截止时间以课程通知为准。
