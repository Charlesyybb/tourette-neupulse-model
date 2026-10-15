# 第 2 章：受迫耦合振子模型 —— LaTeX 工程

## 文件构成

| 文件 | 说明 |
|---|---|
| `main.tex` | 驱动文件：preamble + 记号宏 + 公式编号设置。**只有这里有配置** |
| `chapter2_body.tex` | 第 2 章正文（2.1–2.18 节，公式 2.1–2.37） |
| `appendixA_body.tex` | 附录 A（A.1–A.11 节，公式 A.1–A.43） |
| `refs.bib` | 参考文献，30 条 |
| `Makefile` | `make` / `make quick` / `make clean` |

## 编译

```bash
make            # 完整：xelatex → bibtex → xelatex ×2
make quick      # 单遍，仅查语法错误（交叉引用会是 ??）
```

或手动：

```bash
xelatex main && bibtex main && xelatex main && xelatex main
```

**必须用 xelatex**（不能用 pdflatex，中文会失败）。

已验证：24 页，0 错误，0 未定义引用，公式编号 (2.1)–(2.37) 无缺号。

## 环境说明（重要）

`main.tex` 的 CJK 部分写成了**双路径**，会自动检测：

- **有 `ctex`**（你本地的 TeX Live / CTeX 套装应该有）→ 用 ctex，章节标题显示
  「第 2 章」「附录 A」，中文断行与标点挤压符合中文排版规范。**推荐路径。**
- **无 `ctex`** → 回落到 `fontspec` + Noto Serif CJK，章节标题显示
  「Chapter 2」「Appendix A」。数学与结构完全一致，仅中文排版细节略差。

若你本地缺 ctex：

```bash
tlmgr install ctex xecjk zhnumber        # TeX Live
# 或 Debian/Ubuntu:  apt install texlive-lang-chinese
```

## 并入你自己的手稿

`chapter2_body.tex` 与 `appendixA_body.tex` **不含任何 preamble**，可直接
`\input`。你的主文件需提供：

**宏包**：`amsmath, amssymb, amsthm, mathtools, booktabs, tabularx, enumitem,
natbib, hyperref, cleveref`（hyperref 必须在 cleveref 之前）

**设置**：
```latex
\numberwithin{equation}{chapter}   % 让公式编号成为 2.x
\newtheorem{hypo}{命题}
\newtheorem{remark}{说明}[chapter]
```

**记号宏**：把 `main.tex` 中「记号宏」整段复制过去（`\wbar`、`\geff`、
`\Feff`、`\Om`、`\Ostar`、`\dd`、`\ii`、`\Var`、`\Rea`、`\Ima`、`\Kc`、
`\Thf`、`\TPM`、`\YGTSS`、`\keyeq`）。

> 注意：`\TPM` 与 `\YGTSS` 用了 `\ensuremath` 包裹，因为正文和公式里都要用。
> `\geff` 本身含下标，若需再加个体下标请写 `\gamma_{\mathrm{eff},i}`，
> 不要写 `\geff_{,i}`（双下标报错）。

## 待办：参考文献核对

`refs.bib` 中标 `% [待核实]` 的 6 条需要你在 CrossRef / PubMed 上补齐卷号页码，
**投稿前必须处理**：

- `Millard2024` —— 期刊信息完全缺失。这是对个体化假设（H3）**最强的反证**
  之一（tACS 对峰值 alpha 频率无总体效应的元分析），务必找到并在
  2.18 节正面引用。
- `Berchicci2011` —— 我用它支撑「成人 mu 去同步峰 10–12 Hz」，但该文是婴幼儿
  研究，**引用不当**。建议改引 Pfurtscheller 系列或成人 mu 峰频的直接文献。
- `Ali2013`、`Tyulkina2018`、`Cellier2022`、`Lorenz2019` —— 主体可靠，核对卷页。

## 结构对照

正文 18 节 ↔ 附录 11 节的推导对应关系：

```
2.6  OA 降维        → A.1  旋转系代数
2.7  极坐标         → A.2  实虚部分离
2.8  主方程 (2.15)  → A.3  消去 Φ
2.9.1 区制 I        → A.4  线性化 + 半高半宽 + 守恒关系
2.9.2 区制 II       → A.5  Adler 约化 + AM-GM 说明
2.11 噪声 (2.26)    → A.6  Kc = 2(γ+D) 精确插值
2.12 最优频率       → A.7  偏移公式 + 数值表
2.18(6) 抖动        → A.8  量级估计（负面结果）
2.14 观测模型       → A.9  Rice 公式 → 唯象门控
2.16 可辨识性       → A.10 Fisher 零方向
2.13.1 次谐波       → A.11 判别实验设计
```
