# 妥瑞氏研究算法文件提取

从三份聊天 JSON 导出中提取 create_file 的原始文本，按时间顺序应用 str_replace 及 Bash 中的文本替换操作。未执行聊天中的历史 Shell 命令或重新生成模拟数据。所有记录中的文本修改均成功匹配。

## 内容

- fkm/fkm/individualize.py：个体化刺激频率算法，包括功率谱非周期背景拟合、mu 峰估计、不确定度收缩、频率量化及回退策略。
- fkm/fkm/core.py：受迫 Kuramoto 模型解析核心。
- fkm/fkm/numerics.py：振子数值模拟及自相关估计。
- fkm/fkm/observation.py、waveform.py、units.py：观测模型、波形及单位。
- fkm/scripts/：四个演示及验证脚本。
- fkm/tests/：四个测试模块及初始化文件。
- latex/：论文 LaTeX 源文件、参考文献、Makefile，及用户上传的 main(2).pdf（原样复制为 main.pdf，并非重新编译）。

## 未恢复部分及验证范围

旧会话仅提供结果缓存 results/m2_oa.json、results/m2_spont.json、results/m2_driven.json 的路径，没有对应文件完整字节，故未恢复这些缓存。运行 m2_validation.py report 前需要重新运行相应模拟生成缓存。旧下载链接不代表文件仍然可取。

已对全部提取 Python 文件做语法检查，通过。当前环境未安装 pytest，未重新运行测试。原项目 README 中的测试数量和数值结论属于历史记录，不是此次提取的复验结果。本次只恢复已有内容，未审核或修改算法的科学结论。

依赖及运行方法参见 fkm/README.md。extraction_provenance.json 记录初始文件来源；extraction_issues.json 为空表示此次记录内的文本修改均成功处理。
