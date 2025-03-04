# Bertalign 项目文档

## 项目概述

Bertalign 是一个自动多语言句子对齐工具，旨在促进多语言平行语料库和翻译记忆库的构建。这些资源在翻译相关研究中有广泛的应用，包括基于语料库的翻译研究、对比语言学、计算机辅助翻译、译者教育和机器翻译等领域。

本项目使用先进的语言模型和动态规划算法，实现了高质量的多语言文本对齐功能，特别是针对中文、日文和英文等语言的对齐进行了优化。

## 技术原理

Bertalign 使用 [sentence-transformers](https://github.com/UKPLab/sentence-transformers) 来表示源语言和目标语言的句子，使得不同语言中语义相似的句子被映射到相似的向量空间中。然后，执行基于动态规划的两步算法：

1. 第一步：找到1-1对齐的近似锚点
2. 第二步：将搜索路径限制在锚点之间，提取所有有效的1对多、多对1或多对多的对齐关系

## 项目结构

```
bertalign/
├── alignment_tools/           # 文本对齐工具集
│   ├── v0/                    # 双语对齐工具
│   │   ├── simplified_bertalign.py       # 简化版Bertalign
│   │   ├── english_chinese_align.py      # 英中文本对齐工具
│   │   ├── english_japanese_align.py     # 英日文本对齐工具
│   │   ├── chinese_japanese_align.py     # 中日文本对齐工具
│   │   └── evaluate_alignment.py         # 对齐结果评估工具
│   ├── v1/                    # 第一版三语对齐工具
│   │   ├── trilingual_alignment.py       # 基础三语对齐工具
│   │   └── alignment_to_excel.py         # 对齐结果转Excel工具
│   ├── v2/                    # 第二版三语对齐工具（改进版）
│   │   └── trilingual_alignment_improved.py  # 改进版三语对齐工具
│   ├── v3/                    # 第三版对齐工具（使用更先进的模型）
│   └── README.md              # 对齐工具说明文件
├── bertalign/                 # 核心库
│   ├── __init__.py            # 初始化文件
│   ├── aligner.py             # 对齐器主类
│   ├── corelib.py             # 核心算法库
│   ├── encoder.py             # 句子编码器
│   ├── eval.py                # 评估脚本
│   └── utils.py               # 工具函数
├── text+berg/                 # 测试语料库
├── 平行语料库样本/            # 中日文平行语料库样本
├── requirements.txt           # 依赖包列表
├── setup.py                   # 安装脚本
└── README.md                  # 项目主说明文件
```

## 核心组件

### 1. Bertalign 核心库

Bertalign 核心库提供了通用的多语言句子对齐功能，支持25种语言之间的对齐：

- **aligner.py**: 定义了 `Bertalign` 类，提供主要的对齐接口
- **corelib.py**: 实现了核心的对齐算法，包括动态规划和相似度计算
- **encoder.py**: 实现了句子编码器，使用预训练的多语言模型
- **eval.py**: 提供对齐结果的评估功能
- **utils.py**: 提供各种辅助功能，如句子切分和语言检测

### 2. 文本对齐工具集

文本对齐工具集提供了针对特定语言对的优化实现和更高级的功能：

#### v0 - 双语对齐工具

- **simplified_bertalign.py**: 简化版的Bertalign，专为中日文本对齐设计
- **english_chinese_align.py**: 专为英中文本对齐设计的工具
- **english_japanese_align.py**: 专为英日文本对齐设计的工具
- **chinese_japanese_align.py**: 专为中日文本对齐设计的工具
- **evaluate_alignment.py**: 评估对齐结果的工具

#### v1 - 基础三语对齐工具

- **trilingual_alignment.py**: 基础的三语对齐工具，使用简单的对齐策略
- **alignment_to_excel.py**: 将对齐结果转换为Excel格式的工具

#### v2 - 改进版三语对齐工具

- **trilingual_alignment_improved.py**: 改进版的三语对齐工具，使用更精确的对齐策略

#### v3 - 使用更先进模型的对齐工具

- 该目录计划实现使用更先进模型（如XLM-RoBERTa、mT5、BLOOM等）的对齐工具

## 支持的语言

Bertalign 支持25种语言之间的对齐，包括：
加泰罗尼亚语(ca)、中文(zh)、捷克语(cs)、丹麦语(da)、荷兰语(nl)、英语(en)、芬兰语(fi)、法语(fr)、德语(de)、希腊语(el)、匈牙利语(hu)、冰岛语(is)、意大利语(it)、立陶宛语(lt)、拉脱维亚语(lv)、挪威语(no)、波兰语(pl)、葡萄牙语(pt)、罗马尼亚语(ro)、俄语(ru)、斯洛伐克语(sk)、斯洛文尼亚语(sl)、西班牙语(es)、瑞典语(sv)和土耳其语(tr)。

## 使用方法

### 基本用法

```python
from bertalign import Bertalign

# 初始化对齐器，自动检测源语言和目标语言
aligner = Bertalign(src_text, tgt_text)

# 执行句子对齐
alignments = aligner.align_sents()

# 打印对齐结果
aligner.print_sents(alignments)
```

### 中日文本对齐

```bash
python alignment_tools/v0/chinese_japanese_align.py --zh 源中文文件.txt --ja 目标日文文件.txt --output 对齐结果.txt
```

### 三语对齐

基础版（适用于结构相似的文本）：
```bash
python alignment_tools/v1/trilingual_alignment.py --zh1 中文版本1.txt --zh2 中文版本2.txt --zh-ja-alignment 中日对齐结果.txt --output 三语对齐结果.xlsx
```

改进版（适用于结构差异较大的文本）：
```bash
python alignment_tools/v2/trilingual_alignment_improved.py --zh1 中文版本1.txt --zh2 中文版本2.txt --zh-ja-alignment 中日对齐结果.txt --output 三语对齐结果_改进版.xlsx
```

## 性能表现

根据实际测试，Bertalign 在中日文本对齐上的表现：
- 总共找到5686个对齐句对
- 平均相似度为0.6471，中位数相似度为0.6639
- 约55.15%的对齐句对相似度在0.6-0.8之间
- 中文句子平均长度为42.34个字符，日文句子平均长度为100.98个字符
- 中日句子长度比例平均值为0.56

这表明对齐质量很好，大多数句对具有较高的语义相似度。

在公开的德法平行语料库 Text+Berg 上，Bertalign 的表现优于传统的基于长度、字典或机器翻译的对齐方法。

## 技术特点

1. **基于语义的对齐**：使用预训练的多语言模型（如LaBSE）捕获句子的语义，而不仅仅依赖于长度或词汇匹配。

2. **无需分词**：对于中文和日文等语言，不需要额外的分词步骤，模型可以直接处理完整的句子。

3. **高效的动态规划算法**：使用优化的动态规划算法找到最佳对齐路径，支持1对1、1对多、多对1和多对多的对齐关系。

4. **多语言支持**：支持25种语言之间的任意对齐，特别优化了中文、日文和英文之间的对齐。

5. **三语对齐**：支持三种语言版本的对齐，适用于不同结构相似度的文本。

## 依赖库

- numba==0.60.0
- faiss-gpu==1.7.2 (或 faiss-cpu==1.7.2)
- googletrans==4.0.0rc1
- sentence-splitter==1.4
- sentence-transformers==3.2.1

## 未来发展方向

1. **更先进的模型**：计划在v3版本中集成更先进的模型，如XLM-RoBERTa、mT5、BLOOM等，以进一步提高对齐质量。

2. **更多语言支持**：扩展对更多低资源语言的支持。

3. **更高效的算法**：优化算法以处理更大规模的文本。

4. **用户界面**：开发图形用户界面，使工具更易于使用。

5. **更多评估指标**：增加更多评估指标，以全面评估对齐质量。

## 贡献者

- Jason (bfsujason@163.com)

## 许可证

请参阅项目根目录下的 LICENCE 文件。
