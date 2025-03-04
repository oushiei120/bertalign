# 文本对齐工具集

本目录包含了用于文本对齐的各种工具，分为不同版本，每个版本都有其特点和适用场景。

## 目录结构

```
alignment_tools/
├── v0/                         # 双语对齐工具
│   ├── simplified_bertalign.py # 简化版Bertalign
│   └── english_chinese_align.py # 英中文本对齐工具
├── v1/                         # 第一版三语对齐工具
│   ├── trilingual_alignment.py # 基础三语对齐工具
│   └── alignment_to_excel.py   # 对齐结果转Excel工具
├── v2/                         # 第二版三语对齐工具（改进版）
│   └── trilingual_alignment_improved.py # 改进版三语对齐工具
├── chinese_japanese_align.py   # 中日文本对齐基础工具
└── README.md                   # 本说明文件
```

## 版本说明

### v0 - 双语对齐工具

基础的双语对齐工具，用于两种语言之间的文本对齐：

#### simplified_bertalign.py
简化版的Bertalign工具，专为中日文本对齐设计：
- 使用LaBSE模型进行句子编码
- 通过动态规划算法找到最佳对齐路径
- 可处理5000到100000字符的文本
- 避免了原始Bertalign可能遇到的内存和分段错误问题

使用方法：
```bash
python alignment_tools/v0/simplified_bertalign.py --src 源语言文件.txt --tgt 目标语言文件.txt --output 对齐结果.txt
```

#### english_chinese_align.py
专为英中文本对齐设计的工具：
- 使用相同的LaBSE模型和动态规划算法
- 针对英中语言对进行了优化
- 提供详细的对齐结果和统计信息

使用方法：
```bash
python alignment_tools/v0/english_chinese_align.py --en 英文文件.txt --zh 中文文件.txt --output 对齐结果.txt
```

### v1 - 基础三语对齐工具

基础版本的三语对齐工具，使用简单的对齐策略：
- 通过中文1作为桥梁，连接中文2和日文
- 直接合并两个对齐结果
- 适用于结构相似度高的文本

使用方法：
```bash
python alignment_tools/v1/trilingual_alignment.py --zh1 平行语料库样本/细雪中.txt --zh2 平行语料库样本/细雪中第二个版本.txt --zh-ja-alignment 平行语料库样本/细雪对齐结果.txt --output 细雪三语对齐结果.xlsx
```

### v2 - 改进版三语对齐工具

改进版的三语对齐工具，使用更精确的对齐策略：
- 先在段落级别对齐两个中文版本
- 然后在匹配的段落内部进行句子级别的对齐
- 最后将中文1-中文2对齐结果与中文1-日文对齐结果合并
- 适用于结构差异较大的文本

使用方法：
```bash
python alignment_tools/v2/trilingual_alignment_improved.py --zh1 平行语料库样本/细雪中.txt --zh2 平行语料库样本/细雪中第二个版本.txt --zh-ja-alignment 平行语料库样本/细雪对齐结果.txt --output 细雪三语对齐结果_改进版.xlsx
```

### 中日文本对齐工具

基础的中日文本对齐工具，用于生成中文和日文的对齐结果：
- 使用LaBSE模型进行句子编码
- 通过动态规划算法找到最佳对齐路径
- 生成包含相似度信息的对齐结果

使用方法：
```bash
python alignment_tools/chinese_japanese_align.py --zh 平行语料库样本/细雪中.txt --ja 平行语料库样本/细雪日.txt --output 平行语料库样本/细雪对齐结果.txt
```

## 输出格式

对齐工具会生成以下格式的输出：

1. 双语对齐工具(v0)：生成文本格式的对齐结果，包含相似度信息
2. 中日对齐工具：生成文本格式的对齐结果，包含相似度信息
3. 三语对齐工具(v1和v2)：生成Excel格式的对齐结果，包含以下内容：
   - 三种语言的文本内容
   - 相似度分数
   - 基于相似度的检查类别（使用不同颜色标记）

## 选择建议

- 对于双语对齐：
  - 中日文本对齐：使用 `chinese_japanese_align.py` 或 `v0/simplified_bertalign.py`
  - 英中文本对齐：使用 `v0/english_chinese_align.py`
  
- 对于三语对齐：
  - 如果已有中日对齐结果，需要整合第二个中文版本：
    - 当两个中文版本结构相似时，使用v1版本，速度更快
    - 当两个中文版本差异较大时，建议使用v2版本，精度更高

## 性能参考

根据实际测试，我们的中日文本对齐工具在《细雪》文本上的表现：
- 总共找到5686个对齐句对
- 平均相似度为0.6471，中位数相似度为0.6639
- 约55.15%的对齐句对相似度在0.6-0.8之间
- 中文句子平均长度为42.34个字符，日文句子平均长度为100.98个字符
- 中日句子长度比例平均值为0.56

这表明对齐质量很好，大多数句对具有较高的语义相似度。
