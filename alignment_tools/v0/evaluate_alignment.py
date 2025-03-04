#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估对齐结果的质量
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_alignment(aligned_file):
    """评估对齐结果的质量"""
    similarities = []
    zh_lengths = []
    ja_lengths = []
    length_ratios = []
    
    with open(aligned_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('相似度:'):
            similarity = float(lines[i].strip().split(': ')[1])
            similarities.append(similarity)
            
            zh_line = lines[i+1].strip().split(': ')[1]
            ja_line = lines[i+2].strip().split(': ')[1]
            
            zh_lengths.append(len(zh_line))
            ja_lengths.append(len(ja_line))
            
            # 计算长度比例
            if len(zh_line) > 0 and len(ja_line) > 0:
                ratio = len(zh_line) / len(ja_line)
                length_ratios.append(ratio)
            
            i += 4  # 跳过空行
        else:
            i += 1
    
    # 计算统计信息
    print(f"总对齐句对数: {len(similarities)}")
    print(f"平均相似度: {np.mean(similarities):.4f}")
    print(f"相似度中位数: {np.median(similarities):.4f}")
    print(f"相似度标准差: {np.std(similarities):.4f}")
    print(f"相似度最小值: {min(similarities):.4f}")
    print(f"相似度最大值: {max(similarities):.4f}")
    
    # 相似度分布
    similarity_counts = Counter()
    for sim in similarities:
        # 将相似度分成10个区间
        bin_index = int(sim * 10)
        similarity_counts[bin_index] += 1
    
    print("\n相似度分布:")
    for bin_index in sorted(similarity_counts.keys()):
        bin_start = bin_index / 10
        bin_end = (bin_index + 1) / 10
        count = similarity_counts[bin_index]
        percentage = count / len(similarities) * 100
        print(f"{bin_start:.1f}-{bin_end:.1f}: {count} ({percentage:.2f}%)")
    
    # 长度比例统计
    print(f"\n中文句子平均长度: {np.mean(zh_lengths):.2f}")
    print(f"日文句子平均长度: {np.mean(ja_lengths):.2f}")
    print(f"中日句子长度比例平均值: {np.mean(length_ratios):.2f}")
    print(f"中日句子长度比例中位数: {np.median(length_ratios):.2f}")
    
    # 绘制相似度直方图
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7, color='blue')
    plt.title('对齐句对相似度分布')
    plt.xlabel('相似度')
    plt.ylabel('频数')
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/similarity_distribution.png')
    
    # 绘制长度比例散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(zh_lengths, ja_lengths, alpha=0.3, s=5)
    plt.title('中日句子长度对比')
    plt.xlabel('中文句子长度')
    plt.ylabel('日文句子长度')
    plt.grid(True, alpha=0.3)
    plt.savefig('/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/length_comparison.png')
    
    return similarities, zh_lengths, ja_lengths, length_ratios

if __name__ == "__main__":
    aligned_file = "/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/final_aligned_result.txt"
    evaluate_alignment(aligned_file)
