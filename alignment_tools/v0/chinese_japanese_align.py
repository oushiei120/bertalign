#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中日文本对齐脚本
使用Bertalign方法对中文和日文文本进行自动对齐
"""

import re
import os
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    """清理文本，移除多余的空白字符"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text, language):
    """根据语言分割句子"""
    if language == 'zh':
        # 中文句子分割规则 - 基于常见的中文句号、问号和感叹号
        pattern = r'(?<=[。？！.?!])'
        sentences = re.split(pattern, text)
    elif language == 'ja':
        # 日文句子分割规则 - 基于常见的日文句号和问号
        pattern = r'(?<=[。？！])'
        sentences = re.split(pattern, text)
    else:
        raise ValueError(f"不支持的语言: {language}")
    
    # 过滤空句子
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    return sentences

def encode_sentences(sentences, model, batch_size=32):
    """使用预训练模型编码句子"""
    embeddings = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="编码批次"):
        batch = sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    
    # 合并所有批次的嵌入
    if len(embeddings) > 1:
        import torch
        embeddings = torch.cat(embeddings, dim=0)
    else:
        embeddings = embeddings[0]
    
    # 转换为numpy数组
    return embeddings.cpu().numpy()

def compute_similarity_matrix(zh_embeddings, ja_embeddings):
    """计算中文和日文句子之间的相似度矩阵"""
    # 归一化嵌入
    zh_embeddings = zh_embeddings / np.linalg.norm(zh_embeddings, axis=1, keepdims=True)
    ja_embeddings = ja_embeddings / np.linalg.norm(ja_embeddings, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarity_matrix = np.dot(zh_embeddings, ja_embeddings.T)
    return similarity_matrix

def dynamic_programming_align(similarity_matrix, gap_penalty=-0.2):
    """使用动态规划算法找到最佳对齐路径"""
    m, n = similarity_matrix.shape
    # 初始化得分矩阵和回溯矩阵
    score_matrix = np.zeros((m+1, n+1))
    backtrack = np.zeros((m+1, n+1), dtype=int)
    
    # 初始化第一行和第一列
    for i in range(1, m+1):
        score_matrix[i, 0] = score_matrix[i-1, 0] + gap_penalty
        backtrack[i, 0] = 1  # 向上
    
    for j in range(1, n+1):
        score_matrix[0, j] = score_matrix[0, j-1] + gap_penalty
        backtrack[0, j] = 2  # 向左
    
    # 填充得分矩阵
    for i in range(1, m+1):
        for j in range(1, n+1):
            # 计算三种可能的得分
            match = score_matrix[i-1, j-1] + similarity_matrix[i-1, j-1]
            delete = score_matrix[i-1, j] + gap_penalty
            insert = score_matrix[i, j-1] + gap_penalty
            
            # 选择最大得分
            if match >= delete and match >= insert:
                score_matrix[i, j] = match
                backtrack[i, j] = 0  # 对角线
            elif delete >= insert:
                score_matrix[i, j] = delete
                backtrack[i, j] = 1  # 向上
            else:
                score_matrix[i, j] = insert
                backtrack[i, j] = 2  # 向左
    
    # 回溯找到最佳对齐路径
    alignments = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrack[i, j] == 0:  # 对角线
            alignments.append((i-1, j-1, similarity_matrix[i-1, j-1]))
            i -= 1
            j -= 1
        elif i > 0 and backtrack[i, j] == 1:  # 向上
            i -= 1
        else:  # 向左
            j -= 1
    
    # 反转对齐结果
    alignments.reverse()
    return alignments

def align_texts(zh_file, ja_file, output_file, model_name='LaBSE', batch_size=64, gap_penalty=-0.3, max_chars=None):
    """对齐中文和日文文本文件"""
    print(f"读取中文文件: {zh_file}")
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_text = f.read()
    
    print(f"读取日文文件: {ja_file}")
    with open(ja_file, 'r', encoding='utf-8') as f:
        ja_text = f.read()
    
    print(f"中文文本长度: {len(zh_text)} 字符")
    print(f"日文文本长度: {len(ja_text)} 字符")
    
    # 如果指定了最大字符数，则截取文本
    if max_chars:
        zh_text = zh_text[:max_chars]
        ja_text = ja_text[:max_chars]
        print(f"截取文本至 {max_chars} 字符")
    
    print("清理文本...")
    zh_text = clean_text(zh_text)
    ja_text = clean_text(ja_text)
    
    print("分割句子...")
    zh_sentences = split_sentences(zh_text, 'zh')
    ja_sentences = split_sentences(ja_text, 'ja')
    
    print(f"中文句子数量: {len(zh_sentences)}")
    print(f"日文句子数量: {len(ja_sentences)}")
    
    print(f"加载模型 {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("编码句子...")
    zh_embeddings = encode_sentences(zh_sentences, model, batch_size=batch_size)
    ja_embeddings = encode_sentences(ja_sentences, model, batch_size=batch_size)
    
    print("计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(zh_embeddings, ja_embeddings)
    
    print("执行动态规划对齐...")
    alignments = dynamic_programming_align(similarity_matrix, gap_penalty=gap_penalty)
    
    # 提取对齐结果
    aligned_pairs = []
    for zh_idx, ja_idx, similarity in alignments:
        aligned_pairs.append((zh_sentences[zh_idx], ja_sentences[ja_idx], similarity))
    
    print(f"找到 {len(aligned_pairs)} 个对齐句对")
    
    # 保存对齐结果
    print(f"保存对齐结果到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for zh_sent, ja_sent, similarity in aligned_pairs:
            f.write(f"相似度: {similarity:.4f}\n")
            f.write(f"中文: {zh_sent}\n")
            f.write(f"日文: {ja_sent}\n\n")
    
    # 分析对齐结果
    stats = analyze_and_print_statistics(aligned_pairs)
    
    # 打印前5个对齐结果
    print("\n前5个对齐结果:")
    for i, (zh_sent, ja_sent, similarity) in enumerate(aligned_pairs[:5]):
        print(f"对齐 {i+1}:")
        print(f"相似度: {similarity:.4f}")
        print(f"中文: {zh_sent}")
        print(f"日文: {ja_sent}")
        print()
    
    return aligned_pairs, stats

def analyze_and_print_statistics(aligned_pairs):
    """分析对齐结果并打印统计信息"""
    if not aligned_pairs:
        print("没有找到对齐句对")
        return {}
    
    # 提取相似度
    similarities = [sim for _, _, sim in aligned_pairs]
    
    # 计算基本统计信息
    avg_similarity = sum(similarities) / len(similarities)
    median_similarity = sorted(similarities)[len(similarities) // 2]
    min_similarity = min(similarities)
    max_similarity = max(similarities)
    
    # 计算相似度分布
    sim_ranges = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0
    }
    
    for sim in similarities:
        if sim < 0.2:
            sim_ranges["0.0-0.2"] += 1
        elif sim < 0.4:
            sim_ranges["0.2-0.4"] += 1
        elif sim < 0.6:
            sim_ranges["0.4-0.6"] += 1
        elif sim < 0.8:
            sim_ranges["0.6-0.8"] += 1
        else:
            sim_ranges["0.8-1.0"] += 1
    
    # 计算百分比分布
    total = len(similarities)
    sim_percentages = {k: (v / total) * 100 for k, v in sim_ranges.items()}
    
    # 计算句子长度统计
    zh_lengths = [len(zh) for zh, _, _ in aligned_pairs]
    ja_lengths = [len(ja) for _, ja, _ in aligned_pairs]
    
    avg_zh_length = sum(zh_lengths) / len(zh_lengths)
    avg_ja_length = sum(ja_lengths) / len(ja_lengths)
    
    # 计算中日句子长度比例
    length_ratios = [len(zh) / len(ja) if len(ja) > 0 else 0 for zh, ja, _ in aligned_pairs]
    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    
    # 打印统计信息
    print("\n对齐结果统计:")
    print(f"总对齐句对数量: {len(aligned_pairs)}")
    print(f"平均相似度: {avg_similarity:.4f}")
    print(f"中位数相似度: {median_similarity:.4f}")
    print(f"最小相似度: {min_similarity:.4f}")
    print(f"最大相似度: {max_similarity:.4f}")
    
    print("\n相似度分布:")
    for k, v in sim_percentages.items():
        print(f"  {k}: {v:.2f}% ({sim_ranges[k]} 对)")
    
    print("\n句子长度统计:")
    print(f"中文句子平均长度: {avg_zh_length:.2f} 字符")
    print(f"日文句子平均长度: {avg_ja_length:.2f} 字符")
    print(f"中日句子长度比例平均值: {avg_length_ratio:.2f}")
    
    # 返回统计信息字典
    stats = {
        "total_pairs": len(aligned_pairs),
        "avg_similarity": avg_similarity,
        "median_similarity": median_similarity,
        "min_similarity": min_similarity,
        "max_similarity": max_similarity,
        "similarity_distribution": sim_percentages,
        "avg_zh_length": avg_zh_length,
        "avg_ja_length": avg_ja_length,
        "avg_length_ratio": avg_length_ratio
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='中日文本对齐工具')
    parser.add_argument('--zh', required=True, help='中文文本文件路径')
    parser.add_argument('--ja', required=True, help='日文文本文件路径')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--model', default='LaBSE', help='使用的句子编码模型 (默认: LaBSE)')
    parser.add_argument('--batch-size', type=int, default=64, help='编码批次大小 (默认: 64)')
    parser.add_argument('--gap-penalty', type=float, default=-0.3, help='对齐空位惩罚 (默认: -0.3)')
    parser.add_argument('--max-chars', type=int, help='处理的最大字符数 (默认: 全部)')
    
    args = parser.parse_args()
    
    align_texts(
        args.zh,
        args.ja,
        args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        gap_penalty=args.gap_penalty,
        max_chars=args.max_chars
    )

if __name__ == "__main__":
    main()
