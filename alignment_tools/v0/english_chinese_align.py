#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英中文本对齐脚本
使用Bertalign方法对英语和中文文本进行自动对齐
"""

import re
import os
import numpy as np
from tqdm import tqdm
import argparse
from sentence_transformers import SentenceTransformer

def clean_text(text):
    """清理文本，移除多余的空白字符"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_sentences(text, language):
    """根据语言分割句子"""
    if language == 'en':
        # 英文句子分割规则
        pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s'
        sentences = re.split(pattern, text)
    elif language == 'zh':
        # 中文句子分割规则 - 基于常见的中文句号、问号和感叹号
        pattern = r'(?<=[。？！.?!])'
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

def compute_similarity_matrix(en_embeddings, zh_embeddings):
    """计算英文和中文句子之间的相似度矩阵"""
    # 归一化嵌入
    en_embeddings = en_embeddings / np.linalg.norm(en_embeddings, axis=1, keepdims=True)
    zh_embeddings = zh_embeddings / np.linalg.norm(zh_embeddings, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarity_matrix = np.dot(en_embeddings, zh_embeddings.T)
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

def align_texts(en_file, zh_file, output_file, model_name='LaBSE', batch_size=64, gap_penalty=-0.3, max_chars=None):
    """对齐英文和中文文本文件"""
    print(f"读取英文文件: {en_file}")
    with open(en_file, 'r', encoding='utf-8') as f:
        en_text = f.read()
    
    print(f"读取中文文件: {zh_file}")
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_text = f.read()
    
    print(f"英文文本长度: {len(en_text)} 字符")
    print(f"中文文本长度: {len(zh_text)} 字符")
    
    # 如果指定了最大字符数，则截取文本
    if max_chars:
        en_text = en_text[:max_chars]
        zh_text = zh_text[:max_chars]
        print(f"截取文本至 {max_chars} 字符")
    
    print("清理文本...")
    en_text = clean_text(en_text)
    zh_text = clean_text(zh_text)
    
    print("分割句子...")
    en_sentences = split_sentences(en_text, 'en')
    zh_sentences = split_sentences(zh_text, 'zh')
    
    print(f"英文句子数量: {len(en_sentences)}")
    print(f"中文句子数量: {len(zh_sentences)}")
    
    print(f"加载模型 {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("编码句子...")
    en_embeddings = encode_sentences(en_sentences, model, batch_size=batch_size)
    zh_embeddings = encode_sentences(zh_sentences, model, batch_size=batch_size)
    
    print("计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(en_embeddings, zh_embeddings)
    
    print("执行动态规划对齐...")
    alignments = dynamic_programming_align(similarity_matrix, gap_penalty=gap_penalty)
    
    # 提取对齐结果
    aligned_pairs = []
    for en_idx, zh_idx, similarity in alignments:
        aligned_pairs.append((en_sentences[en_idx], zh_sentences[zh_idx], similarity))
    
    print(f"找到 {len(aligned_pairs)} 个对齐句对")
    
    # 保存对齐结果
    print(f"保存对齐结果到 {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for en_sent, zh_sent, similarity in aligned_pairs:
            f.write(f"相似度: {similarity:.4f}\n")
            f.write(f"英文: {en_sent}\n")
            f.write(f"中文: {zh_sent}\n\n")
    
    # 分析对齐结果
    stats = analyze_and_print_statistics(aligned_pairs)
    
    # 打印前5个对齐结果
    print("\n前5个对齐结果:")
    for i, (en_sent, zh_sent, similarity) in enumerate(aligned_pairs[:5]):
        print(f"对齐 {i+1}:")
        print(f"相似度: {similarity:.4f}")
        print(f"英文: {en_sent}")
        print(f"中文: {zh_sent}")
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
    max_similarity = max(similarities)
    min_similarity = min(similarities)
    median_similarity = sorted(similarities)[len(similarities) // 2]
    
    # 计算相似度分布
    similarity_ranges = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0
    }
    
    for sim in similarities:
        if sim < 0.2:
            similarity_ranges["0.0-0.2"] += 1
        elif sim < 0.4:
            similarity_ranges["0.2-0.4"] += 1
        elif sim < 0.6:
            similarity_ranges["0.4-0.6"] += 1
        elif sim < 0.8:
            similarity_ranges["0.6-0.8"] += 1
        else:
            similarity_ranges["0.8-1.0"] += 1
    
    # 计算百分比
    for key in similarity_ranges:
        similarity_ranges[key] = (similarity_ranges[key] / len(similarities)) * 100
    
    # 计算句子长度统计
    en_lengths = [len(en) for en, _, _ in aligned_pairs]
    zh_lengths = [len(zh) for _, zh, _ in aligned_pairs]
    
    avg_en_length = sum(en_lengths) / len(en_lengths)
    avg_zh_length = sum(zh_lengths) / len(zh_lengths)
    
    # 计算长度比例
    length_ratios = [len(en) / len(zh) if len(zh) > 0 else 0 for en, zh, _ in aligned_pairs]
    avg_length_ratio = sum(length_ratios) / len(length_ratios)
    
    # 统计结果
    stats = {
        "count": len(aligned_pairs),
        "similarity": {
            "average": avg_similarity,
            "median": median_similarity,
            "max": max_similarity,
            "min": min_similarity
        },
        "similarity_distribution": similarity_ranges,
        "length": {
            "average_en": avg_en_length,
            "average_zh": avg_zh_length,
            "ratio_en_zh": avg_length_ratio
        }
    }
    
    # 打印统计信息
    print("\n===== 对齐结果统计 =====")
    print(f"总对齐句对数: {stats['count']}")
    
    print("\n--- 相似度统计 ---")
    print(f"平均相似度: {stats['similarity']['average']:.4f}")
    print(f"中位数相似度: {stats['similarity']['median']:.4f}")
    print(f"最高相似度: {stats['similarity']['max']:.4f}")
    print(f"最低相似度: {stats['similarity']['min']:.4f}")
    
    print("\n--- 相似度分布 ---")
    for range_key, percentage in stats['similarity_distribution'].items():
        print(f"相似度 {range_key}: {percentage:.2f}%")
    
    print("\n--- 句子长度统计 ---")
    print(f"英文句子平均长度: {stats['length']['average_en']:.2f} 字符")
    print(f"中文句子平均长度: {stats['length']['average_zh']:.2f} 字符")
    print(f"英中句子长度比例: {stats['length']['ratio_en_zh']:.2f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='英中文本对齐工具')
    parser.add_argument('--en', required=True, help='英文文本文件路径')
    parser.add_argument('--zh', required=True, help='中文文本文件路径')
    parser.add_argument('--output', required=True, help='对齐结果输出文件路径')
    parser.add_argument('--model', default='LaBSE', help='使用的预训练模型名称 (默认: LaBSE)')
    parser.add_argument('--batch-size', type=int, default=64, help='编码批次大小 (默认: 64)')
    parser.add_argument('--gap-penalty', type=float, default=-0.3, help='对齐间隙惩罚 (默认: -0.3)')
    parser.add_argument('--max-chars', type=int, help='处理的最大字符数 (默认: 处理全部文本)')
    
    args = parser.parse_args()
    
    aligned_pairs, stats = align_texts(
        args.en, 
        args.zh, 
        args.output, 
        model_name=args.model, 
        batch_size=args.batch_size, 
        gap_penalty=args.gap_penalty,
        max_chars=args.max_chars
    )
    
    print("对齐结果统计信息:")
    print(f"总对齐句对数: {stats['count']}")
    print(f"平均相似度: {stats['similarity']['average']:.4f}")
    print(f"中位数相似度: {stats['similarity']['median']:.4f}")
    print(f"最高相似度: {stats['similarity']['max']:.4f}")
    print(f"最低相似度: {stats['similarity']['min']:.4f}")

if __name__ == "__main__":
    main()
