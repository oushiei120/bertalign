#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将额外的英文TXT文件添加到已对齐的中日英数据中
使用语义相似度匹配将新的英文文本与已有的中文对齐
"""

import pandas as pd
import argparse
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import re

def load_aligned_data(aligned_file):
    """
    加载已对齐的CSV文件
    """
    print(f"加载已对齐的文件: {aligned_file}")
    df = pd.read_csv(aligned_file)
    print(f"已对齐文件行数: {len(df)}")
    return df

def load_english_txt(txt_file, encoding='utf-8'):
    """
    加载英文TXT文件并分割成句子
    """
    print(f"加载英文TXT文件: {txt_file}")
    with open(txt_file, 'r', encoding=encoding) as f:
        text = f.read()
    
    # 简单的句子分割
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    print(f"提取到 {len(sentences)} 个英文句子")
    return sentences

def align_new_english(aligned_df, en_sentences, zh_column='红楼梦原文', en_column='New English', 
                     model_name='LaBSE', threshold=0.6, batch_size=32):
    """
    将新的英文句子与已对齐数据中的中文进行对齐
    """
    print(f"使用语义匹配方法对齐新英文到中文 (模型: {model_name}, 阈值: {threshold})...")
    
    # 加载模型
    print(f"加载模型 {model_name}...")
    model = SentenceTransformer(model_name)
    
    # 提取中文文本
    zh_texts = aligned_df[zh_column].dropna().astype(str).tolist()
    
    # 去重
    unique_zh_texts = list(set(zh_texts))
    print(f"唯一中文句子数量: {len(unique_zh_texts)}")
    print(f"英文句子数量: {len(en_sentences)}")
    
    # 创建中文句子到原始行的映射
    zh_to_rows = {}
    for i, row in aligned_df.iterrows():
        if pd.notna(row[zh_column]):
            zh_text = str(row[zh_column])
            if zh_text not in zh_to_rows:
                zh_to_rows[zh_text] = []
            zh_to_rows[zh_text].append(i)
    
    # 编码句子
    print("编码中文句子...")
    zh_embeddings = []
    for i in tqdm(range(0, len(unique_zh_texts), batch_size), desc="中文批次"):
        batch = unique_zh_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        zh_embeddings.append(batch_embeddings)
    
    if len(zh_embeddings) > 1:
        zh_embeddings = torch.cat(zh_embeddings, dim=0)
    else:
        zh_embeddings = zh_embeddings[0]
    
    print("编码英文句子...")
    en_embeddings = []
    for i in tqdm(range(0, len(en_sentences), batch_size), desc="英文批次"):
        batch = en_sentences[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        en_embeddings.append(batch_embeddings)
    
    if len(en_embeddings) > 1:
        en_embeddings = torch.cat(en_embeddings, dim=0)
    else:
        en_embeddings = en_embeddings[0]
    
    # 转换为numpy数组
    zh_embeddings = zh_embeddings.cpu().numpy()
    en_embeddings = en_embeddings.cpu().numpy()
    
    # 归一化嵌入
    zh_embeddings = zh_embeddings / np.linalg.norm(zh_embeddings, axis=1, keepdims=True)
    en_embeddings = en_embeddings / np.linalg.norm(en_embeddings, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    # 分批计算相似度以节省内存
    batch_size = 1000
    matches = []
    
    for i in tqdm(range(0, len(zh_embeddings), batch_size), desc="计算相似度"):
        end_idx = min(i + batch_size, len(zh_embeddings))
        batch_similarity = np.dot(zh_embeddings[i:end_idx], en_embeddings.T)
        
        for j in range(batch_similarity.shape[0]):
            zh_idx = i + j
            best_en_idx = np.argmax(batch_similarity[j])
            best_score = batch_similarity[j][best_en_idx]
            
            if best_score >= threshold:
                matches.append((unique_zh_texts[zh_idx], en_sentences[best_en_idx], best_score))
    
    print(f"找到 {len(matches)} 个匹配")
    
    # 为原始DataFrame添加新的英文列
    aligned_df[en_column] = None
    aligned_df[f'{en_column}_Score'] = None
    
    # 处理匹配结果
    match_count = 0
    for zh_text, en_text, score in tqdm(matches, desc="处理匹配结果"):
        # 找到原始DataFrame中匹配的行索引
        if zh_text in zh_to_rows:
            for row_idx in zh_to_rows[zh_text]:
                aligned_df.at[row_idx, en_column] = en_text
                aligned_df.at[row_idx, f'{en_column}_Score'] = score
                match_count += 1
    
    print(f"成功添加 {match_count} 个新英文翻译")
    return aligned_df

def add_english_txt(aligned_file, txt_file, output_file, zh_column='红楼梦原文', en_column='New English',
                   model_name='LaBSE', threshold=0.6, encoding='utf-8'):
    """
    将英文TXT文件添加到已对齐的数据中
    """
    # 加载已对齐的数据
    aligned_df = load_aligned_data(aligned_file)
    
    # 加载英文TXT文件
    en_sentences = load_english_txt(txt_file, encoding)
    
    # 对齐新英文到中文
    result_df = align_new_english(aligned_df, en_sentences, zh_column, en_column, model_name, threshold)
    
    # 保存结果
    print(f"保存结果到 {output_file}")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 打印最终统计
    matched = result_df[en_column].notna().sum()
    total = len(result_df)
    match_rate = matched / total * 100 if total > 0 else 0
    
    print(f"最终结果: 添加了 {matched}/{total} 行新英文翻译 ({match_rate:.2f}%)")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='将英文TXT文件添加到已对齐的中日英数据中')
    parser.add_argument('--aligned', required=True, help='已对齐的CSV文件路径')
    parser.add_argument('--txt', required=True, help='英文TXT文件路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--zh-column', default='红楼梦原文', help='中文列名 (默认: 红楼梦原文)')
    parser.add_argument('--en-column', default='New English', help='新英文列名 (默认: New English)')
    parser.add_argument('--model', default='LaBSE', help='使用的句子编码模型 (默认: LaBSE)')
    parser.add_argument('--threshold', type=float, default=0.6, help='语义匹配阈值 (默认: 0.6)')
    parser.add_argument('--encoding', default='utf-8', help='TXT文件编码 (默认: utf-8)')
    
    args = parser.parse_args()
    
    add_english_txt(
        args.aligned,
        args.txt,
        args.output,
        zh_column=args.zh_column,
        en_column=args.en_column,
        model_name=args.model,
        threshold=args.threshold,
        encoding=args.encoding
    )

if __name__ == "__main__":
    main()
