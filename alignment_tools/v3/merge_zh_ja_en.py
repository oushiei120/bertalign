#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并中日和中英平行语料库
基于中文文本将中日和中英CSV文件合并成一个多语言平行语料库
"""

import pandas as pd
import argparse
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_csv_files(zhja_file, zhen_file):
    """
    加载中日和中英CSV文件
    """
    print(f"加载中日文件: {zhja_file}")
    zhja_df = pd.read_csv(zhja_file)
    
    print(f"加载中英文件: {zhen_file}")
    zhen_df = pd.read_csv(zhen_file)
    
    print(f"中日文件行数: {len(zhja_df)}")
    print(f"中英文件行数: {len(zhen_df)}")
    
    return zhja_df, zhen_df

def exact_match_merge(zhja_df, zhen_df, zh_column='红楼梦原文'):
    """
    基于完全匹配的中文文本合并数据框
    """
    print("使用完全匹配方法合并...")
    
    # 创建中英文本的字典，以便快速查找
    zhen_dict = dict(zip(zhen_df[zh_column], zhen_df['English Translation']))
    
    # 为zhja_df添加英文翻译列
    zhja_df['English Translation'] = zhja_df[zh_column].map(zhen_dict)
    
    # 计算匹配统计
    matched = zhja_df['English Translation'].notna().sum()
    total = len(zhja_df)
    match_rate = matched / total * 100 if total > 0 else 0
    
    print(f"完全匹配结果: 匹配 {matched}/{total} 行 ({match_rate:.2f}%)")
    
    return zhja_df

def semantic_match_merge(zhja_df, zhen_df, model_name='LaBSE', zh_column='红楼梦原文', threshold=0.7):
    """
    使用语义相似度匹配中文文本并合并数据框
    """
    print(f"使用语义匹配方法合并 (模型: {model_name}, 阈值: {threshold})...")
    
    # 加载模型
    print(f"加载模型 {model_name}...")
    model = SentenceTransformer(model_name)
    
    # 提取未匹配的行
    unmatched_zhja = zhja_df[zhja_df['English Translation'].isna()]
    if len(unmatched_zhja) == 0:
        print("没有需要语义匹配的行")
        return zhja_df
    
    print(f"需要语义匹配的行数: {len(unmatched_zhja)}")
    
    # 编码中文句子
    print("编码中文句子...")
    zhja_texts = unmatched_zhja[zh_column].tolist()
    zhen_texts = zhen_df[zh_column].tolist()
    
    # 确保所有文本都是字符串类型
    zhja_texts = [str(text) for text in zhja_texts if pd.notna(text)]
    zhen_texts = [str(text) for text in zhen_texts if pd.notna(text)]
    
    print(f"有效的中文句子数量: {len(zhja_texts)}")
    print(f"有效的中英句子数量: {len(zhen_texts)}")
    
    if len(zhja_texts) == 0 or len(zhen_texts) == 0:
        print("没有有效的句子可以匹配")
        return zhja_df
    
    zhja_embeddings = model.encode(zhja_texts, convert_to_tensor=True, show_progress_bar=True)
    zhen_embeddings = model.encode(zhen_texts, convert_to_tensor=True, show_progress_bar=True)
    
    # 转换为numpy数组
    zhja_embeddings = zhja_embeddings.cpu().numpy()
    zhen_embeddings = zhen_embeddings.cpu().numpy()
    
    # 归一化嵌入
    zhja_embeddings = zhja_embeddings / np.linalg.norm(zhja_embeddings, axis=1, keepdims=True)
    zhen_embeddings = zhen_embeddings / np.linalg.norm(zhen_embeddings, axis=1, keepdims=True)
    
    # 计算相似度矩阵
    print("计算相似度矩阵...")
    similarity_matrix = np.dot(zhja_embeddings, zhen_embeddings.T)
    
    # 找到最佳匹配
    print("寻找最佳匹配...")
    best_matches = []
    for i, row in enumerate(tqdm(similarity_matrix)):
        best_idx = np.argmax(row)
        best_score = row[best_idx]
        
        if best_score >= threshold:
            best_matches.append((
                unmatched_zhja.iloc[i].name,  # 原始索引
                zhen_df.iloc[best_idx]['English Translation'],
                best_score
            ))
    
    # 更新数据框
    print(f"找到 {len(best_matches)} 个语义匹配")
    for idx, en_text, score in best_matches:
        zhja_df.at[idx, 'English Translation'] = en_text
        zhja_df.at[idx, 'Match Score'] = score
    
    # 计算匹配统计
    matched = zhja_df['English Translation'].notna().sum()
    total = len(zhja_df)
    match_rate = matched / total * 100 if total > 0 else 0
    
    print(f"语义匹配后结果: 匹配 {matched}/{total} 行 ({match_rate:.2f}%)")
    
    return zhja_df

def bertalign_match(zhja_df, zhen_df, zh_column='红楼梦原文'):
    """
    使用Bertalign进行中文文本对齐并合并数据框
    """
    try:
        from bertalign import Bertalign
        print("使用Bertalign进行对齐...")
        
        # 提取未匹配的行
        unmatched_zhja = zhja_df[zhja_df['English Translation'].isna()]
        if len(unmatched_zhja) == 0:
            print("没有需要Bertalign匹配的行")
            return zhja_df
        
        print(f"需要Bertalign匹配的行数: {len(unmatched_zhja)}")
        
        # 准备文本
        zhja_texts = unmatched_zhja[zh_column].tolist()
        zh_text = "\n".join(zhja_texts)
        
        zhen_texts = zhen_df[zh_column].tolist()
        en_texts = zhen_df['English Translation'].tolist()
        
        # 创建中英文本的字典，以便后续查找
        zhen_dict = dict(zip(zhen_texts, en_texts))
        
        # 使用Bertalign进行对齐
        aligner = Bertalign(zh_text, "\n".join(zhen_texts))
        alignments = aligner.align_sents()
        
        # 更新数据框
        matches_count = 0
        for src_idxs, tgt_idxs in alignments:
            if len(src_idxs) == 1 and len(tgt_idxs) == 1:  # 1-1对齐
                src_idx = src_idxs[0]
                tgt_idx = tgt_idxs[0]
                
                if src_idx < len(zhja_texts) and tgt_idx < len(zhen_texts):
                    orig_idx = unmatched_zhja.iloc[src_idx].name
                    zhja_df.at[orig_idx, 'English Translation'] = zhen_dict.get(zhen_texts[tgt_idx], '')
                    zhja_df.at[orig_idx, 'Bertalign Match'] = True
                    matches_count += 1
        
        print(f"Bertalign找到 {matches_count} 个匹配")
        
        # 计算匹配统计
        matched = zhja_df['English Translation'].notna().sum()
        total = len(zhja_df)
        match_rate = matched / total * 100 if total > 0 else 0
        
        print(f"Bertalign匹配后结果: 匹配 {matched}/{total} 行 ({match_rate:.2f}%)")
        
        return zhja_df
    
    except ImportError:
        print("警告: 无法导入Bertalign，跳过此匹配方法")
        return zhja_df

def merge_files(zhja_file, zhen_file, output_file, use_bertalign=False, use_semantic=True, 
               model_name='LaBSE', threshold=0.7, zh_column='红楼梦原文'):
    """
    合并中日和中英CSV文件
    """
    # 加载文件
    zhja_df, zhen_df = load_csv_files(zhja_file, zhen_file)
    
    # 添加英文翻译列
    zhja_df['English Translation'] = None
    zhja_df['Match Score'] = None
    
    # 步骤1: 尝试完全匹配
    zhja_df = exact_match_merge(zhja_df, zhen_df, zh_column)
    
    # 步骤2: 对于未匹配的行，尝试使用Bertalign
    if use_bertalign:
        zhja_df = bertalign_match(zhja_df, zhen_df, zh_column)
    
    # 步骤3: 对于仍未匹配的行，尝试使用语义匹配
    if use_semantic:
        zhja_df = semantic_match_merge(zhja_df, zhen_df, model_name, zh_column, threshold)
    
    # 保存结果
    print(f"保存合并结果到 {output_file}")
    zhja_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 打印最终统计
    matched = zhja_df['English Translation'].notna().sum()
    total = len(zhja_df)
    match_rate = matched / total * 100 if total > 0 else 0
    
    print(f"最终结果: 匹配 {matched}/{total} 行 ({match_rate:.2f}%)")
    
    return zhja_df

def main():
    parser = argparse.ArgumentParser(description='合并中日和中英平行语料库')
    parser.add_argument('--zhja', required=True, help='中日CSV文件路径')
    parser.add_argument('--zhen', required=True, help='中英CSV文件路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--zh-column', default='红楼梦原文', help='中文列名 (默认: 红楼梦原文)')
    parser.add_argument('--use-bertalign', action='store_true', help='使用Bertalign进行对齐')
    parser.add_argument('--use-semantic', action='store_true', help='使用语义匹配')
    parser.add_argument('--model', default='LaBSE', help='使用的句子编码模型 (默认: LaBSE)')
    parser.add_argument('--threshold', type=float, default=0.7, help='语义匹配阈值 (默认: 0.7)')
    
    args = parser.parse_args()
    
    merge_files(
        args.zhja,
        args.zhen,
        args.output,
        use_bertalign=args.use_bertalign,
        use_semantic=args.use_semantic,
        model_name=args.model,
        threshold=args.threshold,
        zh_column=args.zh_column
    )

if __name__ == "__main__":
    main()
