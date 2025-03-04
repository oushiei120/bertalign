#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
英文文本直接与中日文件中的中文进行对齐
使用Bertalign或语义相似度方法将英文文本与中文对齐，然后合并日文翻译
"""

import pandas as pd
import argparse
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

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

def direct_align_with_bertalign(zhja_df, zhen_df, zh_column='红楼梦原文', en_column='English Translation'):
    """
    使用Bertalign直接对齐英文和中文
    """
    try:
        from bertalign import Bertalign
        print("使用Bertalign进行英文到中文的对齐...")
        
        # 提取中文和英文文本
        zh_texts = zhja_df[zh_column].dropna().astype(str).tolist()
        en_texts = zhen_df[en_column].dropna().astype(str).tolist()
        
        print(f"中文句子数量: {len(zh_texts)}")
        print(f"英文句子数量: {len(en_texts)}")
        
        # 使用Bertalign进行对齐
        print("初始化Bertalign...")
        aligner = Bertalign("\n".join(zh_texts), "\n".join(en_texts))
        
        print("执行句子对齐...")
        alignments = aligner.align_sents()
        
        print(f"找到 {len(alignments)} 个对齐关系")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(columns=zhja_df.columns.tolist() + ['English Translation'])
        
        # 处理对齐结果
        for src_idxs, tgt_idxs in tqdm(alignments, desc="处理对齐结果"):
            if len(src_idxs) > 0 and len(tgt_idxs) > 0:
                # 获取中文句子和对应的行
                for src_idx in src_idxs:
                    if src_idx < len(zh_texts):
                        # 找到原始DataFrame中匹配的行
                        matching_rows = zhja_df[zhja_df[zh_column] == zh_texts[src_idx]]
                        
                        if not matching_rows.empty:
                            # 对于每个匹配的中文行，添加对应的英文翻译
                            for _, row in matching_rows.iterrows():
                                new_row = row.copy()
                                # 合并目标句子（如果有多个）
                                en_text = " ".join([en_texts[i] for i in tgt_idxs if i < len(en_texts)])
                                new_row['English Translation'] = en_text
                                # 添加到结果DataFrame
                                result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
        
        print(f"成功对齐 {len(result_df)} 行")
        return result_df
    
    except ImportError:
        print("警告: 无法导入Bertalign，将使用语义匹配方法")
        return semantic_align_en_to_zh(zhja_df, zhen_df, zh_column, en_column)

def semantic_align_en_to_zh(zhja_df, zhen_df, zh_column='红楼梦原文', en_column='English Translation', model_name='LaBSE', threshold=0.6, batch_size=32):
    """
    使用语义相似度直接对齐英文和中文
    """
    print(f"使用语义匹配方法对齐英文到中文 (模型: {model_name}, 阈值: {threshold})...")
    
    # 加载模型
    print(f"加载模型 {model_name}...")
    model = SentenceTransformer(model_name)
    
    # 提取中文和英文文本
    zh_texts = zhja_df[zh_column].dropna().astype(str).tolist()
    en_texts = zhen_df[en_column].dropna().astype(str).tolist()
    
    print(f"中文句子数量: {len(zh_texts)}")
    print(f"英文句子数量: {len(en_texts)}")
    
    # 创建中文句子到原始行的映射
    zh_to_rows = {}
    for i, row in zhja_df.iterrows():
        if pd.notna(row[zh_column]):
            zh_text = str(row[zh_column])
            if zh_text not in zh_to_rows:
                zh_to_rows[zh_text] = []
            zh_to_rows[zh_text].append(i)
    
    # 编码句子
    print("编码中文句子...")
    zh_embeddings = []
    for i in tqdm(range(0, len(zh_texts), batch_size), desc="中文批次"):
        batch = zh_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        zh_embeddings.append(batch_embeddings)
    
    if len(zh_embeddings) > 1:
        zh_embeddings = torch.cat(zh_embeddings, dim=0)
    else:
        zh_embeddings = zh_embeddings[0]
    
    print("编码英文句子...")
    en_embeddings = []
    for i in tqdm(range(0, len(en_texts), batch_size), desc="英文批次"):
        batch = en_texts[i:i+batch_size]
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
                matches.append((zh_idx, best_en_idx, best_score))
    
    print(f"找到 {len(matches)} 个匹配")
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(columns=zhja_df.columns.tolist() + ['English Translation'])
    
    # 处理匹配结果
    for zh_idx, en_idx, score in tqdm(matches, desc="处理匹配结果"):
        zh_text = zh_texts[zh_idx]
        en_text = en_texts[en_idx]
        
        # 找到原始DataFrame中匹配的行索引
        if zh_text in zh_to_rows:
            for row_idx in zh_to_rows[zh_text]:
                new_row = zhja_df.iloc[row_idx].copy()
                new_row['English Translation'] = en_text
                new_row['Match Score'] = score
                # 添加到结果DataFrame
                result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(f"成功对齐 {len(result_df)} 行")
    return result_df

def align_and_merge(zhja_file, zhen_file, output_file, use_bertalign=True, 
                   model_name='LaBSE', threshold=0.6, zh_column='红楼梦原文', en_column='English Translation'):
    """
    对齐英文到中文并合并日文翻译
    """
    # 加载文件
    zhja_df, zhen_df = load_csv_files(zhja_file, zhen_file)
    
    # 对齐英文到中文
    if use_bertalign:
        try:
            result_df = direct_align_with_bertalign(zhja_df, zhen_df, zh_column, en_column)
        except Exception as e:
            print(f"Bertalign对齐失败: {e}")
            print("切换到语义匹配方法...")
            result_df = semantic_align_en_to_zh(zhja_df, zhen_df, zh_column, en_column, model_name, threshold)
    else:
        result_df = semantic_align_en_to_zh(zhja_df, zhen_df, zh_column, en_column, model_name, threshold)
    
    # 保存结果
    print(f"保存对齐结果到 {output_file}")
    result_df.to_csv(output_file, index=False, encoding='utf-8')
    
    # 打印最终统计
    total_zhja = len(zhja_df)
    aligned = len(result_df)
    align_rate = aligned / total_zhja * 100 if total_zhja > 0 else 0
    
    print(f"最终结果: 对齐 {aligned}/{total_zhja} 行 ({align_rate:.2f}%)")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='英文文本直接与中日文件中的中文进行对齐')
    parser.add_argument('--zhja', required=True, help='中日CSV文件路径')
    parser.add_argument('--zhen', required=True, help='中英CSV文件路径')
    parser.add_argument('--output', required=True, help='输出CSV文件路径')
    parser.add_argument('--zh-column', default='红楼梦原文', help='中文列名 (默认: 红楼梦原文)')
    parser.add_argument('--en-column', default='English Translation', help='英文列名 (默认: English Translation)')
    parser.add_argument('--use-bertalign', action='store_true', help='使用Bertalign进行对齐')
    parser.add_argument('--model', default='LaBSE', help='使用的句子编码模型 (默认: LaBSE)')
    parser.add_argument('--threshold', type=float, default=0.6, help='语义匹配阈值 (默认: 0.6)')
    
    args = parser.parse_args()
    
    align_and_merge(
        args.zhja,
        args.zhen,
        args.output,
        use_bertalign=args.use_bertalign,
        model_name=args.model,
        threshold=args.threshold,
        zh_column=args.zh_column,
        en_column=args.en_column
    )

if __name__ == "__main__":
    main()
