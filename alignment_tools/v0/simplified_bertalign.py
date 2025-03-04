import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# 设置环境变量以使用CPU而不是GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def clean_text(text):
    """清理文本"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sentences_zh(text):
    """分割中文句子"""
    # 使用更简单的正则表达式
    text = re.sub(r'([。！？])', r'\1\n', text)
    
    sentences = text.split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def split_sentences_ja(text):
    """分割日文句子"""
    # 使用更简单的正则表达式
    text = re.sub(r'([。！？])', r'\1\n', text)
    
    sentences = text.split('\n')
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def dynamic_programming_align(sim_matrix, gap_penalty=-0.2):
    """使用动态规划进行句子对齐"""
    m, n = sim_matrix.shape
    score = np.zeros((m+1, n+1))
    backtrack = np.zeros((m+1, n+1, 2), dtype=int)
    
    # 初始化第一行和第一列
    for i in range(1, m+1):
        score[i, 0] = score[i-1, 0] + gap_penalty
        backtrack[i, 0] = [i-1, 0]
    for j in range(1, n+1):
        score[0, j] = score[0, j-1] + gap_penalty
        backtrack[0, j] = [0, j-1]
    
    # 填充DP表
    for i in range(1, m+1):
        for j in range(1, n+1):
            match = score[i-1, j-1] + sim_matrix[i-1, j-1]
            delete = score[i-1, j] + gap_penalty
            insert = score[i, j-1] + gap_penalty
            
            if match >= delete and match >= insert:
                score[i, j] = match
                backtrack[i, j] = [i-1, j-1]
            elif delete >= insert:
                score[i, j] = delete
                backtrack[i, j] = [i-1, j]
            else:
                score[i, j] = insert
                backtrack[i, j] = [i, j-1]
    
    # 回溯找到对齐路径
    alignments = []
    i, j = m, n
    while i > 0 or j > 0:
        prev_i, prev_j = backtrack[i, j]
        if prev_i == i-1 and prev_j == j-1:  # 匹配
            alignments.append((i-1, j-1))
        elif prev_i == i-1 and prev_j == j:  # 删除
            pass
        else:  # 插入
            pass
        i, j = prev_i, prev_j
    
    alignments.reverse()
    return alignments

def align_texts(zh_text, ja_text, model_name='LaBSE'):
    """对齐中文和日文文本"""
    print("清理文本...")
    zh_text = clean_text(zh_text)
    ja_text = clean_text(ja_text)
    
    print("分割句子...")
    zh_sentences = split_sentences_zh(zh_text)
    ja_sentences = split_sentences_ja(ja_text)
    
    print(f"中文句子数量: {len(zh_sentences)}")
    print(f"日文句子数量: {len(ja_sentences)}")
    
    print(f"加载模型 {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        
        print("编码句子...")
        zh_embeddings = model.encode(zh_sentences, show_progress_bar=True)
        ja_embeddings = model.encode(ja_sentences, show_progress_bar=True)
        
        print("计算相似度矩阵...")
        similarity_matrix = cosine_similarity(zh_embeddings, ja_embeddings)
        
        print("执行动态规划对齐...")
        alignments = dynamic_programming_align(similarity_matrix)
        
        print(f"找到 {len(alignments)} 个对齐句对")
        
        result = []
        for zh_idx, ja_idx in alignments:
            result.append((zh_sentences[zh_idx], ja_sentences[ja_idx], similarity_matrix[zh_idx, ja_idx]))
        
        return result
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    # 文件路径
    zh_file = "/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/细雪中.txt"
    ja_file = "/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/细雪日.txt"
    
    # 读取文件内容
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_text = f.read()
    
    with open(ja_file, 'r', encoding='utf-8') as f:
        ja_text = f.read()
    
    # 使用完整文本
    print("中文文本长度:", len(zh_text))
    print("日文文本长度:", len(ja_text))
    
    # 执行对齐
    aligned_pairs = align_texts(zh_text, ja_text)
    
    # 保存对齐结果
    output_file = "/Users/oushiei/Documents/GitHub/bertalign/平行语料库样本/final_aligned_result.txt"
    print(f"保存对齐结果到 {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for zh_sent, ja_sent, similarity in aligned_pairs:
            f.write(f"相似度: {similarity:.4f}\n")
            f.write(f"中文: {zh_sent}\n")
            f.write(f"日文: {ja_sent}\n\n")
    
    print("对齐完成！")
    
    # 打印部分对齐结果
    print("\n前10个对齐结果:")
    for i, (zh_sent, ja_sent, similarity) in enumerate(aligned_pairs[:10]):
        print(f"对齐 {i+1}:")
        print(f"相似度: {similarity:.4f}")
        print(f"中文: {zh_sent}")
        print(f"日文: {ja_sent}")
        print()

if __name__ == "__main__":
    main()
