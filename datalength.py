import json
import numpy as np

# 读取数据集
def analyze_tokens(file_path):
    token_lengths = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'tokens' in data:
                    # 计算tokens的长度
                    if isinstance(data['tokens'], list):
                        token_lengths.append(len(data['tokens']))
                    elif isinstance(data['tokens'], str):
                        token_lengths.append(len(data['tokens'].split()))
            except json.JSONDecodeError:
                continue
    
    # 计算统计信息
    if token_lengths:
        avg_length = np.mean(token_lengths)
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        median_length = np.median(token_lengths)
        
        print(f"总样本数: {len(token_lengths)}")
        print(f"平均长度: {avg_length:.2f}")
        print(f"最大长度: {max_length}")
        print(f"最小长度: {min_length}")
        print(f"中位数长度: {median_length:.2f}")
        
        # 显示长度分布
        print("\n长度分布:")
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(token_lengths, p):.2f}")
    else:
        print("未找到tokens数据")
    
    return token_lengths

# 分析指定文件
token_lengths = analyze_tokens('/data/zhongyikun/medmax/MoDE-official/data/ScienceQA/train_data.jsonl')