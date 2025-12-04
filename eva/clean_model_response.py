import json
import re
import string
from collections import Counter

# ================= 配置 =================
# 输入文件路径
INPUT_FILE = "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/best_prompt/1204_1019.josnl"
# =======================================

def aggressive_clean(text):
    if not text: return ""
    
    # 1. 基础解码
    try:
        text = text.encode('utf-8').decode('unicode_escape')
    except:
        pass

    # 2. [结构提取] 优先利用 【Output】 标记 (最强分割)
    if "【Output】" in text:
        text = text.split("【Output】")[-1]
    elif "Output:" in text:
        text = text.split("Output:")[-1]

    # 3. [尾部截断] 切掉 Example 等后续
    for stopper in ["【Example】", "【Input】", "<|im_end|>", "### Example", "Example:", "Input:"]:
        if stopper in text:
            text = text.split(stopper)[0]

    text = text.strip()

    # 4. [锚点定位法] 正面寻找系统提示词的开头
    # 既然很难删掉前面的废话，不如直接找 "You are..." 在哪里
    # 常见的 System Prompt 开头锚点：
    anchors = [
        "You are a", "You are an", "Act as a", "Your task is", 
        "Generate python code", "Complete the following",
        "Please act as", "As a "
    ]
    
    # 找到最早出现的锚点，保留从那里开始的内容
    first_anchor_idx = len(text)
    found_anchor = False
    
    for anchor in anchors:
        # 忽略大小写查找
        idx = text.lower().find(anchor.lower())
        if idx != -1 and idx < first_anchor_idx:
            first_anchor_idx = idx
            found_anchor = True
            
    if found_anchor:
        # 只要找到了锚点，就大胆地丢弃前面的所有内容
        # print(f"[Debug] 切除前缀: {text[:first_anchor_idx]}...") # 调试用
        text = text[first_anchor_idx:]
    else:
        # 如果没找到锚点，回退到原来的正则清洗逻辑 (保底)
        patterns_to_remove = [
            r"^Explanation:.*?\n", r"^The code.*?\.\s*", r"^Analysis:\s*",
            r"^Sure,.*?:", r"^Here is.*?:", r"^.*?:", # 去掉开头带冒号的短语
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, "", text, count=1, flags=re.MULTILINE | re.IGNORECASE).strip()

    # 5. 清洗 Markdown 和多余空白
    text = text.replace("```python", "").replace("```", "").strip()
    return text

def normalize_for_counting(text):
    """
    归一化函数：用于统计去重
    将 "You are a coder." 和 "you are a coder" 视为同一个
    """
    # 转小写
    norm = text.lower()
    # 去除标点符号
    norm = norm.translate(str.maketrans('', '', string.punctuation))
    # 去除所有空白字符（包括换行）
    norm = "".join(norm.split())
    return norm

def main():
    print(f"📂 Reading from: {INPUT_FILE}")
    
    # 存储结构：{ normalized_key: { "original": longest_version, "count": N } }
    grouped_prompts = {}
    total_valid = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                # 兼容格式
                json_part = line[line.find('{'):]
                item = json.loads(json_part)
                raw_generated = item.get('generated_system_prompt', '')
                
                # 1. 强力清洗
                clean = aggressive_clean(raw_generated)
                
                # 过滤太短的无效生成
                if clean and len(clean) > 15:
                    total_valid += 1
                    
                    # 2. 归一化键值
                    key = normalize_for_counting(clean)
                    
                    if key not in grouped_prompts:
                        grouped_prompts[key] = {"original": clean, "count": 0}
                    
                    grouped_prompts[key]["count"] += 1
                    
                    # 总是保留最长/最完整的那个版本作为代表展示 (有时候清洗过度会变短)
                    if len(clean) > len(grouped_prompts[key]["original"]):
                        grouped_prompts[key]["original"] = clean
                        
            except Exception:
                pass

    print(f"✅ 有效提取总数: {total_valid}")
    if total_valid == 0: return

    # 排序
    sorted_groups = sorted(grouped_prompts.values(), key=lambda x: x['count'], reverse=True)
    
    print("\n" + "="*60)
    print("🏆 智能聚合后的 Top Prompt")
    print("="*60)
    
    for i, item in enumerate(sorted_groups[:5], 1):
        count = item['count']
        ratio = count / total_valid
        original_text = item['original']
        
        print(f"\n🥇 Rank {i} (Count: {count}/{total_valid}, Ratio: {ratio:.1%})")
        print("-" * 30)
        print(original_text)
        print("-" * 30)
        
        if i == 1:
            best_prompt = original_text

    print(f"\n💡 建议：请直接复制 Rank 1 的内容。即使占比没有达到 80%，它也是当前模型认为概率密度最高的“最大公约数”。")

if __name__ == "__main__":
    main()