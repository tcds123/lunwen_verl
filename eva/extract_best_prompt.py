import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
from datetime import datetime

# ================= 配置区域 =================
# 模型路径 (保持您文件中的路径)
MERGED_MODEL_PATH = "/data/zhuldz/lunwen/rl/train/verl1/a_model_grpo_standard/qwen3_4b_code_generation_iter_0/global_step_420/actor/huggingface"

# 默认数据集 (修改此处可直接切换 'humaneval' 或 'mbpp')
DEFAULT_DATASET = "mbpp" 

# 数据集配置中心
DATASET_CONFIGS = {
    "humaneval": {
        "path": "/data/zhuldz/lunwen/generation/humaneval_pro.json",
        "output_dir": "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/best_prompt",
        "format": "json",
        "keys": {
            "problem": "raw_problem",
            "solution": "raw_solution"
        }
    },
    "mbpp": {
        "path": "/data/zhuldz/lunwen/data/mbpp/mbpp.jsonl",
        "output_dir": "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/best_prompt",
        "format": "jsonl",
        "keys": {
            "problem": "text",  # MBPP 使用 'text' 字段
            "solution": "code"  # MBPP 使用 'code' 字段
        }
    }
}

# [Oracle 模式] 模板 (保持原样，与训练对齐)
ZERO_SHOT_TEMPLATE = """I will provide you with some examples of generating system prompts. Please carefully study and understand the content and structure of these examples.\n\nBased on the examples above, generate an English system prompt for the following input (follow the same format as examples),IMPORTANT RULES:\nOutput ONLY the final system prompt, with NO intermediate thinking, explanations, or reasoning.\nDo NOT include phrases like 'Let me think', 'First, I need to', or any similar thought process.\nIt is not allowed to output any thinking and explanatory statements, only the generated system prompts:

【Input】
Original prompt: {raw_problem}
Correct code: {raw_solution}
"""

# ===========================================

def load_data(dataset_name, limit=50):
    """加载并标准化数据格式"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    path = config["path"]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    
    print(f"📚 Loading {dataset_name} data from: {path}")
    
    raw_data = []
    # JSON 格式 (列表)
    if config["format"] == "json":
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    # JSONL 格式 (每行一个对象)
    elif config["format"] == "jsonl":
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
    
    # 提取统一格式
    processed_data = []
    keys = config["keys"]
    
    # 截取前 N 条
    data_to_process = raw_data[:limit] if len(raw_data) > limit else raw_data
    
    for item in data_to_process:
        prob = item.get(keys["problem"])
        sol = item.get(keys["solution"])
        if prob and sol:
            processed_data.append({
                "raw_problem": prob,
                "raw_solution": sol
            })
            
    return processed_data, config["output_dir"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["humaneval", "mbpp"], help="选择数据集")
    parser.add_argument("--limit", type=int, default=50, help="提取样本数量")
    args = parser.parse_args()

    print(f"🚀 Loading Model from: {MERGED_MODEL_PATH}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
    except OSError as e:
        print(f"❌ 加载失败：在 {MERGED_MODEL_PATH} 找不到模型文件。")
        print(f"错误详情: {e}")
        return

    # 加载数据
    eval_data, output_dir = load_data(args.dataset, args.limit)
    
    # 准备输出文件
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f"extracted_{args.dataset}_{timestamp}.jsonl")

    extracted_prompts = []
    print(f"🔄 开始为 {len(eval_data)} 条数据提取 System Prompt...")

    for item in tqdm(eval_data):
        # 构造输入
        input_text = ZERO_SHOT_TEMPLATE.format(
            raw_problem=str(item['raw_problem']).strip(),
            raw_solution=str(item['raw_solution']).strip()
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512, # 长度给够，保留完整输出以便分析
                do_sample=False,    # 贪婪解码
                temperature=0.0
            )
        
        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成部分 (去掉 Input)
        generated_part = generated_full[len(input_text):].strip()
        
        # 【保留原始输出】不做 split 清洗，只保存模型吐出的完整内容
        extracted_prompts.append(generated_part)

    # 统计与分析
    print("\n" + "="*50)
    print(f"📊 {args.dataset} 提示词统计 (Top 5)")
    print("="*50)
    
    counter = Counter(extracted_prompts)
    most_common = counter.most_common(5)
    
    for i, (prompt, count) in enumerate(most_common, 1):
        ratio = count / len(extracted_prompts) * 100
        print(f"\n🏆 Rank {i} (Count: {count}, Ratio: {ratio:.1f}%)")
        print("-" * 20)
        # 只打印前200个字符预览
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-" * 20)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in extracted_prompts:
            f.write(json.dumps({"dataset": args.dataset, "generated_system_prompt": p}, ensure_ascii=False) + "\n")
            
    print(f"\n💾 提取结果已保存至: {output_file}")

if __name__ == "__main__":
    main()