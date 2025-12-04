import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from collections import Counter

# ================= 配置区域 =================
MERGED_MODEL_PATH = "/data/zhuldz/lunwen/rl/train/verl1/a_model_grpo_standard/qwen3_4b_code_generation_iter_0/global_step_420/actor/huggingface"

# 3. 测试数据路径 (您的数据文件)
DATA_PATH = "/data/zhuldz/lunwen/generation/humaneval_pro.json"

# 4. 输出文件
OUTPUT_LOG = "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/best_prompt/1204_1019.josnl"

# ===========================================

# [Oracle 模式] 模板：包含问题和真值，模拟训练时的输入分布
ZERO_SHOT_TEMPLATE = """I will provide you with some examples of generating system prompts. Please carefully study and understand the content and structure of these examples.\n\nBased on the examples above, generate an English system prompt for the following input (follow the same format as examples),IMPORTANT RULES:\nOutput ONLY the final system prompt, with NO intermediate thinking, explanations, or reasoning.\nDo NOT include phrases like 'Let me think', 'First, I need to', or any similar thought process.\nIt is not allowed to output any thinking and explanatory statements, only the generated system prompts:

【Input】
Original prompt: {raw_problem}
Correct code: {raw_solution}
"""

def main():
    print(f"🚀 Loading Full Merged Model from: {MERGED_MODEL_PATH}")
    
    # 1. 直接加载全量模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_PATH, 
            dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        )
    except OSError as e:
        print(f"❌ 加载失败：在 {MERGED_MODEL_PATH} 找不到模型文件。")
        print(f"错误详情: {e}")
        print("请确保您已经运行了合并脚本，并且该目录下有 config.json 文件。")
        return

    # 2. 准备数据
    if not os.path.exists(DATA_PATH):
        print(f"❌ 错误：找不到数据文件 {DATA_PATH}")
        return

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
        # 只取前 20 条验证收敛性
        eval_data = data[:20] if len(data) > 20 else data

    extracted_prompts = []
    print(f"🔄 开始为 {len(eval_data)} 条数据提取 System Prompt (Full Model Oracle Mode)...")

    # 3. 批量生成
    for item in tqdm(eval_data):
        # 获取问题和真值
        p_text = item.get('raw_problem', '')
        s_text = item.get('raw_solution', '')
        
        if not p_text or not s_text: 
            continue

        # 构造输入
        input_text = ZERO_SHOT_TEMPLATE.format(
            raw_problem=p_text.strip(),
            raw_solution=s_text.strip()
        )
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False, # 贪婪解码
                temperature=0.05
            )
        
        generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成部分
        generated_part = generated_full[len(input_text):].strip()
        
        # 清洗截断
        for stop_word in ["Correct code:", "Input:", "Original prompt:", "<|im_end|>"]:
            if stop_word in generated_part:
                generated_part = generated_part.split(stop_word)[0].strip()
            
        extracted_prompts.append(generated_part)

    # 4. 统计与分析
    print("\n" + "="*20)
    print("📊 提示词收敛情况统计 (Top 5)")
    print("="*20)
    
    counter = Counter(extracted_prompts)
    most_common = counter.most_common(5)
    
    best_prompt = None
    for i, (prompt, count) in enumerate(most_common, 1):
        ratio = count / len(extracted_prompts) * 100
        print(f"\n🏆 Rank {i} (出现 {count} 次, 占比 {ratio:.1f}%):")
        print("-" * 20)
        print(prompt)
        print("-" * 20)
        if i == 1:
            best_prompt = prompt

    # 5. 保存结果
    with open(OUTPUT_LOG, 'w') as f:
        for p in extracted_prompts:
            f.write(json.dumps({"generated_system_prompt": p}) + "\n")
            
    print(f"\n💾 提取结果已保存至: {OUTPUT_LOG}")
    
    if best_prompt:
        print("\n✅ 操作指南：")
        print("如果 Rank 1 的提示词看起来是通用的（不包含具体代码细节），")
        print("请将其复制并粘贴到您的 eva/ 评估脚本中作为 System Prompt。")

if __name__ == "__main__":
    main()