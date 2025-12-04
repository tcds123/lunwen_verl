# 文件名: extract_best_prompt.py
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

# ================= 配置 =================
# 1. 基础模型路径 (和训练时一致)
BASE_MODEL_PATH = "/data/zhuldz/lunwen/models/Qwen3-4B" 
# 2. 训练后的 LoRA/Checkpoint 路径 (请修改为您实际的 output 路径)
ADAPTER_PATH = "/data/zhuldz/lunwen/rl/train/verl1/a_model_grpo_standard/qwen3_4b_code_generation_iter_0/global_step_450/actor/huggingface" 
# 3. 测试数据 (使用 humaneval_pro.json 或训练数据)
DATA_PATH = "/data/zhuldz/lunwen/data/humaneval/humaneval_pro.json"
# 4. 输出文件
OUTPUT_LOG = "extracted_prompts.jsonl"
# =======================================

def main():
    print(f"🚀 加载基础模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )

    if os.path.exists(ADAPTER_PATH):
        print(f"🔗 加载 LoRA 权重: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model.merge_and_unload() # 合并权重以加速推理
    else:
        print("⚠️ 未找到 Adapter 路径，将使用原始模型进行推理！")

    # 加载数据
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
        # 只取前 50 条做采样即可，看是否收敛
        data = data[:50] if len(data) > 50 else data

    results = []
    print("🔄 开始生成 System Prompts...")

    for item in tqdm(data):
        # 构造输入 (必须与训练时 CustomReward 中的格式一致)
        # 假设训练时输入包含了 "Original prompt: ..." 标记
        prompt_text = item['raw_problem']
        input_text = f"Original prompt: {prompt_text}\nCorrect code:"
        
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, # System Prompt 通常不长
                do_sample=False,    # 使用贪婪解码，看模型最想输出什么
                temperature=0.0
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成部分 (去掉输入)
        generated_part = generated_text[len(input_text):].strip()
        
        # 简单的清洗 (去掉可能的 artifact)
        clean_prompt = generated_part.split('\n')[0].strip() # 假设 Prompt 是一行，或者取第一段
        
        results.append(clean_prompt)

    # 保存并分析
    with open(OUTPUT_LOG, 'w') as f:
        for p in results:
            f.write(json.dumps({"prompt": p}) + "\n")
    
    print("\n📊 统计出现频率最高的 Prompt:")
    from collections import Counter
    counts = Counter(results)
    for p, c in counts.most_common(5):
        print(f"[{c}次] {p}")

    best_prompt = counts.most_common(1)[0][0]
    print(f"\n🏆 提取到的最佳通用 Prompt:\n{best_prompt}")

if __name__ == "__main__":
    main()