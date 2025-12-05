import os
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from collections import Counter
from datetime import datetime

# ================= 配置区域 =================
MERGED_MODEL_PATH = "/data/zhuldz/lunwen/rl/train/verl1/a_model_grpo_7/qwen3_4b_code_generation_iter_0/global_step_1260/actor/huggingface"
DEFAULT_DATASET = "mbpp" 

DATASET_CONFIGS = {
    "humaneval": {
        "path": "/data/zhuldz/lunwen/generation/humaneval_pro.json",
        "output_dir": "/data/zhuldz/lunwen/eva/evalplus_results/humaneval/best_prompt",
        "format": "json",
        "keys": {"problem": "raw_problem", "solution": "raw_solution"}
    },
    "mbpp": {
        "path": "/data/zhuldz/lunwen/data/mbpp/mbpp.jsonl",
        "output_dir": "/data/zhuldz/lunwen/eva/evalplus_results/mbpp/best_prompt",
        "format": "jsonl",
        "keys": {"problem": "text", "solution": "code"}
    }
}

ZERO_SHOT_TEMPLATE = """ generate an English system prompt for the following input,IMPORTANT RULES:\nOutput ONLY the final system prompt, with NO intermediate thinking, explanations, or reasoning.\nDo NOT include phrases like 'Let me think', 'First, I need to', or any similar thought process.\nIt is not allowed to output any thinking and explanatory statements, only the generated system prompts:

【Input】
Original prompt: {raw_problem}
Correct code: {raw_solution}
"""
# ===========================================

def load_data(dataset_name, limit=50):
    if dataset_name not in DATASET_CONFIGS: raise ValueError(f"不支持: {dataset_name}")
    config = DATASET_CONFIGS[dataset_name]
    path = config["path"]
    if not os.path.exists(path): raise FileNotFoundError(f"缺少文件: {path}")
    
    print(f"📚 Loading {dataset_name} from: {path}")
    raw_data = []
    if config["format"] == "json":
        with open(path, 'r') as f: raw_data = json.load(f)
    elif config["format"] == "jsonl":
        with open(path, 'r') as f: 
            raw_data = [json.loads(line) for line in f if line.strip()]
            
    processed_data = []
    for item in raw_data[:limit]:
        prob = item.get(config["keys"]["problem"])
        sol = item.get(config["keys"]["solution"])
        if prob and sol: processed_data.append({"raw_problem": prob, "raw_solution": sol})
    return processed_data, config["output_dir"]

def smart_extract(text):
    """
    智能提取逻辑：解决'复读机'导致的内容丢失问题
    逻辑顺序：先找 Output 标记，保留其后内容；然后再处理 Original prompt 截断
    """
    if not text: return ""
    
    # 1. [找头] 优先定位 Output 标记
    # 如果模型先复读了 Input，这里会直接跳过复读部分，定位到真正的输出
    start_markers = ["【Output】", "Output:", "### Output"]
    for m in start_markers:
        if m in text:
            # 取标记之后的内容，抛弃前面的复读
            text = text.split(m, 1)[-1].strip()
            break
            
    # 2. [去尾] 截断模型幻觉出来的“下一题”
    # 此时 text 已经是 Output 之后的内容了，如果再出现 Original prompt，说明是下一题的开始
    stop_markers = ["Original prompt:", "【Input】", "<|im_end|>", "Input:"]
    for m in stop_markers:
        if m in text:
            text = text.split(m, 1)[0].strip()
            
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, choices=["humaneval", "mbpp"])
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    print(f"🚀 Loading Model: {MERGED_MODEL_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    except Exception as e:
        print(f"❌ Load Error: {e}"); return

    eval_data, output_dir = load_data(args.dataset, args.limit)
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_file = os.path.join(output_dir, f"extracted_{args.dataset}_{timestamp}.jsonl")

    extracted_prompts = []
    print(f"🔄 Extracting {len(eval_data)} samples...")

    for item in tqdm(eval_data):
        input_text = ZERO_SHOT_TEMPLATE.format(
            raw_problem=str(item['raw_problem']).strip(),
            raw_solution=str(item['raw_solution']).strip()
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_gen = full_text[len(input_text):].strip()
        
        # --- 关键修改：使用智能提取 ---
        # 这将保留 Explanation (如果它在 Output 后)，但去除 Input 复读和 Next Sample 幻觉
        final_prompt = smart_extract(raw_gen)
        
        if final_prompt:
            extracted_prompts.append(final_prompt)

    # 统计
    print("\n" + "="*50)
    counter = Counter(extracted_prompts)
    for i, (p, c) in enumerate(counter.most_common(3), 1):
        print(f"\n🏆 Rank {i} (Count: {c}, {c/len(extracted_prompts):.1%})")
        print("-" * 20)
        print(p[:300] + "..." if len(p)>300 else p)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for p in extracted_prompts:
            f.write(json.dumps({"dataset": args.dataset, "generated_system_prompt": p}, ensure_ascii=False) + "\n")
    print(f"\n💾 Saved to: {output_file}")

if __name__ == "__main__":
    main()