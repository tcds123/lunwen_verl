import os
import json
import time
import re
import requests
import subprocess
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, write_jsonl
from evalplus.data import get_mbpp_plus, write_jsonl

# ================= 配置区域 =================
# API 配置 (Qwen3-Code-480B)
API_KEY = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
BASE_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
MODEL_NAME = "f-devops-qwen3-coder-480b-a35b-instruct"

# 输出文件
#OUTPUT_FILE = "samples_qwen480b_evalplus_humaneval.jsonl"
OUTPUT_FILE = "samples_qwen480b_evalplus_mbpp.jsonl"
# ===========================================

class Qwen480BAPI:
    """集成您提供的 API 客户端逻辑"""
    def __init__(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        self.generation_params = {
            "temperature": 0.0, # 评估通常用 0 温度以保证结果可复现 (Pass@1)
            "top_p": 1.0,
            "max_tokens": 1024, 
            "n": 1,
            "stream": False
        }

    def generate(self, prompt: str, retry_count: int = 5) -> str:
        messages = [
            {"role": "system", "content": "You are a professional Python code generator. Please complete the function based on the provided docstring. Output the executable code directly."},
            {"role": "user", "content": prompt}
        ]
        
        data = {
            "model": MODEL_NAME,
            "messages": messages,
            **self.generation_params
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(BASE_URL, headers=self.headers, json=data, timeout=120)
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    time.sleep(5 * (attempt + 1))
                    continue
                else:
                    print(f"API Error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Request Error: {e}")
                time.sleep(2)
        return ""

def extract_code(text: str) -> str:
    """清洗 API 返回的 Markdown"""
    # 优先提取 ```python 代码块
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    
    # 其次提取通用代码块
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    
    return text

def main():
    print("🚀 启动标准 EvalPlus 评估流程 (API: Qwen-480B)")
    
    # --- 【关键修改】检查是否已有结果文件 ---
    if os.path.exists(OUTPUT_FILE):
        print(f"\n✨ 发现已存在的生成结果: {OUTPUT_FILE}")
        print("⏭️  跳过生成步骤，直接开始评估...")
    
    else:
        # 如果文件不存在，则开始生成
        # 1. 加载标准数据集
        # print("📚 正在加载 HumanEval+ 数据集...")
        # problems = get_human_eval_plus()
        print("📚 正在加载 MBPP+ 数据集...")
        problems = get_mbpp_plus()
        api = Qwen480BAPI()
        samples = []
        
        print(f"🔄 开始生成 {len(problems)} 个任务的代码...")
        
        # 2. 遍历生成
        for task_id, problem in tqdm(problems.items()):
            raw_prompt = problem["prompt"]
            
            # 修改为 (MBPP 风格):
            instruct_prompt = (
                f"Please write a Python function to solve the following problem:\n"
                f"{raw_prompt}\n\n"
                f"Output the executable code directly inside a code block.\n"
                f"```python\n"
            )
            
            # 调用 API
            response = api.generate(instruct_prompt)
            
            # 清洗代码
            clean_code = extract_code(response)
            
            # 记录
            samples.append({
                "task_id": task_id,
                "completion": clean_code
            })

        # 3. 保存结果
        write_jsonl(OUTPUT_FILE, samples)
        print(f"💾 生成结果已保存至: {OUTPUT_FILE}")
    
    # --- 4. 调用评估器 ---
    print("\n" + "="*40)
    print("🧪 开始运行 EvalPlus 评分...")
    print("="*40)
    
    cmd = [
        "evalplus.evaluate",
        #"--dataset", "humaneval",
        "--dataset", "mbpp",
        "--samples", OUTPUT_FILE,
        "--min-time-limit", "1",
        "--i-just-wanna-run" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ 评估命令执行出错，请检查环境或手动运行。")

if __name__ == "__main__":
    main()