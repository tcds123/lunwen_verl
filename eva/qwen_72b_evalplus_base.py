import sys
import os
import re
import subprocess
from tqdm import tqdm
from evalplus.data import get_human_eval_plus, write_jsonl

# ================= 配置区域 =================
# 输出文件路径
OUTPUT_FILE = "samples_b_72b_humaneval_plus.jsonl"
# ===========================================

# --- 1. 动态导入 generation 目录下的 BModelAPI ---
# 获取当前脚本路径 (eva/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取 generation 目录路径
generation_dir = os.path.join(os.path.dirname(current_dir), 'generation')
# 将其加入 Python 搜索路径
sys.path.append(generation_dir)

try:
    from b_model_api import BModelAPI
    print("✅ 成功导入 BModelAPI")
except ImportError:
    print(f"❌ 无法导入 BModelAPI，请确认 {generation_dir}/b_model_api.py 存在")
    exit(1)

def extract_code(text: str) -> str:
    """清洗 API 返回的 Markdown，提取纯代码"""
    # 优先提取 ```python 代码块
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1)
    
    # 其次提取通用代码块
    pattern_generic = r"```\s*(.*?)\s*```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic: return match_generic.group(1)
    
    # 如果没有代码块，直接返回（Instruct模型偶尔会直接给代码）
    return text

def main():
    print("🚀 启动 B模型 (Qwen-72B) EvalPlus 评估流程")
    
    # --- 1. 检查是否已存在结果，避免重复烧钱 ---
    if os.path.exists(OUTPUT_FILE):
        print(f"\n✨ 发现已存在的生成结果: {OUTPUT_FILE}")
        print("⏭️  跳过生成步骤，直接开始评估...")
    
    else:
        # --- 2. 初始化 API ---
        # 直接使用您 generation/b_model_api.py 里配置好的默认 Key 和 URL
        api = BModelAPI()
        
        # --- 3. 加载 HumanEval+ 数据 ---
        print("📚 正在加载 HumanEval+ 数据集...")
        problems = get_human_eval_plus()
        samples = []
        
        print(f"🔄 开始生成 {len(problems)} 个任务的代码...")
        
        # --- 4. 遍历生成 ---
        for task_id, problem in tqdm(problems.items()):
            raw_prompt = problem["prompt"]
            
            # 构造 Prompt：引导 Instruct 模型补全代码
            # HumanEval 的 prompt 是函数签名，我们需要把它包在 markdown 里让模型续写
            instruct_prompt = (
                f"Please complete the following Python function based on the provided docstring.\n"
                f"Do not include any explanation, just the code.\n\n"
                f"```python\n{raw_prompt}\n```"
            )
            
            # 调用您的 BModelAPI
            # 注意：BModelAPI.generate_code 内部已经封装了 system prompt
            response = api.generate_code(instruct_prompt)
            
            if not response:
                print(f"⚠️ Task {task_id} 生成失败/为空")
                clean_code = ""
            else:
                # 清洗 markdown
                clean_code = extract_code(response)
            
            # 记录结果
            samples.append({
                "task_id": task_id,
                "completion": clean_code
            })

        # --- 5. 保存结果 ---
        write_jsonl(OUTPUT_FILE, samples)
        print(f"💾 生成结果已保存至: {OUTPUT_FILE}")
    
    # --- 6. 调用评估器 ---
    print("\n" + "="*40)
    print("🧪 开始运行 EvalPlus 评分...")
    print("="*40)
    
    cmd = [
        "evalplus.evaluate",
        "--dataset", "humaneval",
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