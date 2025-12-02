import requests
import json
import logging
import time
import os
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional

# ================= 配置区域 =================
# API 配置
API_KEY = "msk-4b8773bf749c892f2c9803aa69ef94b8b96e7cf807da78cbfdf8606ed919adef"
BASE_URL = "https://aimpapi.midea.com/t-aigc/f-devops-qwen3-coder-480b-a35b-instruct/v1/chat/completions"
MODEL_NAME = "f-devops-qwen3-coder-480b-a35b-instruct"  # 通常模型名与路径一致，或者可以留空视平台而定

# 数据路径 (请根据实际情况调整)
INPUT_JSON_PATH = "/data/zhuldz/lunwen/data/CodeEval-Pro/dataset/humaneval_pro.json"
OUTPUT_FILENAME = "humaneval_pro_qwen480b.json"  # 输出文件名，避免覆盖原文件
# ===========================================

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("qwen480b_gen.log"),
        logging.StreamHandler()
    ]
)

class Qwen480BAPI:
    """Qwen3-Code-480B API 客户端"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.base_url = BASE_URL
        self.model_name = MODEL_NAME
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # 生成参数 (保持与之前一致的采样策略)
        self.generation_params = {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 2048,
            "n": 1,
            "stream": False
        }

    def generate(self, prompt: str, retry_count: int = 3) -> str:
        """生成代码，包含重试机制"""
        messages = [
            {
                "role": "system", 
                "content": "You are a professional code generation assistant. Generate correct and efficient code based on the user's request."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        data = {
            "model": self.model_name,
            "messages": messages,
            **self.generation_params
        }
        
        for attempt in range(retry_count):
            try:
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=120  # 480B 可能较慢，增加超时时间
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"].strip()
                        # 尝试提取 Markdown 代码块
                        clean_code = self._extract_code_block(content)
                        logging.info(f"生成成功 ({time.time() - start_time:.2f}s): {len(clean_code)} chars")
                        return clean_code
                
                elif response.status_code == 429:
                    wait_time = 5 * (attempt + 1)
                    logging.warning(f"API限流 (429), 等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logging.error(f"API错误 {response.status_code}: {response.text}")
            
            except Exception as e:
                logging.error(f"请求异常 (尝试 {attempt+1}/{retry_count}): {e}")
                time.sleep(2)
        
        return ""

    def _extract_code_block(self, text: str) -> str:
        """简单清洗：如果包含 ```python ... ``` 则提取，否则返回原文本"""
        import re
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # 备用：尝试提取不带语言标记的代码块
        pattern_generic = r"```\s*(.*?)\s*```"
        match_generic = re.search(pattern_generic, text, re.DOTALL)
        if match_generic:
            return match_generic.group(1).strip()
        return text

def process_dataset():
    """读取数据集并进行生成"""
    if not os.path.exists(INPUT_JSON_PATH):
        logging.error(f"输入文件不存在: {INPUT_JSON_PATH}")
        return

    # 初始化 API
    api = Qwen480BAPI()
    
    # 读取数据
    logging.info(f"正在读取数据: {INPUT_JSON_PATH}")
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict): data = [data]
    
    total = len(data)
    logging.info(f"开始处理 {total} 条数据...")
    
    # 进度条处理
    success_count = 0
    for i, item in enumerate(tqdm(data, desc="Qwen-480B 生成中")):
        # 检查必要的字段
        if "raw_problem" not in item:
            continue
            
        prompt = item["raw_problem"]
        
        # 调用 API 生成
        generated_code = api.generate(prompt)
        
        if generated_code:
            item["predict_code"] = generated_code
            success_count += 1
        else:
            item["predict_code"] = "" # 失败留空
            
        # 每 10 条保存一次，防止意外中断
        if (i + 1) % 10 == 0:
            _save_intermediate(data, i + 1)

    # 最终保存
    current_dir = Path(__file__).parent
    output_path = current_dir / OUTPUT_FILENAME
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    logging.info(f"处理完成! 成功: {success_count}/{total}")
    logging.info(f"结果已保存至: {output_path}")

def _save_intermediate(data, count):
    """保存中间结果"""
    try:
        current_dir = Path(__file__).parent
        temp_path = current_dir / f"{OUTPUT_FILENAME}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    process_dataset()