import requests
import json
import logging
import time
from typing import List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class BModelAPI:
    """基于API调用的B模型替代方案"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化API客户端
        
        Args:
            api_key: API密钥，默认为文档中的密钥
            base_url: API基础URL，默认为文档中的URL
        """
        self.api_key = api_key or "msk-1aeb9a660245cc33b21ee04e67fb88da012808b548b794e07aba1cde1591f8c6"
        self.base_url = base_url or "https://aimpapi.midea.com/t-aigc/aimp-qwen2-5-72b-ascend/v1/chat/completions"
        self.model_name = "Qwen25-72B-Instruct"
        
        # 请求头
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # API调用参数（与当前B模型配置保持一致）
        self.generation_params = {
            "temperature": 0.3,      # 与gen_code.py保持一致
            "top_p": 0.9,           # 与gen_code.py保持一致
            "max_tokens": 2048,     # 与gen_code.py保持一致
            "n": 1,                 # 每次生成一个样本
            "stream": False
        }
        
        logging.info("B模型API客户端初始化完成")

    def smart_combine_prompts(self, generated_prompt: str, original_prompt: str, max_tokens: int = 4000) -> str:
        """
        智能合并提示词，优先保证原始提示词的完整性
        如果tokens超限，优先截断生成提示词
        
        注意：API版本中我们无法准确计算token数量，使用字符数作为近似估计
        """
        # 使用字符数作为token的近似估计（平均1个token≈4个字符）
        estimated_token_ratio = 4
        
        # 首先检查原始提示词是否超限
        original_chars = len(original_prompt)
        estimated_original_tokens = original_chars / estimated_token_ratio
        
        if estimated_original_tokens >= max_tokens:
            # 如果原始提示词本身就超限，需要截断原始提示词
            max_chars = int((max_tokens - 100) * estimated_token_ratio)
            truncated_original = original_prompt[:max_chars]
            return f"[Original (truncated)]{truncated_original}"
        
        # 计算剩余可用tokens
        remaining_tokens = max_tokens - estimated_original_tokens - 100  # 预留100tokens给格式标记
        
        if remaining_tokens <= 0:
            # 如果剩余tokens不足，只保留原始提示词
            return f"[Original]{original_prompt}"
        
        # 估计生成提示词的token数量
        generated_chars = len(generated_prompt)
        estimated_generated_tokens = generated_chars / estimated_token_ratio
        
        if estimated_generated_tokens <= remaining_tokens:
            # 如果生成提示词可以完整保留
            return f"[Generated]{generated_prompt}\n[Original]{original_prompt}"
        else:
            # 需要截断生成提示词
            max_chars = int(remaining_tokens * estimated_token_ratio)
            truncated_generated = generated_prompt[:max_chars]
            return f"[Generated (truncated)]{truncated_generated}\n[Original]{original_prompt}"

    def generate_code(self, prompt: str) -> str:
        """
        调用API生成代码
        """
        # 定义重试参数
        max_retries = 5
        base_wait_time = 5  # 基础等待时间5秒
        
        # [修正1] 必须加上这个循环，retry机制才能生效
        for attempt in range(max_retries):
            try:
                # 构建消息
                messages = [
                    {
                        "role": "system", 
                        "content": "You are a professional code generation assistant. Generate correct and efficient code."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                
                # 构建请求数据
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    **self.generation_params
                }
                
                # 发送请求
                start_time = time.time()
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=60  # 60秒超时
                )
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # 提取生成的代码
                    if "choices" in result and len(result["choices"]) > 0:
                        generated_code = result["choices"][0]["message"]["content"].strip()
                        
                        # 记录使用情况
                        usage = result.get("usage", {})
                        logging.info(f"API调用成功 - 耗时: {response_time:.2f}s, "
                                    f"Tokens: {usage.get('total_tokens', 'N/A')}, "
                                    f"代码长度: {len(generated_code)}字符")
                        
                        return generated_code
                    else:
                        logging.error(f"API响应格式异常: {result}")
                        return ""
                
                # [修正2] 处理 429 限流
                elif response.status_code == 429:
                    logging.warning(f"触发限流 (429)，尝试重试... ({attempt + 1}/{max_retries})")
                    time.sleep(base_wait_time * (2 ** attempt)) # 指数退避
                    continue # 进入下一次循环
                
                # [修正3] 处理其他非200错误
                else:
                    logging.error(f"API调用失败 - 状态码: {response.status_code}")
                    time.sleep(2) 
                    continue

            except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
                logging.error(f"请求异常: {str(e)}，正在重试... ({attempt + 1}/{max_retries})")
                time.sleep(base_wait_time * (2 ** attempt))
                continue
            except Exception as e:
                logging.error(f"未知错误: {str(e)}")
                return "" # 未知错误直接返回
        
        # 循环结束仍未返回，说明失败
        logging.error("达到最大重试次数，生成失败")
        return ""

    def generate_code_from_prompt(self, generated_prompt: str, original_prompt: str = None, k: int = 1, max_tokens: int = 4000) -> List[str]:
        """
        调用B模型生成k个代码样本
        支持智能合并提示词，优先保证原始提示词的完整性
        
        Args:
            generated_prompt: 生成的提示词
            original_prompt: 原始提示词（可选）
            k: 生成样本数量
            max_tokens: 最大token限制
            
        Returns:
            List[str]: 生成的代码列表
        """
        codes = []
        
        # 如果提供了original_prompt，进行智能合并
        if original_prompt is not None:
            combined_prompt = self.smart_combine_prompts(generated_prompt, original_prompt, max_tokens)
        else:
            # 如果没有原始提示词，直接使用生成提示词（但需要检查长度）
            estimated_tokens = len(generated_prompt) / 4  # 近似估计
            if estimated_tokens > max_tokens:
                # 截断生成提示词
                max_chars = int(max_tokens * 4)
                combined_prompt = generated_prompt[:max_chars]
            else:
                combined_prompt = generated_prompt
        
        for i in range(k):
            logging.info(f"正在生成第 {i+1}/{k} 个代码样本")
            code = self.generate_code(combined_prompt)
            codes.append(code.strip())
            
            # 添加短暂延迟避免API限流
            if i < k - 1:
                time.sleep(1)
        
        return codes

    def load_b_model(self, *args, **kwargs):
        """
        兼容性方法，保持与原有接口一致
        对于API版本，此方法不需要实际加载模型
        """
        logging.info("API版本B模型已就绪（无需加载本地模型）")
        return self, None  # 返回self作为model，None作为tokenizer

# 创建全局实例
b_model_api = BModelAPI()

# 兼容性函数，保持与原有b_interface.py相同的接口
def load_b_model(model_path=None, device_id=None):
    """加载B模型（API版本）"""
    return b_model_api, None

def generate_code_from_prompt(model, tokenizer, generated_prompt, original_prompt=None, k=1, max_tokens=4000):
    """生成代码的兼容性接口"""
    return b_model_api.generate_code_from_prompt(generated_prompt, original_prompt, k, max_tokens)

def smart_combine_prompts(model, tokenizer, generated_prompt, original_prompt, max_tokens=4000):
    """智能合并提示词的兼容性接口"""
    return b_model_api.smart_combine_prompts(generated_prompt, original_prompt, max_tokens)

# 测试代码
if __name__ == "__main__":
    try:
        # 测试API连接
        print("正在测试B模型API...")
        api_client = BModelAPI()
        
        # 测试生成代码
        test_prompt = "Write a Python function to calculate the factorial of a number."
        
        codes = api_client.generate_code_from_prompt(
            generated_prompt=test_prompt,
            original_prompt=None,
            k=1
        )
        
        print(f"\n生成的代码数量：{len(codes)}")
        if codes and codes[0]:
            print(f"生成的代码（前200字符）：{codes[0][:200]}...")
        else:
            print("代码生成失败")
            
    except Exception as e:
        print(f"测试出错：{str(e)}")