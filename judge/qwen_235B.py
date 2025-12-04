import requests
import json
from typing import Dict, Optional, Any
import time
import uuid
from pathlib import Path


# --------------------------
# 从txt文件加载提示词模板（模块级常量）
# --------------------------
def load_prompt_template() -> str:
    """从同目录的txt文件加载C模型评判提示词模板"""
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "c_model_prompt.txt"  # 提示词文件路径

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"提示词模板文件不存在：{prompt_file}，请检查文件名是否正确")
    except Exception as e:
        raise Exception(f"读取提示词模板失败：{str(e)}")


# 加载模板（仅在模块初始化时执行一次）
C_MODEL_PROMPT_TEMPLATE = load_prompt_template()


# --------------------------
# C模型评判类（适配Qwen3 API）
# --------------------------
class CModelJudge:
    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        self.timeout = 60  # API超时时间（秒）
        self.retry_times = 3  # 重试次数


    def _send_request(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """发送API请求，带重试机制"""
        retry_count = 0
        while retry_count < self.retry_times:
            try:
                response = requests.post(
                    url=self.api_url,
                    headers=self.headers,
                    json=request_body,
                    timeout=self.timeout,
                    stream=False  # 强制非流式，确保返回完整响应
                )
                print(f"API请求成功，状态码：{response.status_code}")
                response.raise_for_status()  # 触发4xx/5xx错误
                return response.json()  # 返回解析后的JSON

            except requests.exceptions.RequestException as e:
                retry_count += 1
                remaining = self.retry_times - retry_count
                wait_time = 2 ** retry_count  # 指数退避重试
                print(f"请求失败（{retry_count}/{self.retry_times}）：{str(e)}")
                if remaining > 0:
                    print(f"将在{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"超过最大重试次数（{self.retry_times}次）：{str(e)}")


    def judge(
        self,
        model_c: str,
        prompt: str,  # A模型生成的提示词
        generated_code: str,  # B模型生成的代码
        code_ground_truth: str,  # 代码真值
        b_model_input: str,  # 新增：B模型的输入
        user_prompt: str
    ) -> Dict[str, Any]:
        """调用C模型进行评判，返回结构化结果"""
        # 1. 填充提示词模板
        judge_prompt = C_MODEL_PROMPT_TEMPLATE.format(
            prompt=prompt,
            generated_code=generated_code,
            code_ground_truth=code_ground_truth,
            b_model_input=b_model_input,  # 新增：B模型的输入
            user_prompt=user_prompt
        )

        # 2. 构造符合Qwen3 API要求的请求体
        request_body = {
            "model": model_c,
            "messages": [{"role": "user", "content": judge_prompt}],
            "stream": False,  # 非流式响应（必须，否则返回分段数据）
            "chat_template_kwargs": {"enable_thinking": False},  # C模型无需推理过程
            "temperature": 0.2,  # 低随机性，确保评判稳定
            "max_tokens": 500  # 限制响应长度，避免冗余
        }

        # 3. 发送请求
        request_id = f"judge-{uuid.uuid4().hex[:10]}"
        print(f"\n=== C模型评判请求 ===")
        print(f"请求ID：{request_id}")
        print(f"提示词（前500字符）：{judge_prompt[:500]}...")

        try:
            response_json = self._send_request(request_body)
            return self._parse_judge_result(response_json, request_id)
        except Exception as e:
            return {
                "request_id": request_id,
                "valid": False,
                "error": f"评判请求失败：{str(e)}",
                "raw_response": None
            }


    def _parse_judge_result(self, response: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """解析Qwen3 API的响应，提取C模型评判结果"""
        try:
            # 打印原始响应（调试用）
            print(f"\n=== 原始API响应（Qwen3格式） ===")
            print(json.dumps(response, ensure_ascii=False, indent=2))

            # 4. 按API文档提取C模型输出内容（严格路径）
            if "choices" not in response:
                raise ValueError("响应中缺少'choices'字段（不符合Qwen3 API格式）")
            
            choices = response["choices"]
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError("'choices'字段不是非空列表（不符合Qwen3 API格式）")
            
            first_choice = choices[0]
            if "message" not in first_choice:
                raise ValueError("第一个choice中缺少'message'字段（不符合Qwen3 API格式）")
            
            message = first_choice["message"]
            if "content" not in message:
                raise ValueError("'message'中缺少'content'字段（C模型未返回评判内容）")
            
            # 提取并处理评判文本
            judge_text = message["content"]
            if not isinstance(judge_text, str):
                judge_text = str(judge_text)  # 强制转为字符串
            judge_text = judge_text.strip()

            if not judge_text:
                raise ValueError("C模型返回的'content'为空（无有效评判内容）")

            # 5. 提取JSON部分（按提示词要求，C模型应输出JSON）
            json_start = judge_text.find("{")
            json_end = judge_text.rfind("}") + 1
            if json_start == -1 or json_end == 0:
                raise ValueError(f"C模型输出不包含有效JSON，原始文本：{judge_text[:200]}...")
            
            # 解析JSON
            try:
                judge_json = json.loads(judge_text[json_start:json_end])
            except json.JSONDecodeError as e:
                raise ValueError(f"C模型输出的JSON格式错误：{str(e)}，原始JSON部分：{judge_text[json_start:json_end]}")

            # 6. 验证必要字段
            required_fields = ["total_score", "match_prompt", "score_details", "reason"]
            missing_fields = [f for f in required_fields if f not in judge_json]
            if missing_fields:
                raise ValueError(f"评判结果缺少必要字段：{missing_fields}")

            # 7. 验证分数格式 - 修复：允许浮点数，因为提示词模板要求0.0~10.0的连续值
            if not isinstance(judge_json["total_score"], (int, float)):
                raise ValueError(f"总分必须是数字（整数或浮点数），实际为：{type(judge_json['total_score'])}")

            return {
                "request_id": request_id,
                "valid": True,
                "result": judge_json,
                "raw_response": response
            }

        except Exception as e:
            print(f"解析评判结果失败：{str(e)}")
            return {
                "request_id": request_id,
                "valid": False,
                "error": str(e),
                "raw_response": response
            }


# --------------------------
# 批量处理JSON文件的函数（核心修改）
# --------------------------
def process_json_evaluation(
    source_json_path: str,
    judge: CModelJudge,
    model_c: str,
    sleep_time: int = 3  # 每条数据评判后的休眠时间（秒）
):
    """读取源JSON文件，逐条评分后在当前目录生成新文件（源文件不变）"""
    # 1. 读取源JSON文件（仅读取，不修改）
    source_path = Path(source_json_path)
    if not source_path.exists():
        raise FileNotFoundError(f"源JSON文件不存在：{source_json_path}")
    
    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 加载数据到内存（不修改源文件）
    
    # 统一数据格式为列表
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"JSON数据必须是列表或字典，实际为：{type(data)}")
    
    total = len(data)
    print(f"\n=== 开始批量评分 ===")
    print(f"总数据量：{total}条，每条评分后休眠{sleep_time}秒")
    print(f"源文件将保持不变，结果将保存至当前脚本目录")

    # 2. 逐条处理：为每个样本添加evaluation_result键（API返回内容）
    for i, item in enumerate(data, 1):
        print(f"\n--- 处理第{i}/{total}条数据 ---")
        
        # 检查必要字段
        required_keys = ["raw_problem", "predict_code", "raw_solution"]
        missing_keys = [k for k in required_keys if k not in item]
        if missing_keys:
            print(f"跳过：缺少必要字段{missing_keys}")
            item["evaluation_result"] = {"valid": False, "error": f"缺少字段：{missing_keys}"}
            continue
        
        # 提取字段内容
        prompt = item["raw_problem"].strip()
        generated_code = item["predict_code"].strip()
        ground_truth = item["raw_solution"].strip()
        
        # 检查内容是否为空
        if not prompt:
            print("跳过：raw_problem为空")
            item["evaluation_result"] = {"valid": False, "error": "raw_problem为空"}
            continue
        if not generated_code:
            print("跳过：predict_code为空")
            item["evaluation_result"] = {"valid": False, "error": "predict_code为空"}
            continue
        if not ground_truth:
            print("跳过：raw_solution为空")
            item["evaluation_result"] = {"valid": False, "error": "raw_solution为空"}
            continue
        
        try:
            eval_result = judge.judge(
                model_c=model_c,
                prompt=prompt,
                generated_code=generated_code,
                code_ground_truth=ground_truth
            )
            item["evaluation_result"] = eval_result  # 添加API返回内容作为新键
            print(f"第{i}条评分完成（请求ID：{eval_result['request_id']}）")
        except Exception as e:
            error_msg = f"评分失败：{str(e)}"
            print(error_msg)
            item["evaluation_result"] = {"valid": False, "error": error_msg}
        
        # 评分后休眠（避免API请求过于频繁）
        if i < total:  # 最后一条不休眠
            print(f"休眠{sleep_time}秒...")
            time.sleep(sleep_time)
    
    # 3. 在当前脚本目录生成新文件（源文件不变，不备份）
    current_dir = Path(__file__).parent  # 当前脚本所在目录
    source_filename = source_path.name  # 源文件的文件名（如"humaneval_pro.json"）
    new_filename = f"evaluated_{source_filename}"  # 新文件名（添加前缀标识）
    new_file_path = current_dir / new_filename  # 新文件完整路径

    # 写入新文件
    with open(new_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n所有数据处理完成，新文件已生成：{new_file_path}")


if __name__ == "__main__":
    # 配置参数
    API_URL = "https://aimpapi.midea.com/t-aigc/aimp-qwen3-235b-a22b/v1/chat/completions"
    AUTH_TOKEN = "msk-8a895e7fa53a8785f9cc4dc0364fae9064ccc540bbd419b5ba7cde8340ec2af8"
    MODEL_C = "/model/qwen3-235b-a22b"  # C模型标识
    SOURCE_JSON_PATH = "/data/zhuldz/lunwen/generation/humaneval_pro.json"  # 替换为你的源JSON文件路径
    SLEEP_TIME = 5  # 每条数据评分后的休眠时间（秒）

    try:
        c_judge = CModelJudge(API_URL, AUTH_TOKEN)
    except Exception as e:
        print(f"初始化评判器失败：{str(e)}")
        exit(1)

    # 批量处理并生成新文件
    try:
        process_json_evaluation(
            source_json_path=SOURCE_JSON_PATH,
            judge=c_judge,
            model_c=MODEL_C,
            sleep_time=SLEEP_TIME
        )
    except Exception as e:
        print(f"处理失败：{str(e)}")
        exit(1)